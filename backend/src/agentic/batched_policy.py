"""One GPU model, dynamic microbatches, and separate per-episode policy audits."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import time

from agentic.chat_template_contract import render_agent_tool_prompt
from agentic.local_policy import LocalCheckpointAgentPolicy, parse_local_tool_call
from agentic.loop import PolicyAction
from agentic.policy import PolicyOutputError
from agentic.policy_actions import controller_override_attempt, validate_policy_arguments_for_state
from core.inference_metrics import InferenceMetrics


def decode_action(output, audit, allowed, capability, checkpoint, prompt_tokens, count, latency):
    """Keep the native parser and current state-specific argument authority."""
    metrics = InferenceMetrics(model=checkpoint, backend="transformers", thinking_mode="disabled",
                               prompt_tokens=prompt_tokens, completion_tokens=count,
                               request_latency_ms=round(latency, 3))
    audit.update(raw_output=output, raw_output_sha256=hashlib.sha256(output.encode()).hexdigest(),
                 inference_metrics=metrics.model_dump(mode="json"),
                 tool_call_open_count=output.count("<tool_call>"), tool_call_close_count=output.count("</tool_call>"),
                 parser_contract="exact-single-tool-call-envelope-or-json.v1")
    name, arguments = parse_local_tool_call(output)
    raw = dict(arguments)
    override = controller_override_attempt(name, raw)
    audit.update(parsed_action=name, model_raw_arguments=raw,
                 controller_override_attempt=override, model_contract_compliant=not override)
    if name not in allowed:
        raise PolicyOutputError(f"local policy proposed {name}, allowed: {allowed}", code="ACTION_NOT_ALLOWED", raw_output=output)
    try:
        validated = validate_policy_arguments_for_state(name, arguments, capability=capability or {})
    except ValueError as exc:
        raise PolicyOutputError(str(exc), code="POLICY_ARGUMENT_INVALID",
                                detail_code=getattr(exc, "rejection_code", None), raw_output=output) from exc
    return PolicyAction(action=name, arguments=validated, model_arguments=raw,
                        controller_override_attempt=override, model_contract_compliant=not override,
                        token_usage=count, inference_metrics=metrics)


@dataclass
class _Request:
    messages: list
    tools: list
    allowed: list
    capability: dict
    future: asyncio.Future
    submitted: float


class BatchedCheckpointEngine:
    def __init__(self, model_policy, *, batch_size=4, wait_ms=15, max_padded_tokens=24000):
        if batch_size < 1 or wait_ms < 0 or max_padded_tokens < 1:
            raise ValueError("Invalid batch limits")
        if model_policy.do_sample or model_policy.structured_decoding_mode != "native":
            raise ValueError("This measured batch path supports greedy native decoding only")
        self.policy = model_policy
        self.batch_size, self.wait_ms, self.max_padded_tokens = batch_size, wait_ms, max_padded_tokens
        self.queue = asyncio.Queue()
        self.worker = None
        self.closed = False
        self.batches = []

    def session(self):
        return BatchedSessionPolicy(self)

    async def submit(self, messages, tools, allowed, capability):
        if self.closed:
            raise RuntimeError("Batch engine is closed")
        if self.worker is None:
            self.worker = asyncio.create_task(self._serve())
        future = asyncio.get_running_loop().create_future()
        await self.queue.put(_Request(messages, tools, allowed, capability or {}, future, time.perf_counter()))
        return await future

    async def _serve(self):
        while True:
            first = await self.queue.get()
            if first is None:
                return
            requests = [first]
            if self.batch_size > 1:
                await asyncio.sleep(self.wait_ms / 1000)
            while len(requests) < self.batch_size and not self.queue.empty():
                requests.append(self.queue.get_nowait())
            requests = [r for r in requests if not r.future.cancelled()]
            if not requests:
                continue
            try:
                results = await asyncio.to_thread(self._generate, requests)
                for request, result in zip(requests, results, strict=True):
                    if not request.future.done():
                        request.future.set_result(result)
            except Exception as exc:
                for request in requests:
                    if not request.future.done():
                        request.future.set_exception(exc)

    def _generate(self, requests):
        policy = self.policy
        encoded = [render_agent_tool_prompt(policy.tokenizer, r.messages, tools=r.tools,
                   add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt") for r in requests]
        ids = [e["input_ids"][0].tolist() for e in encoded]
        return self._generate_ids(requests, ids)

    def _generate_ids(self, requests, ids):
        p = self.policy
        torch = p._torch
        width = max(map(len, ids))
        if width * len(ids) > self.max_padded_tokens:
            if len(ids) == 1:
                raise ValueError("Prompt exceeds configured token budget; no silent truncation")
            half = len(ids) // 2
            return self._generate_ids(requests[:half], ids[:half]) + self._generate_ids(requests[half:], ids[half:])
        device = next(p.model.parameters()).device
        tokens = torch.full((len(ids), width), p.tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(tokens)
        for i, row in enumerate(ids):
            tokens[i, width-len(row):] = torch.tensor(row, dtype=torch.long, device=device)
            mask[i, width-len(row):] = 1
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                generated = p.model.generate(input_ids=tokens, attention_mask=mask, do_sample=False,
                    max_new_tokens=p.max_new_tokens, pad_token_id=p.tokenizer.pad_token_id,
                    eos_token_id=p.tokenizer.eos_token_id)
        except torch.cuda.OutOfMemoryError:
            del tokens, mask
            torch.cuda.empty_cache()
            if len(ids) == 1:
                raise
            half = len(ids) // 2
            self.batches.append({"batch_size":len(ids), "event":"oom_split", "padded_tokens":width*len(ids)})
            return self._generate_ids(requests[:half], ids[:half]) + self._generate_ids(requests[half:], ids[half:])
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        suffixes = generated[:, width:].detach().cpu().tolist()
        batch_id = len(self.batches)
        self.batches.append({"batch_id":batch_id, "batch_size":len(ids), "prompt_tokens":list(map(len,ids)),
                             "padded_tokens":width*len(ids), "generation_ms":elapsed})
        results = []
        for request, row, suffix in zip(requests, ids, suffixes, strict=True):
            # Batched generation pads already-completed rows; retain exactly the
            # first EOS and discard only padding emitted after it.
            if p.tokenizer.eos_token_id in suffix:
                suffix = suffix[:suffix.index(p.tokenizer.eos_token_id)+1]
            output = p.tokenizer.decode(suffix, skip_special_tokens=False)
            audit = {"schema_version":"local-policy-generation-audit.batch-v1",
                "prompt_input_ids_sha256":hashlib.sha256(json.dumps(row,separators=(',',':')).encode()).hexdigest(),
                "batch_id":batch_id, "batch_size":len(ids), "generation_ms":elapsed,
                "request_wall_ms":(time.perf_counter()-request.submitted)*1000}
            try:
                action = decode_action(output, audit, request.allowed, request.capability, p.checkpoint,
                                       len(row), len(suffix), audit["request_wall_ms"])
                results.append((action, audit, None))
            except PolicyOutputError as exc:
                results.append((None, audit, exc))
        return results

    async def aclose(self):
        # Called after all episode tasks finish; no in-flight request is dropped.
        self.closed = True
        if self.worker:
            await self.queue.put(None)
            await self.worker


class BatchedSessionPolicy(LocalCheckpointAgentPolicy):
    def __init__(self, engine):
        self.engine = engine
        self._last_generation_audit = None

    async def propose_from_history(self, messages, *, tools, allowed_actions, capability=None):
        self._last_generation_audit = None
        action, audit, error = await self.engine.submit(messages, tools, allowed_actions, capability)
        self._last_generation_audit = audit
        if error:
            raise error
        return action
