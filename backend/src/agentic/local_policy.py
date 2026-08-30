"""Local Hugging Face checkpoint adapter for the production Agent Loop."""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, Literal

from agentic.loop import PolicyAction, PolicyContext
from agentic.policy import (
    AGENT_TOOL_POLICY_SYSTEM_PROMPT,
    PolicyOutputError,
    constrain_policy_context,
    policy_prompt_payload,
)
from agentic.policy_actions import (
    policy_action_schemas,
    policy_tool_call_json_schema,
    validate_policy_arguments,
)
from core.inference_metrics import InferenceMetrics


_TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
StructuredDecodingMode = Literal["native", "json_schema", "qwen_tool_envelope"]


def parse_local_tool_call(text: str) -> tuple[str, dict[str, Any]]:
    """Parse Qwen-style native tool output without accepting prose as an action."""
    match = _TOOL_CALL_PATTERN.search(text)
    candidate = match.group(1) if match else text.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise PolicyOutputError(
            "local policy did not emit one valid tool call",
            code="TOOL_CALL_PARSE_ERROR",
            raw_output=text,
        ) from exc
    name = str(payload.get("name") or payload.get("action") or "").strip()
    arguments = payload.get("arguments") or {}
    if not name or not isinstance(arguments, dict):
        raise PolicyOutputError(
            "local policy tool call is missing name or arguments",
            code="TOOL_CALL_SHAPE_ERROR",
            raw_output=text,
        )
    return name, arguments


class LocalCheckpointAgentPolicy:
    """Run a base model or PEFT adapter through the same bounded action contract."""

    def __init__(
        self,
        checkpoint: str,
        *,
        max_new_tokens: int = 192,
        seed: int = 42,
        do_sample: bool = False,
        temperature: float = 0.8,
        load_in_4bit: bool = False,
        structured_decoding: StructuredDecodingMode | bool = "native",
        revision: str | None = None,
    ) -> None:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError("local checkpoint policy requires a CUDA GPU")
        self.checkpoint = checkpoint
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists() and not (
            revision and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision)
        ):
            raise ValueError("remote Hugging Face checkpoints require an immutable commit revision")
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        # Preserve the first experimental bool API for reproducible reports,
        # while naming each protocol explicitly in new callers.
        self.structured_decoding_mode: StructuredDecodingMode = (
            "json_schema"
            if structured_decoding is True
            else "native"
            if structured_decoding is False
            else structured_decoding
        )
        if self.structured_decoding_mode not in {
            "native",
            "json_schema",
            "qwen_tool_envelope",
        }:
            raise ValueError(f"unknown structured decoding mode: {self.structured_decoding_mode}")
        self._generation_lock = asyncio.Lock()
        if do_sample and temperature <= 0:
            raise ValueError("sampled policy temperature must be positive")
        self._torch = torch
        torch.manual_seed(seed)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            revision=revision,
            trust_remote_code=False,
        )
        if not self.tokenizer.chat_template:
            raise RuntimeError("local checkpoint must provide a native chat template")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        is_adapter = (Path(checkpoint) / "adapter_config.json").is_file()
        loader = AutoPeftModelForCausalLM if is_adapter else AutoModelForCausalLM
        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            if load_in_4bit
            else None
        )
        self.model = loader.from_pretrained(
            checkpoint,
            device_map="auto",
            dtype=dtype,
            quantization_config=quantization_config,
            trust_remote_code=False,
            revision=revision,
        )
        self.model.eval()
        self._structured_backend: Any | None = None
        self._structured_processor_cache: dict[tuple[str, ...], Any] = {}
        if self.structured_decoding_mode != "native":
            try:
                from outlines import from_transformers
                from outlines.backends import OutlinesCoreBackend
            except ImportError as exc:
                raise RuntimeError(
                    "structured local decoding requires the agentic-training dependency 'outlines'"
                ) from exc
            outlines_model = from_transformers(self.model, self.tokenizer)
            self._structured_backend = OutlinesCoreBackend(outlines_model)

    async def propose(self, context: PolicyContext) -> PolicyAction:
        context = constrain_policy_context(context)
        if not context.allowed_actions:
            raise PolicyOutputError(
                "controller supplied no allowed actions",
                code="CONTROLLER_ALLOWLIST_EMPTY",
            )
        tools = policy_action_schemas(context.allowed_actions)
        messages = [
            {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    policy_prompt_payload(context),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ]
        return await self.propose_from_history(
            messages,
            tools=tools,
            allowed_actions=context.allowed_actions,
        )

    async def propose_from_history(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        allowed_actions: list[str],
    ) -> PolicyAction:
        """Sample one action from the same assistant/tool history TRL sees."""
        if not allowed_actions:
            raise PolicyOutputError(
                "controller supplied no allowed actions",
                code="CONTROLLER_ALLOWLIST_EMPTY",
            )
        # One shared checkpoint is cached per worker process. Serialize GPU
        # generation so concurrent requests cannot race one transformers model
        # instance or multiply its peak memory use.
        async with self._generation_lock:
            return await asyncio.to_thread(
                self._propose_from_history_sync,
                messages,
                tools,
                allowed_actions,
            )

    def set_rollout_seed(self, seed: int) -> None:
        """Reset sampling RNG once per rollout for paired checkpoint evaluation."""
        self._torch.manual_seed(seed)
        self._torch.cuda.manual_seed_all(seed)

    def _propose_from_history_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        allowed_actions: list[str],
    ) -> PolicyAction:
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        generation_started = time.perf_counter()
        with self._torch.inference_mode():
            sampling = (
                {"do_sample": True, "temperature": self.temperature}
                if self.do_sample
                else {"do_sample": False}
            )
            generation_kwargs: dict[str, Any] = {}
            if self.structured_decoding_mode != "native":
                from transformers import LogitsProcessorList

                generation_kwargs["logits_processor"] = LogitsProcessorList(
                    [self._structured_logits_processor(allowed_actions)]
                )
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **generation_kwargs,
                **sampling,
            )
        request_latency_ms = (time.perf_counter() - generation_started) * 1000
        completion_ids = generated[0, prompt_tokens:]
        # Native Qwen tool calls are extracted from their <tool_call> envelope,
        # so trailing control tokens never reach the JSON parser. Structured
        # decoding intentionally emits plain JSON; strip tokenizer control
        # tokens there while leaving the native audit surface unchanged.
        output = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=self.structured_decoding_mode == "json_schema",
        )
        action, arguments = parse_local_tool_call(output)
        if action not in allowed_actions:
            raise PolicyOutputError(
                f"local policy proposed {action}, allowed: {allowed_actions}",
                code="ACTION_NOT_ALLOWED",
                raw_output=output,
            )
        try:
            validated = validate_policy_arguments(action, arguments)
        except ValueError as exc:
            raise PolicyOutputError(
                str(exc),
                code="POLICY_ARGUMENT_INVALID",
                raw_output=output,
            ) from exc
        return PolicyAction(
            action=action,
            arguments=validated,
            token_usage=int(completion_ids.numel()),
            inference_metrics=InferenceMetrics(
                model=self.checkpoint,
                backend="transformers",
                thinking_mode="disabled",
                prompt_tokens=prompt_tokens,
                completion_tokens=int(completion_ids.numel()),
                request_latency_ms=round(request_latency_ms, 3),
            ),
        )

    def _structured_logits_processor(self, allowed_actions: list[str]) -> Any:
        """Compile one state-scoped JSON grammar without changing the prompt."""
        if self._structured_backend is None:
            raise RuntimeError("structured decoding is not enabled")
        cache = getattr(self, "_structured_processor_cache", None)
        if cache is None:
            cache = self._structured_processor_cache = {}
        cache_key = tuple(allowed_actions)
        cached = cache.get(cache_key)
        if cached is not None:
            # Outlines processors keep one guide cursor per generation.  The
            # compiled vocabulary index is immutable and expensive, so reuse
            # the processor only after resetting its per-request cursor.  GPU
            # generation is serialized by ``_generation_lock`` above.
            reset = getattr(cached, "reset", None)
            if callable(reset):
                reset()
            return cached
        schema_text = json.dumps(
            policy_tool_call_json_schema(allowed_actions),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if self.structured_decoding_mode == "json_schema":
            processor = self._structured_backend.get_json_schema_logits_processor(schema_text)
        elif self.structured_decoding_mode == "qwen_tool_envelope":
            from outlines_core.json_schema import build_regex_from_schema

            json_regex = build_regex_from_schema(schema_text)
            envelope_regex = r"<tool_call>\n(" + json_regex + r")\n</tool_call>"
            processor = self._structured_backend.get_regex_logits_processor(envelope_regex)
        else:
            raise RuntimeError("structured decoding is not enabled")
        cache[cache_key] = processor
        return processor

    def close(self) -> None:
        """Release one checkpoint before the next fixed-set evaluation arm."""
        del self.model
        self._torch.cuda.empty_cache()


__all__ = [
    "LocalCheckpointAgentPolicy",
    "StructuredDecodingMode",
    "parse_local_tool_call",
]
