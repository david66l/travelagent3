"""Local Hugging Face checkpoint adapter for the production Agent Loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Literal

from agentic.chat_template_contract import (
    AGENT_CHAT_TEMPLATE_KWARGS,
    AGENT_CHAT_TEMPLATE_SHA256,
    AGENT_RENDER_PROTOCOL_VERSION,
    install_agent_chat_template,
    render_agent_tool_prompt,
)
from agentic.loop import PolicyAction, PolicyContext
from agentic.policy import (
    AGENT_TOOL_POLICY_SYSTEM_PROMPT,
    PolicyOutputError,
    constrain_policy_context,
    policy_prompt_payload,
)
from agentic.policy_actions import (
    controller_override_attempt,
    policy_action_schemas_for_state,
    policy_tool_call_json_schema_for_state,
    validate_policy_arguments,
    validate_policy_arguments_for_state,
)
from core.inference_metrics import InferenceMetrics


_TOOL_CALL_ENVELOPE_PATTERN = re.compile(
    r"\A\s*<tool_call>\s*(?P<payload>\{.*\})\s*</tool_call>\s*"
    r"(?:(?:<\|im_end\|>|<\|endoftext\|>)\s*)?\Z",
    re.DOTALL,
)
StructuredDecodingMode = Literal["native", "json_schema", "qwen_tool_envelope"]


def _tokenizer_compatibility_kwargs(checkpoint: str | Path) -> dict[str, Any]:
    """Normalize legacy Qwen tokenizer metadata without editing a checkpoint."""

    config_path = Path(checkpoint) / "tokenizer_config.json"
    if not config_path.is_file():
        return {}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    # Transformers 4.57 expects a named mapping here, while some Qwen adapter
    # saves contain the older list form. The tokens already live in the saved
    # added-token table, so an empty mapping preserves them and avoids the
    # incompatible model-specific alias parser.
    return (
        {"extra_special_tokens": {}}
        if isinstance(config.get("extra_special_tokens"), list)
        else {}
    )


def parse_local_tool_call(text: str) -> tuple[str, dict[str, Any]]:
    """Parse Qwen-style native tool output without accepting prose as an action."""
    match = _TOOL_CALL_ENVELOPE_PATTERN.fullmatch(text)
    candidate = match.group("payload") if match else text.strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise PolicyOutputError(
            "local policy did not emit one valid tool call",
            code="TOOL_CALL_PARSE_ERROR",
            raw_output=text,
        ) from exc
    if not isinstance(payload, dict):
        raise PolicyOutputError(
            "local policy tool call must be one JSON object",
            code="TOOL_CALL_SHAPE_ERROR",
            raw_output=text,
        )
    name_keys = [key for key in ("name", "action") if key in payload]
    expected_keys = {name_keys[0], "arguments"} if len(name_keys) == 1 else set()
    name = str(payload.get(name_keys[0]) or "").strip() if name_keys else ""
    arguments = payload.get("arguments")
    if not name or not isinstance(arguments, dict) or set(payload) != expected_keys:
        raise PolicyOutputError(
            "local policy tool call must contain exactly one name and arguments object",
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
        from peft import PeftConfig, PeftModel
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
        self._last_generation_audit: dict[str, Any] | None = None
        if do_sample and temperature <= 0:
            raise ValueError("sampled policy temperature must be positive")
        self._torch = torch
        torch.manual_seed(seed)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            revision=revision,
            trust_remote_code=False,
            **_tokenizer_compatibility_kwargs(checkpoint),
        )
        if not self.tokenizer.chat_template:
            raise RuntimeError("local checkpoint must provide a native chat template")
        install_agent_chat_template(self.tokenizer)
        self.render_protocol_version = AGENT_RENDER_PROTOCOL_VERSION
        self.chat_template_sha256 = AGENT_CHAT_TEMPLATE_SHA256
        self.chat_template_kwargs = dict(AGENT_CHAT_TEMPLATE_KWARGS)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        is_adapter = (Path(checkpoint) / "adapter_config.json").is_file()
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
        model_kwargs = {
            "device_map": "auto",
            "dtype": dtype,
            "quantization_config": quantization_config,
            "trust_remote_code": False,
        }
        if is_adapter:
            # AutoPeftModel performs a second implicit tokenizer load solely to
            # check embedding size. That path cannot receive tokenizer kwargs
            # and crashes on legacy Qwen list-form ``extra_special_tokens``.
            # Loading the identical base model and adapter explicitly avoids
            # that unrelated tokenizer side effect while preserving weights.
            peft_config = PeftConfig.from_pretrained(
                checkpoint,
                revision=revision,
            )
            base_revision = getattr(peft_config, "revision", None)
            base_model = AutoModelForCausalLM.from_pretrained(
                peft_config.base_model_name_or_path,
                revision=base_revision,
                **model_kwargs,
            )
            self.model = PeftModel.from_pretrained(
                base_model,
                checkpoint,
                config=peft_config,
                revision=revision,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                revision=revision,
                **model_kwargs,
            )
        self.model.eval()
        self._structured_backend: Any | None = None
        self._structured_processor_cache: dict[str, Any] = {}
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
        tools = policy_action_schemas_for_state(
            context.allowed_actions,
            capability=context.capability,
        )
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
            capability=context.capability,
        )

    async def propose_from_history(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        allowed_actions: list[str],
        capability: dict[str, Any] | None = None,
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
                capability,
            )

    def set_rollout_seed(self, seed: int) -> None:
        """Reset sampling RNG once per rollout for paired checkpoint evaluation."""
        self._torch.manual_seed(seed)
        self._torch.cuda.manual_seed_all(seed)

    @property
    def last_generation_audit(self) -> dict[str, Any] | None:
        """Expose the latest raw local completion for offline audit evidence."""
        return dict(self._last_generation_audit) if self._last_generation_audit else None

    def _propose_from_history_sync(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        allowed_actions: list[str],
        capability: dict[str, Any] | None,
    ) -> PolicyAction:
        self._last_generation_audit = None
        encoded = render_agent_tool_prompt(
            self.tokenizer,
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
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
                    [self._structured_logits_processor(allowed_actions, capability=capability)]
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
        inference_metrics = InferenceMetrics(
            model=self.checkpoint,
            backend="transformers",
            thinking_mode="disabled",
            prompt_tokens=prompt_tokens,
            completion_tokens=int(completion_ids.numel()),
            request_latency_ms=round(request_latency_ms, 3),
        )
        prompt_token_ids = encoded["input_ids"][0].detach().cpu().tolist()
        prompt_input_ids_sha256 = hashlib.sha256(
            json.dumps(prompt_token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        # Native Qwen tool calls are extracted from their <tool_call> envelope,
        # so trailing control tokens never reach the JSON parser. Structured
        # decoding intentionally emits plain JSON; strip tokenizer control
        # tokens there while leaving the native audit surface unchanged.
        output = self.tokenizer.decode(
            completion_ids,
            skip_special_tokens=self.structured_decoding_mode == "json_schema",
        )
        self._last_generation_audit = {
            "schema_version": "local-policy-generation-audit.v2",
            "prompt_input_ids_sha256": prompt_input_ids_sha256,
            "raw_output": output,
            "raw_output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
            "inference_metrics": inference_metrics.model_dump(mode="json"),
            "tool_call_open_count": output.count("<tool_call>"),
            "tool_call_close_count": output.count("</tool_call>"),
            "parser_contract": "exact-single-tool-call-envelope-or-json.v1",
        }
        action, arguments = parse_local_tool_call(output)
        raw_arguments = dict(arguments)
        override_attempt = controller_override_attempt(action, raw_arguments)
        self._last_generation_audit.update(
            {
                "parsed_action": action,
                "model_raw_arguments": raw_arguments,
                "controller_override_attempt": override_attempt,
                "model_contract_compliant": not override_attempt,
            }
        )
        if action not in allowed_actions:
            raise PolicyOutputError(
                f"local policy proposed {action}, allowed: {allowed_actions}",
                code="ACTION_NOT_ALLOWED",
                raw_output=output,
            )
        try:
            validated = (
                validate_policy_arguments_for_state(
                    action,
                    arguments,
                    capability=capability,
                )
                if capability is not None
                else validate_policy_arguments(action, arguments)
            )
        except ValueError as exc:
            raise PolicyOutputError(
                str(exc),
                code="POLICY_ARGUMENT_INVALID",
                detail_code=getattr(exc, "rejection_code", None),
                raw_output=output,
            ) from exc
        return PolicyAction(
            action=action,
            arguments=validated,
            model_arguments=raw_arguments,
            controller_override_attempt=override_attempt,
            model_contract_compliant=not override_attempt,
            token_usage=int(completion_ids.numel()),
            inference_metrics=inference_metrics,
        )

    def _structured_logits_processor(
        self,
        allowed_actions: list[str],
        *,
        capability: dict[str, Any] | None = None,
    ) -> Any:
        """Compile one state-scoped JSON grammar without changing the prompt."""
        if self._structured_backend is None:
            raise RuntimeError("structured decoding is not enabled")
        cache = getattr(self, "_structured_processor_cache", None)
        if cache is None:
            cache = self._structured_processor_cache = {}
        schema_text = json.dumps(
            policy_tool_call_json_schema_for_state(
                allowed_actions,
                capability=capability or {},
            ),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        cache_key = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()
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
