"""Pure helpers for reproducible OpenAI-compatible streaming benchmarks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from core.inference_metrics import InferenceMetrics, summarize_inference_metrics


class VLLMBenchmarkCase(BaseModel):
    case_id: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    expected_action: str | None = None
    expected_arguments: dict[str, Any] | None = None
    family: str = "unspecified"


class StreamAccumulator:
    """Collect usage and the first meaningful streamed model delta."""

    def __init__(self) -> None:
        self.ttft_ms: float | None = None
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_prompt_tokens = 0
        self.finish_reason: str | None = None
        self.tool_names: dict[int, str] = {}
        self.tool_arguments: dict[int, str] = {}
        self.content = ""
        self.reasoning_content = ""

    def consume(self, chunk: dict[str, Any], *, elapsed_ms: float) -> None:
        usage = chunk.get("usage") or {}
        self.prompt_tokens = int(usage.get("prompt_tokens") or self.prompt_tokens)
        self.completion_tokens = int(usage.get("completion_tokens") or self.completion_tokens)
        details = usage.get("prompt_tokens_details") or {}
        self.cached_prompt_tokens = int(details.get("cached_tokens") or self.cached_prompt_tokens)
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                self.finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") or {}
            meaningful = bool(
                delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")
            )
            if meaningful and self.ttft_ms is None:
                self.ttft_ms = elapsed_ms
            self.content += str(delta.get("content") or "")
            self.reasoning_content += str(delta.get("reasoning_content") or "")
            for fallback_index, tool_call in enumerate(delta.get("tool_calls") or []):
                index = int(tool_call.get("index", fallback_index))
                function = tool_call.get("function") or {}
                if function.get("name"):
                    self.tool_names[index] = self.tool_names.get(index, "") + str(function["name"])
                if function.get("arguments"):
                    self.tool_arguments[index] = self.tool_arguments.get(index, "") + str(
                        function["arguments"]
                    )

    def inference_metrics(
        self,
        *,
        model: str,
        total_latency_ms: float,
        thinking_enabled: bool,
    ) -> InferenceMetrics:
        tpot_ms = None
        if self.ttft_ms is not None and self.completion_tokens > 1:
            tpot_ms = max(0.0, total_latency_ms - self.ttft_ms) / (self.completion_tokens - 1)
        return InferenceMetrics(
            model=model,
            backend="vllm-http",
            thinking_mode="enabled" if thinking_enabled else "disabled",
            streaming=True,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens,
            request_latency_ms=round(total_latency_ms, 3),
            ttft_ms=round(self.ttft_ms, 3) if self.ttft_ms is not None else None,
            tpot_ms=round(tpot_ms, 3) if tpot_ms is not None else None,
            finish_reason=self.finish_reason,
        )


def build_chat_payload(
    case: VLLMBenchmarkCase,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    thinking_enabled: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": case.messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": thinking_enabled},
    }
    if case.tools:
        payload.update(
            {
                "tools": case.tools,
                "tool_choice": "required",
                "parallel_tool_calls": False,
            }
        )
    return payload


def summarize_benchmark_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [run["inference_metrics"] for run in runs if run.get("inference_metrics")]
    contract_consistent = [run for run in runs if not run.get("label_contract_conflict", False)]
    return {
        "runs": len(runs),
        "successful_runs": sum(bool(run.get("success")) for run in runs),
        "http_errors": sum(bool(run.get("http_error")) for run in runs),
        "action_mismatches": sum(bool(run.get("action_mismatch")) for run in runs),
        "argument_mismatches": sum(bool(run.get("argument_mismatch")) for run in runs),
        "reasoning_runs": sum(bool(run.get("reasoning_detected")) for run in runs),
        "label_contract_conflicts": sum(bool(run.get("label_contract_conflict")) for run in runs),
        "contract_consistent_runs": len(contract_consistent),
        "contract_consistent_successful_runs": sum(
            bool(run.get("policy_contract_success", run.get("success")))
            for run in contract_consistent
        ),
        "policy_contract_successful_runs": sum(
            bool(run.get("policy_contract_success", run.get("success"))) for run in runs
        ),
        "inference": summarize_inference_metrics(measured),
    }


def percentage_change(base: float, candidate: float) -> float | None:
    if base == 0:
        return None
    return round((candidate - base) / base * 100, 3)


def compare_curriculum_profiles(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Decompose paired rollout latency into rounds, length and compute proxies."""

    def extract(report: dict[str, Any]) -> dict[str, float]:
        inference = report["inference_metrics"]
        requests = int(inference["requests"])
        completion_total = int(inference["completion_tokens"]["total"])
        request_mean = float(inference["request_latency_ms"]["mean"])
        return {
            "successful_rollouts": int(report["behavior_gate"]["successful_rollouts"]),
            "rollouts": int(report["behavior_gate"]["rollouts"]),
            "rollout_mean_ms": float(report["rollout_latency"]["mean_ms"]),
            "rollout_p95_ms": float(report["rollout_latency"]["p95_ms"]),
            "requests": requests,
            "prompt_tokens": int(inference["prompt_tokens"]["total"]),
            "completion_tokens": completion_total,
            "completion_tokens_per_request": (completion_total / requests if requests else 0.0),
            # Transformers generate() does not expose TTFT. This diagnostic
            # proxy includes prefill and must never be presented as TPOT.
            "request_ms_per_completion_token_proxy": (
                request_mean * requests / completion_total if completion_total else 0.0
            ),
            "request_mean_ms": request_mean,
            "request_p95_ms": float(inference["request_latency_ms"]["p95"]),
        }

    base_values = extract(base)
    candidate_values = extract(candidate)
    changes = {
        key: percentage_change(float(base_values[key]), float(candidate_values[key]))
        for key in base_values
        if key not in {"successful_rollouts", "rollouts"}
    }
    signals = []
    if (changes["requests"] or 0) > 10:
        signals.append("MORE_POLICY_ROUNDS")
    if (changes["completion_tokens"] or 0) > 10:
        signals.append("MORE_COMPLETION_TOKENS")
    if (changes["request_ms_per_completion_token_proxy"] or 0) > 10:
        signals.append("SLOWER_REQUEST_COMPUTE_PROXY")
    if not signals:
        signals.append("NO_SINGLE_DOMINANT_REGRESSION")
    return {
        "schema_version": "paired-inference-profile-comparison.v1",
        "scope": "paired Transformers rollout diagnosis; TPOT requires HTTP streaming",
        "base": base_values,
        "candidate": candidate_values,
        "changes_percent": changes,
        "diagnostic_signals": signals,
    }


__all__ = [
    "StreamAccumulator",
    "VLLMBenchmarkCase",
    "build_chat_payload",
    "compare_curriculum_profiles",
    "percentage_change",
    "summarize_benchmark_runs",
]
