"""Versioned per-request inference evidence and deterministic summaries."""

from __future__ import annotations

from statistics import fmean
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field


INFERENCE_METRICS_SCHEMA_VERSION = "policy-inference-metrics.v1"


class InferenceMetrics(BaseModel):
    """Metrics for one model request, excluding model load and tool execution."""

    schema_version: str = INFERENCE_METRICS_SCHEMA_VERSION
    model: str
    backend: str
    task_type: str = "agent_policy"
    thinking_mode: Literal["enabled", "disabled", "unknown"] = "unknown"
    streaming: bool = False
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cached_prompt_tokens: int = Field(default=0, ge=0)
    request_latency_ms: float = Field(default=0.0, ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    tpot_ms: float | None = Field(default=None, ge=0)
    finish_reason: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile for a bounded metric list."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "mean": round(fmean(values), 3),
        "p50": round(percentile(values, 0.50), 3),
        "p95": round(percentile(values, 0.95), 3),
    }


def summarize_inference_metrics(
    metrics: Sequence[InferenceMetrics | dict[str, Any]],
) -> dict[str, Any]:
    """Summarize one homogeneous or mixed set without hiding missing TTFT data."""
    parsed = [
        item if isinstance(item, InferenceMetrics) else InferenceMetrics(**item) for item in metrics
    ]
    ttft = [item.ttft_ms for item in parsed if item.ttft_ms is not None]
    tpot = [item.tpot_ms for item in parsed if item.tpot_ms is not None]
    return {
        "schema_version": INFERENCE_METRICS_SCHEMA_VERSION,
        "requests": len(parsed),
        "models": sorted({item.model for item in parsed}),
        "backends": sorted({item.backend for item in parsed}),
        "thinking_modes": sorted({item.thinking_mode for item in parsed}),
        "streaming_requests": sum(item.streaming for item in parsed),
        "prompt_tokens": {
            "total": sum(item.prompt_tokens for item in parsed),
            **_distribution([float(item.prompt_tokens) for item in parsed]),
        },
        "completion_tokens": {
            "total": sum(item.completion_tokens for item in parsed),
            **_distribution([float(item.completion_tokens) for item in parsed]),
        },
        "cached_prompt_tokens": {
            "total": sum(item.cached_prompt_tokens for item in parsed),
            **_distribution([float(item.cached_prompt_tokens) for item in parsed]),
        },
        "request_latency_ms": _distribution([item.request_latency_ms for item in parsed]),
        "ttft_ms": {"measured_requests": len(ttft), **_distribution(ttft)},
        "tpot_ms": {"measured_requests": len(tpot), **_distribution(tpot)},
    }


__all__ = [
    "INFERENCE_METRICS_SCHEMA_VERSION",
    "InferenceMetrics",
    "percentile",
    "summarize_inference_metrics",
]
