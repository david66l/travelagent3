"""Versioned tool-observation contract shared by online and training runtimes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from schemas import ToolResult


OBSERVATION_SCHEMA_VERSION = "observation.v1"


class ObservationError(BaseModel):
    """Machine-readable tool failure exposed to the policy."""

    code: str
    retryable: bool = False
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ObservationEnvelope(BaseModel):
    """Stable action observation used by LangGraph, replay and RL episodes."""

    schema_version: str = OBSERVATION_SCHEMA_VERSION
    ok: bool
    tool: str
    data: Any = None
    source: str = "unavailable"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    latency_ms: int = Field(default=0, ge=0)
    cache_hit: bool = False
    is_fallback: bool = False
    error: ObservationError | None = None
    snapshot_version: str | None = None
    environment_version: str | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _check_success_error_exclusivity(self) -> "ObservationEnvelope":
        if self.ok and self.error is not None:
            raise ValueError("successful observation cannot contain an error")
        if not self.ok and self.error is None:
            raise ValueError("failed observation must contain an error")
        return self

    @classmethod
    def from_tool_result(
        cls,
        *,
        tool: str,
        result: ToolResult,
        tool_call_id: str | None = None,
        source: str | None = None,
        cache_hit: bool = False,
        snapshot_version: str | None = None,
        environment_version: str | None = None,
    ) -> "ObservationEnvelope":
        ok = result.data_source != "unavailable" and result.data is not None
        error = None
        if not ok:
            error = ObservationError(
                code="TOOL_UNAVAILABLE",
                retryable=True,
                message=result.fallback_reason or f"{tool} returned no usable data",
            )
        return cls(
            ok=ok,
            tool=tool,
            data=result.data,
            source=source or result.data_source,
            confidence=result.confidence,
            latency_ms=result.latency_ms,
            cache_hit=cache_hit,
            is_fallback=result.is_fallback,
            error=error,
            snapshot_version=snapshot_version,
            environment_version=environment_version,
            tool_call_id=tool_call_id,
        )

    @classmethod
    def failure(
        cls,
        *,
        tool: str,
        code: str,
        message: str,
        retryable: bool,
        tool_call_id: str | None = None,
        latency_ms: int = 0,
        details: dict[str, Any] | None = None,
    ) -> "ObservationEnvelope":
        return cls(
            ok=False,
            tool=tool,
            source="unavailable",
            latency_ms=latency_ms,
            tool_call_id=tool_call_id,
            error=ObservationError(
                code=code,
                retryable=retryable,
                message=message,
                details=details or {},
            ),
        )
