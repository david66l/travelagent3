from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentic.observations import ObservationEnvelope
from schemas import ToolResult


def test_tool_result_maps_to_success_observation() -> None:
    observation = ObservationEnvelope.from_tool_result(
        tool="get_weather",
        tool_call_id="call-1",
        result=ToolResult(
            data={"condition": "晴"},
            data_source="api",
            confidence=0.9,
            latency_ms=42,
        ),
    )

    assert observation.ok is True
    assert observation.schema_version == "observation.v1"
    assert observation.tool_call_id == "call-1"
    assert observation.source == "api"
    assert observation.error is None


def test_unavailable_tool_result_maps_to_structured_error() -> None:
    observation = ObservationEnvelope.from_tool_result(
        tool="get_route_matrix",
        result=ToolResult(
            data=None,
            data_source="unavailable",
            is_fallback=True,
            fallback_reason="route provider timeout",
        ),
    )

    assert observation.ok is False
    assert observation.error is not None
    assert observation.error.code == "TOOL_UNAVAILABLE"
    assert observation.error.retryable is True


def test_failed_observation_requires_error() -> None:
    with pytest.raises(ValidationError):
        ObservationEnvelope(ok=False, tool="get_weather")
