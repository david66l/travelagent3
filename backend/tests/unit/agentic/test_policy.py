"""Tests for the API policy adapter."""

from unittest.mock import AsyncMock

import pytest

from agentic.loop import PolicyContext
from agentic.policy import (
    ApiAgentPolicy,
    NativeToolAgentPolicy,
    PolicyDecision,
    PolicyOutputError,
    policy_prompt_payload,
)


def _context() -> PolicyContext:
    return PolicyContext(
        trajectory_id="trajectory-1",
        goal_version=1,
        plan_version=1,
        original_request="Plan Shanghai",
        current_subtask={"task_id": "weather"},
        hard_constraints={"destination": "Shanghai"},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        failure_summary=[],
        remaining_tasks=3,
        remaining_steps=5,
        allowed_actions=["get_weather"],
    )


@pytest.mark.asyncio
async def test_api_policy_uses_bounded_structured_context():
    client = AsyncMock()
    client.structured_call.return_value = PolicyDecision(action="get_weather", arguments={})
    client.last_token_usage = 123

    action = await ApiAgentPolicy(client).propose(_context())

    assert action.action == "get_weather"
    assert action.arguments == {}
    assert action.token_usage == 123
    assert client.structured_call.await_args.kwargs["task_type"] == "agent_policy"


@pytest.mark.asyncio
async def test_api_policy_rejects_action_outside_controller_allowlist():
    client = AsyncMock()
    client.structured_call.return_value = PolicyDecision(action="solve_itinerary")

    with pytest.raises(PolicyOutputError, match="allowed"):
        await ApiAgentPolicy(client).propose(_context())


@pytest.mark.asyncio
async def test_api_policy_rejects_controller_owned_arguments():
    client = AsyncMock()
    client.structured_call.return_value = PolicyDecision(
        action="get_weather", arguments={"city": "Shanghai"}
    )

    with pytest.raises(PolicyOutputError, match="invalid policy arguments"):
        await ApiAgentPolicy(client).propose(_context())


@pytest.mark.asyncio
async def test_native_tool_policy_uses_state_scoped_schemas():
    client = AsyncMock()
    client.tool_call.return_value = {
        "action": "get_weather",
        "arguments": {"date": "2026-08-12"},
    }
    client.last_token_usage = 41

    action = await NativeToolAgentPolicy(client, model="trained-policy").propose(_context())

    assert action.arguments == {"date": "2026-08-12"}
    assert action.token_usage == 41
    call = client.tool_call.await_args
    assert [tool["function"]["name"] for tool in call.args[1]] == ["get_weather"]
    assert call.kwargs["model_override"] == "trained-policy"


def test_policy_prompt_projection_removes_run_specific_ids_and_timestamps():
    context = _context()
    context.current_subtask.update(
        {
            "updated_at": "2026-08-12T00:00:00Z",
            "verifier_evidence_refs": ["secret-observation"],
        }
    )
    context.relevant_fact_refs = ["trajectory-1:random-fact"]
    context.relevant_artifact_refs = ["trajectory-1:random-artifact"]
    context.relevant_facts = [{"fact_id": "random-fact", "key": "weather"}]
    context.relevant_artifacts = [
        {"artifact_id": "random-artifact", "artifact_type": "weather_snapshot"}
    ]
    context.failure_summary = [
        {
            "failure_id": "random-failure",
            "action_id": "random-action",
            "code": "TOOL_TIMEOUT",
            "created_at": "2026-08-12T00:00:00Z",
        }
    ]

    payload = policy_prompt_payload(context)

    assert payload["trajectory_id"] == "[CURRENT_TRAJECTORY]"
    assert payload["relevant_fact_refs"] == ["fact:0"]
    assert payload["relevant_artifact_refs"] == ["artifact:0"]
    assert payload["relevant_facts"][0]["fact_id"] == "fact:0"
    assert payload["relevant_artifacts"][0]["artifact_id"] == "artifact:0"
    assert payload["failure_summary"] == [{"code": "TOOL_TIMEOUT"}]
    assert "updated_at" not in payload["current_subtask"]
