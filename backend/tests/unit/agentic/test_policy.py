"""Tests for the API policy adapter."""

from unittest.mock import AsyncMock

import pytest

from agentic.loop import PolicyContext
from agentic.policy import ApiAgentPolicy, PolicyDecision, PolicyOutputError


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
    client.structured_call.return_value = PolicyDecision(
        action="get_weather", arguments={"city": "Shanghai"}
    )

    action = await ApiAgentPolicy(client).propose(_context())

    assert action.action == "get_weather"
    assert action.arguments == {"city": "Shanghai"}
    assert client.structured_call.await_args.kwargs["task_type"] == "agent_policy"


@pytest.mark.asyncio
async def test_api_policy_rejects_action_outside_controller_allowlist():
    client = AsyncMock()
    client.structured_call.return_value = PolicyDecision(action="solve_itinerary")

    with pytest.raises(PolicyOutputError, match="allowed"):
        await ApiAgentPolicy(client).propose(_context())
