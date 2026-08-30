import pytest

from agentic.loop import PolicyContext
from scripts.build_verified_clarification_preferences import AskMissingInformationPolicy


@pytest.mark.asyncio
async def test_clarification_repair_policy_asks_for_grounded_missing_field():
    context = PolicyContext(
        trajectory_id="trajectory-1",
        goal_version=1,
        plan_version=1,
        original_request="请规划上海旅行",
        current_subtask={"task_id": "capability_check"},
        hard_constraints={"destination": "上海"},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        relevant_facts=[],
        relevant_artifacts=[],
        failure_summary=[],
        remaining_tasks=3,
        remaining_steps=8,
        allowed_actions=["capability_check", "ask_user", "abort"],
        capability={"status": "needs_user", "evidence": []},
        missing_information=["budget_range"],
    )

    action = await AskMissingInformationPolicy().propose(context)

    assert action.action == "ask_user"
    assert "预算" in action.arguments["question"]
