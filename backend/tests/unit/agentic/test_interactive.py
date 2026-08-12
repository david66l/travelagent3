"""Tests that interactive training drives the production Agent Loop semantics."""

from typing import Any

from agentic.interactive import InteractiveAgentSession
from agentic.loop import ActionOutcome, PolicyAction
from agentic.state import AgentLedgerState, ArtifactRecord, GoalLedger, TaskGraph, TaskNode
from agentic.trajectory import EpisodeReplayVerifier


def _ledger() -> AgentLedgerState:
    return AgentLedgerState(
        goal=GoalLedger(original_request="Plan a trip"),
        task_graph=TaskGraph(
            goal_version=1,
            tasks=(
                TaskNode(
                    task_id="solve",
                    goal="solve",
                    allowed_actions=("solve_itinerary",),
                    success_criteria={"required_artifact_types": ["solver_result"]},
                ),
                TaskNode(
                    task_id="validate",
                    goal="validate",
                    depends_on=("solve",),
                    allowed_actions=("validate_itinerary",),
                    success_criteria={
                        "required_artifact_types": ["validation_report"],
                        "require_hard_pass": True,
                    },
                ),
            ),
        ),
    )


class Executor:
    async def execute(self, *, task, action, ledger) -> ActionOutcome:
        payloads: dict[str, tuple[str, dict[str, Any]]] = {
            "solve": ("solver_result", {"days": [{}]}),
            "validate": ("validation_report", {"hard_pass": True, "hard_violations": []}),
        }
        artifact_type, payload = payloads[task.task_id]
        return ActionOutcome(
            tool_calls_used=1,
            artifacts=[
                ArtifactRecord(
                    artifact_id=f"artifact-{task.task_id}",
                    artifact_type=artifact_type,
                    payload=payload,
                    goal_version=ledger.goal.goal_version,
                    plan_version=ledger.task_graph.plan_version,
                )
            ],
        )


def _session() -> InteractiveAgentSession:
    return InteractiveAgentSession(
        _ledger(),
        executor=Executor(),
        environment_version="env-v1",
        validator_version="validator-v1",
        policy_name="interactive-test",
        policy_version="v1",
    )


async def test_interactive_actions_close_the_same_verified_loop():
    session = _session()
    first = await session.start()

    assert first.done is False
    assert first.next_context.current_subtask["task_id"] == "solve"
    second = await session.submit(PolicyAction(action="solve_itinerary", token_usage=10))
    assert second.done is False
    assert second.committed_step.action.action == "solve_itinerary"
    assert second.next_context.current_subtask["task_id"] == "validate"
    final = await session.submit(PolicyAction(action="validate_itinerary", token_usage=20))

    assert final.done is True
    assert final.status == "finished"
    assert final.termination_reason == "validated_finish"
    assert final.episode is not None
    assert EpisodeReplayVerifier().verify(final.episode) == []
    assert final.episode.final_state["budget"]["used_tokens"] == 30
    assert final.episode.final_state["budget"]["used_tool_calls"] == 2


async def test_invalid_action_is_retried_by_production_controller():
    session = _session()
    first = await session.start()
    assert first.next_context.current_subtask["task_id"] == "solve"

    retry = await session.submit(PolicyAction(action="get_weather"))

    assert retry.done is False
    assert retry.committed_step.verification["error_code"] == "ACTION_NOT_ALLOWED"
    assert retry.next_context.current_subtask["task_id"] == "solve"
    assert retry.next_context.failure_summary[-1]["code"] == "ACTION_NOT_ALLOWED"
    await session.aclose()


async def test_session_rejects_submit_before_start_and_double_start():
    session = _session()
    try:
        await session.submit(PolicyAction(action="solve_itinerary"))
    except RuntimeError as exc:
        assert "not started" in str(exc)
    else:
        raise AssertionError("submit before start must fail")

    await session.start()
    try:
        await session.start()
    except RuntimeError as exc:
        assert "already started" in str(exc)
    else:
        raise AssertionError("double start must fail")
    await session.aclose()
