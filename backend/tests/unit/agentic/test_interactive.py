"""Tests that interactive training drives the production Agent Loop semantics."""

from agentic.interactive import InteractiveAgentSession
from agentic.loop import ActionOutcome, PolicyAction
from agentic.state import AgentLedgerState
from agentic.trajectory import EpisodeReplayVerifier


def _ledger():
    from agentic.runtime import initialize_agent_ledger

    return AgentLedgerState(
        **initialize_agent_ledger({"user_input": "Plan a trip"}, mode="agent")["agent_ledger"]
    )


class Executor:
    async def execute(self, *, task, action, ledger):
        return ActionOutcome(status="awaiting_user")


def _session() -> InteractiveAgentSession:
    return InteractiveAgentSession(
        _ledger(),
        executor=Executor(),
        environment_version="env-v1",
        validator_version="validator-v1",
        policy_name="interactive-test",
        policy_version="v1",
    )


async def test_interactive_model_question_interrupts_without_claiming_completion():
    session = _session()
    await session.start()
    final = await session.submit(
        PolicyAction(action="ask_user", arguments={"question": "Where?"}, token_usage=10)
    )
    assert final.done and final.status == "interrupted"
    assert final.episode.final_state["budget"]["used_tokens"] == 10
    assert EpisodeReplayVerifier().verify(final.episode) == []
    await session.aclose()


async def test_invalid_action_is_retried_by_production_controller():
    session = _session()
    first = await session.start()
    assert first.next_context.current_subtask["task_id"] == "travel_agent"

    retry = await session.submit(PolicyAction(action="retired_tool"))

    assert retry.done is False
    assert retry.committed_step.verification["error_code"] == "ACTION_NOT_ALLOWED"
    assert retry.next_context.current_subtask["task_id"] == "travel_agent"
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
