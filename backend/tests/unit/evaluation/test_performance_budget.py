from datetime import UTC, datetime, timedelta

from agentic.loop import PolicyAction, PolicyContext
from agentic.trajectory import AgentEpisode, TrajectoryStep
from evaluation.performance_budget import PerformanceBudget, evaluate_episode_performance


def _episode(*, total_ms: int = 5000, policy_ms: int = 1000) -> AgentEpisode:
    started = datetime(2026, 8, 13, tzinfo=UTC)
    context = PolicyContext(
        trajectory_id="t1",
        goal_version=1,
        plan_version=1,
        original_request="trip",
        current_subtask={"task_id": "search_candidates"},
        hard_constraints={},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        failure_summary=[],
        remaining_tasks=1,
        remaining_steps=1,
        allowed_actions=["search_pois"],
    )
    return AgentEpisode(
        trajectory_id="t1",
        environment_version="test",
        validator_version="test",
        policy_name="test",
        policy_version="test",
        initial_state={},
        steps=[
            TrajectoryStep(
                step_index=0,
                task_id="search_candidates",
                context=context,
                action=PolicyAction(action="search_pois"),
                state_before_hash="before",
                state_after_hash="after",
                policy_latency_ms=policy_ms,
                action_latency_ms=500,
            )
        ],
        final_state={},
        status="finished",
        created_at=started,
        completed_at=started + timedelta(milliseconds=total_ms),
    )


def test_performance_budget_passes_measured_fast_episode():
    assert evaluate_episode_performance(_episode()).passed is True


def test_performance_budget_reports_policy_and_total_regression():
    report = evaluate_episode_performance(
        _episode(total_ms=12_000, policy_ms=4_000),
        budget=PerformanceBudget(),
    )
    failed = {item.code for item in report.checks if not item.passed}
    assert failed == {"TOTAL_LATENCY", "POLICY_CALL_LATENCY"}


def test_react_budget_allows_bounded_multiple_policy_turns() -> None:
    episode = _episode()
    episode.steps = [
        step.model_copy(update={"step_index": index})
        for index, step in enumerate(episode.steps * 6)
    ]

    report = evaluate_episode_performance(episode)

    policy_check = next(item for item in report.checks if item.code == "POLICY_CALL_COUNT")
    assert policy_check.passed is True
    assert policy_check.actual == 6
    assert policy_check.limit == 6
