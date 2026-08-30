"""Deterministic latency budgets for Agent episodes and release evidence."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic.trajectory import AgentEpisode


class PerformanceBudget(BaseModel):
    total_latency_ms: int = Field(default=10_000, ge=1)
    policy_call_ms: int = Field(default=3_000, ge=1)
    action_batch_ms: int = Field(default=5_000, ge=1)
    solver_ms: int = Field(default=3_000, ge=1)
    # A real ReAct research loop normally needs several observe-decide turns.
    # Six permits bounded tool choice without accepting runaway repetition.
    maximum_policy_calls: int = Field(default=6, ge=0)


class PerformanceCheck(BaseModel):
    code: str
    passed: bool
    actual: int
    limit: int


class PerformanceBudgetReport(BaseModel):
    passed: bool
    checks: list[PerformanceCheck]
    stage_latency_ms: dict[str, int]


def evaluate_episode_performance(
    episode: AgentEpisode | dict,
    *,
    budget: PerformanceBudget | None = None,
) -> PerformanceBudgetReport:
    parsed = episode if isinstance(episode, AgentEpisode) else AgentEpisode(**episode)
    limits = budget or PerformanceBudget()
    total_ms = 0
    if parsed.completed_at is not None:
        total_ms = max(0, int((parsed.completed_at - parsed.created_at).total_seconds() * 1000))

    policy_steps = [step for step in parsed.steps if step.action.decision_source == "policy"]
    max_policy_ms = max((step.policy_latency_ms for step in policy_steps), default=0)
    max_action_ms = max((step.action_latency_ms for step in parsed.steps), default=0)
    solver_ms = max(
        (
            step.action_latency_ms
            for step in parsed.steps
            if step.action.action == "solve_itinerary"
        ),
        default=0,
    )
    checks = [
        PerformanceCheck(
            code="TOTAL_LATENCY",
            passed=total_ms <= limits.total_latency_ms,
            actual=total_ms,
            limit=limits.total_latency_ms,
        ),
        PerformanceCheck(
            code="POLICY_CALL_LATENCY",
            passed=max_policy_ms <= limits.policy_call_ms,
            actual=max_policy_ms,
            limit=limits.policy_call_ms,
        ),
        PerformanceCheck(
            code="ACTION_BATCH_LATENCY",
            passed=max_action_ms <= limits.action_batch_ms,
            actual=max_action_ms,
            limit=limits.action_batch_ms,
        ),
        PerformanceCheck(
            code="SOLVER_LATENCY",
            passed=solver_ms <= limits.solver_ms,
            actual=solver_ms,
            limit=limits.solver_ms,
        ),
        PerformanceCheck(
            code="POLICY_CALL_COUNT",
            passed=len(policy_steps) <= limits.maximum_policy_calls,
            actual=len(policy_steps),
            limit=limits.maximum_policy_calls,
        ),
    ]
    return PerformanceBudgetReport(
        passed=all(item.passed for item in checks),
        checks=checks,
        stage_latency_ms={
            step.task_id: max(
                step.policy_latency_ms + step.action_latency_ms,
                # Parallel tasks share the batch timer; max keeps the report
                # conservative without pretending the same batch ran twice.
                0,
            )
            for step in parsed.steps
        },
    )
