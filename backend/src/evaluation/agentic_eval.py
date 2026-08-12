"""Paired deterministic-versus-agent evaluation and release gates.

The evaluator intentionally consumes normalized run records.  Production,
replay and mocked test harnesses can therefore share the same metrics without
letting a model grade its own output.
"""

from __future__ import annotations

from datetime import datetime
from math import ceil
from statistics import fmean
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic.trajectory import AgentEpisode, EpisodeReplayVerifier
from evaluation.validator import ItineraryValidator


EvalMode = Literal["deterministic", "agent"]


class EvaluationRun(BaseModel):
    """One scenario result expressed in policy-independent metrics."""

    scenario_id: str = Field(min_length=1)
    mode: EvalMode
    hard_pass: bool
    validated_draft: bool
    task_completion_rate: float = Field(ge=0, le=1)
    fallback: bool = False
    tool_calls: int = Field(default=0, ge=0)
    solver_calls: int = Field(default=0, ge=0)
    episode_steps: int = Field(default=0, ge=0)
    token_usage: int = Field(default=0, ge=0)
    latency_ms: int = Field(default=0, ge=0)
    soft_score: float = Field(default=0, ge=0, le=1)
    termination_reason: str | None = None
    hard_violation_codes: list[str] = Field(default_factory=list)


class ModeSummary(BaseModel):
    mode: EvalMode
    scenarios: int
    hard_pass_rate: float
    validated_draft_rate: float
    mean_task_completion_rate: float
    fallback_rate: float
    mean_tool_calls: float
    mean_solver_calls: float
    mean_episode_steps: float
    mean_token_usage: float
    p50_latency_ms: int
    p95_latency_ms: int
    mean_soft_score: float
    termination_reasons: dict[str, int] = Field(default_factory=dict)
    violation_codes: dict[str, int] = Field(default_factory=dict)


class MetricDelta(BaseModel):
    """Agent metric minus deterministic metric for paired scenarios."""

    hard_pass_rate: float
    validated_draft_rate: float
    mean_task_completion_rate: float
    fallback_rate: float
    mean_tool_calls: float
    p95_latency_ms: int
    mean_soft_score: float


class ReleaseGateConfig(BaseModel):
    """Conservative defaults for promoting shadow traffic to agent traffic."""

    minimum_paired_scenarios: int = Field(default=300, ge=1)
    minimum_agent_hard_pass_rate: float = Field(default=0.98, ge=0, le=1)
    minimum_agent_validated_draft_rate: float = Field(default=0.95, ge=0, le=1)
    minimum_agent_task_completion_rate: float = Field(default=0.98, ge=0, le=1)
    maximum_agent_fallback_rate: float = Field(default=0.05, ge=0, le=1)
    maximum_hard_pass_regression: float = Field(default=0.0, ge=0, le=1)
    maximum_validated_draft_regression: float = Field(default=0.0, ge=0, le=1)
    maximum_p95_latency_ratio: float = Field(default=1.50, ge=1)
    maximum_mean_tool_calls: float = Field(default=16, ge=0)


class GateCheck(BaseModel):
    code: str
    passed: bool
    actual: float | int
    expected: str


class ComparisonReport(BaseModel):
    paired_scenarios: int
    deterministic: ModeSummary
    agent: ModeSummary
    delta: MetricDelta
    release_eligible: bool
    checks: list[GateCheck]


class AgenticEvaluator:
    """Build normalized metrics, paired summaries and a promotion decision."""

    def from_deterministic_state(
        self,
        scenario_id: str,
        state: dict[str, Any],
        *,
        latency_ms: int,
        tool_calls: int,
        token_usage: int = 0,
    ) -> EvaluationRun:
        """Normalize one legacy graph result without inventing cost metrics.

        Latency and calls are mandatory because final graph state cannot
        reconstruct them reliably.  Quality is always recalculated with the
        deterministic validator when no report was persisted.
        """
        itinerary = state.get("itinerary") or []
        report = state.get("validation_report")
        if not isinstance(report, dict):
            constraints = _deterministic_constraints(state)
            report = (
                ItineraryValidator()
                .validate(
                    itinerary,
                    constraints=constraints,
                    facts=state.get("poi_candidates") or [],
                )
                .model_dump(mode="json")
            )
        hard_pass = bool(report.get("hard_pass"))
        validated_draft = bool(itinerary) and hard_pass
        soft_scores = report.get("soft_scores") or {}
        soft_score = fmean(float(value) for value in soft_scores.values()) if soft_scores else 0.0
        fallback_reasons = [str(item) for item in state.get("fallback_used") or []]
        solve_status = str(state.get("solve_status") or "")
        fallback = bool(fallback_reasons) or "fallback" in solve_status.lower()
        termination_reason = (
            "validated_finish"
            if validated_draft
            else str(state.get("termination_reason") or state.get("stage") or "partial_finish")
        )
        return EvaluationRun(
            scenario_id=scenario_id,
            mode="deterministic",
            hard_pass=hard_pass,
            validated_draft=validated_draft,
            task_completion_rate=1.0 if validated_draft else 0.0,
            fallback=fallback,
            tool_calls=tool_calls,
            solver_calls=1 if itinerary or solve_status else 0,
            episode_steps=len(state.get("execution_trace") or []),
            token_usage=token_usage,
            latency_ms=latency_ms,
            soft_score=soft_score,
            termination_reason=termination_reason,
            hard_violation_codes=[
                str(item.get("code"))
                for item in report.get("hard_violations", [])
                if item.get("code")
            ],
        )

    def from_agent_episode(
        self,
        scenario_id: str,
        episode: AgentEpisode | dict[str, Any],
    ) -> EvaluationRun:
        parsed = episode if isinstance(episode, AgentEpisode) else AgentEpisode(**episode)
        replay_errors = EpisodeReplayVerifier().verify(parsed)
        if replay_errors:
            raise ValueError(f"invalid agent episode: {', '.join(replay_errors)}")

        final_state = parsed.final_state or {}
        task_graph = final_state.get("task_graph") or {}
        tasks = task_graph.get("tasks") or []
        autonomous_required = [
            task
            for task in tasks
            if task.get("required", True) and "ask_user" not in (task.get("allowed_actions") or [])
        ]
        completed = sum(
            task.get("status") in {"succeeded", "skipped"} for task in autonomous_required
        )
        completion_rate = completed / len(autonomous_required) if autonomous_required else 0.0

        artifacts = list((final_state.get("artifacts") or {}).values())
        report = _latest_artifact_payload(artifacts, "validation_report")
        has_draft = _latest_artifact_payload(artifacts, "itinerary_draft") is not None
        has_solver_result = _latest_artifact_payload(artifacts, "solver_result") is not None
        hard_pass = bool(report and report.get("hard_pass"))
        validated_draft = hard_pass and (has_draft or has_solver_result)
        soft_scores = (report or {}).get("soft_scores") or {}
        soft_score = fmean(float(value) for value in soft_scores.values()) if soft_scores else 0.0

        actions = [step.action.action for step in parsed.steps]
        observations = [item for step in parsed.steps for item in step.observations]
        latency_ms = _elapsed_ms(parsed.created_at, parsed.completed_at)
        termination_reason = parsed.termination_reason
        is_fallback = parsed.status == "failed" or bool(
            termination_reason and "fallback" in termination_reason.lower()
        )

        return EvaluationRun(
            scenario_id=scenario_id,
            mode="agent",
            hard_pass=hard_pass,
            validated_draft=validated_draft,
            task_completion_rate=completion_rate,
            fallback=is_fallback,
            tool_calls=len(observations),
            solver_calls=sum(action == "solve_itinerary" for action in actions),
            episode_steps=len(parsed.steps),
            token_usage=sum(step.action.token_usage for step in parsed.steps),
            latency_ms=latency_ms,
            soft_score=soft_score,
            termination_reason=termination_reason,
            hard_violation_codes=[
                str(item.get("code"))
                for item in (report or {}).get("hard_violations", [])
                if item.get("code")
            ],
        )

    def compare(
        self,
        deterministic_runs: list[EvaluationRun | dict[str, Any]],
        agent_runs: list[EvaluationRun | dict[str, Any]],
        *,
        gate: ReleaseGateConfig | None = None,
    ) -> ComparisonReport:
        deterministic = [_parse_run(item, "deterministic") for item in deterministic_runs]
        agent = [_parse_run(item, "agent") for item in agent_runs]
        deterministic_by_id = _unique_by_scenario(deterministic)
        agent_by_id = _unique_by_scenario(agent)
        deterministic_ids = set(deterministic_by_id)
        agent_ids = set(agent_by_id)
        if deterministic_ids != agent_ids:
            missing_agent = sorted(deterministic_ids - agent_ids)
            missing_deterministic = sorted(agent_ids - deterministic_ids)
            raise ValueError(
                "scenario sets differ; "
                f"missing agent={missing_agent}, missing deterministic={missing_deterministic}"
            )
        paired_ids = sorted(deterministic_ids)
        if not paired_ids:
            raise ValueError("no paired scenario ids found")

        deterministic_summary = _summarize(
            [deterministic_by_id[item] for item in paired_ids], "deterministic"
        )
        agent_summary = _summarize([agent_by_id[item] for item in paired_ids], "agent")
        delta = MetricDelta(
            hard_pass_rate=agent_summary.hard_pass_rate - deterministic_summary.hard_pass_rate,
            validated_draft_rate=(
                agent_summary.validated_draft_rate - deterministic_summary.validated_draft_rate
            ),
            mean_task_completion_rate=(
                agent_summary.mean_task_completion_rate
                - deterministic_summary.mean_task_completion_rate
            ),
            fallback_rate=agent_summary.fallback_rate - deterministic_summary.fallback_rate,
            mean_tool_calls=agent_summary.mean_tool_calls - deterministic_summary.mean_tool_calls,
            p95_latency_ms=agent_summary.p95_latency_ms - deterministic_summary.p95_latency_ms,
            mean_soft_score=agent_summary.mean_soft_score - deterministic_summary.mean_soft_score,
        )
        checks = _release_checks(
            len(paired_ids), deterministic_summary, agent_summary, gate or ReleaseGateConfig()
        )
        return ComparisonReport(
            paired_scenarios=len(paired_ids),
            deterministic=deterministic_summary,
            agent=agent_summary,
            delta=delta,
            release_eligible=all(check.passed for check in checks),
            checks=checks,
        )


def _parse_run(value: EvaluationRun | dict[str, Any], expected_mode: EvalMode) -> EvaluationRun:
    parsed = value if isinstance(value, EvaluationRun) else EvaluationRun(**value)
    if parsed.mode != expected_mode:
        raise ValueError(f"expected {expected_mode} run, got {parsed.mode}")
    return parsed


def _deterministic_constraints(state: dict[str, Any]) -> dict[str, Any]:
    slots = state.get("slots") or {}
    profile = state.get("profile") or {}
    if "trip" in profile or "personal" in profile:
        profile = {**(profile.get("personal") or {}), **(profile.get("trip") or {})}
    if isinstance(profile.get("constraints"), dict):
        profile = {**profile, **profile["constraints"]}

    def value(*keys: str) -> Any:
        for key in keys:
            if slots.get(key) not in (None, "", []):
                return slots[key]
            if profile.get(key) not in (None, "", []):
                return profile[key]
        return None

    return {
        "travel_days": value("travel_days", "days"),
        "total_budget": value("total_budget", "budget_range", "budget"),
        "max_transit_minutes": value("max_transit_minutes"),
        "must_visit": value("must_visit") or [],
        "interests": value("interests") or [],
    }


def _unique_by_scenario(runs: list[EvaluationRun]) -> dict[str, EvaluationRun]:
    result: dict[str, EvaluationRun] = {}
    for run in runs:
        if run.scenario_id in result:
            raise ValueError(f"duplicate scenario id for {run.mode}: {run.scenario_id}")
        result[run.scenario_id] = run
    return result


def _latest_artifact_payload(
    artifacts: list[dict[str, Any]], artifact_type: str
) -> dict[str, Any] | None:
    matches = [item for item in artifacts if item.get("artifact_type") == artifact_type]
    if not matches:
        return None
    payload = matches[-1].get("payload")
    return payload if isinstance(payload, dict) else None


def _elapsed_ms(started_at: datetime, completed_at: datetime | None) -> int:
    if completed_at is None:
        return 0
    return max(0, int((completed_at - started_at).total_seconds() * 1000))


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values)


def _percentile(values: list[int], percentile: float) -> int:
    ordered = sorted(values)
    index = max(0, ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _summarize(runs: list[EvaluationRun], mode: EvalMode) -> ModeSummary:
    return ModeSummary(
        mode=mode,
        scenarios=len(runs),
        hard_pass_rate=_rate([run.hard_pass for run in runs]),
        validated_draft_rate=_rate([run.validated_draft for run in runs]),
        mean_task_completion_rate=fmean(run.task_completion_rate for run in runs),
        fallback_rate=_rate([run.fallback for run in runs]),
        mean_tool_calls=fmean(run.tool_calls for run in runs),
        mean_solver_calls=fmean(run.solver_calls for run in runs),
        mean_episode_steps=fmean(run.episode_steps for run in runs),
        mean_token_usage=fmean(run.token_usage for run in runs),
        p50_latency_ms=_percentile([run.latency_ms for run in runs], 0.50),
        p95_latency_ms=_percentile([run.latency_ms for run in runs], 0.95),
        mean_soft_score=fmean(run.soft_score for run in runs),
        termination_reasons=_counts([run.termination_reason or "unknown" for run in runs]),
        violation_codes=_counts([code for run in runs for code in run.hard_violation_codes]),
    )


def _release_checks(
    paired_scenarios: int,
    deterministic: ModeSummary,
    agent: ModeSummary,
    config: ReleaseGateConfig,
) -> list[GateCheck]:
    latency_limit = int(deterministic.p95_latency_ms * config.maximum_p95_latency_ratio)
    if deterministic.p95_latency_ms == 0:
        latency_passed = agent.p95_latency_ms == 0
    else:
        latency_passed = agent.p95_latency_ms <= latency_limit
    return [
        GateCheck(
            code="MINIMUM_PAIRED_SCENARIOS",
            passed=paired_scenarios >= config.minimum_paired_scenarios,
            actual=paired_scenarios,
            expected=f">= {config.minimum_paired_scenarios}",
        ),
        GateCheck(
            code="AGENT_HARD_PASS_RATE",
            passed=agent.hard_pass_rate >= config.minimum_agent_hard_pass_rate,
            actual=agent.hard_pass_rate,
            expected=f">= {config.minimum_agent_hard_pass_rate}",
        ),
        GateCheck(
            code="AGENT_VALIDATED_DRAFT_RATE",
            passed=agent.validated_draft_rate >= config.minimum_agent_validated_draft_rate,
            actual=agent.validated_draft_rate,
            expected=f">= {config.minimum_agent_validated_draft_rate}",
        ),
        GateCheck(
            code="AGENT_TASK_COMPLETION_RATE",
            passed=(agent.mean_task_completion_rate >= config.minimum_agent_task_completion_rate),
            actual=agent.mean_task_completion_rate,
            expected=f">= {config.minimum_agent_task_completion_rate}",
        ),
        GateCheck(
            code="AGENT_FALLBACK_RATE",
            passed=agent.fallback_rate <= config.maximum_agent_fallback_rate,
            actual=agent.fallback_rate,
            expected=f"<= {config.maximum_agent_fallback_rate}",
        ),
        GateCheck(
            code="HARD_PASS_REGRESSION",
            passed=(
                agent.hard_pass_rate
                >= deterministic.hard_pass_rate - config.maximum_hard_pass_regression
            ),
            actual=agent.hard_pass_rate - deterministic.hard_pass_rate,
            expected=f">= -{config.maximum_hard_pass_regression}",
        ),
        GateCheck(
            code="VALIDATED_DRAFT_REGRESSION",
            passed=(
                agent.validated_draft_rate
                >= deterministic.validated_draft_rate - config.maximum_validated_draft_regression
            ),
            actual=agent.validated_draft_rate - deterministic.validated_draft_rate,
            expected=f">= -{config.maximum_validated_draft_regression}",
        ),
        GateCheck(
            code="P95_LATENCY_RATIO",
            passed=latency_passed,
            actual=agent.p95_latency_ms,
            expected=f"<= {config.maximum_p95_latency_ratio}x deterministic ({latency_limit} ms)",
        ),
        GateCheck(
            code="MEAN_TOOL_CALLS",
            passed=agent.mean_tool_calls <= config.maximum_mean_tool_calls,
            actual=agent.mean_tool_calls,
            expected=f"<= {config.maximum_mean_tool_calls}",
        ),
    ]
