"""Tests for paired policy evaluation and conservative release gates."""

from datetime import UTC, datetime, timedelta

import pytest

from agentic.loop import (
    PolicyAction,
    PolicyContext,
    PolicyRouteTrace,
    PolicyShadowTrace,
)
from agentic.observations import ObservationEnvelope
from agentic.trajectory import AgentEpisode, TrajectoryStep
from evaluation.agentic_eval import AgenticEvaluator, EvaluationRun, ReleaseGateConfig


def _context(task_id: str, action: str) -> PolicyContext:
    return PolicyContext(
        trajectory_id="trajectory-1",
        goal_version=1,
        plan_version=1,
        original_request="plan a trip",
        current_subtask={"task_id": task_id},
        hard_constraints={},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        failure_summary=[],
        remaining_tasks=1,
        remaining_steps=10,
        allowed_actions=[action],
    )


def _episode() -> AgentEpisode:
    started = datetime(2026, 1, 1, tzinfo=UTC)
    action = PolicyAction(
        action="solve_itinerary",
        token_usage=12,
        route_trace=PolicyRouteTrace(
            requested_target="student",
            executed_target="teacher",
            family="recovery",
            reason="student schema failure",
            fallback_used=True,
            fallback_error_code="INVALID_SCHEMA",
        ),
        shadow_trace=PolicyShadowTrace(
            candidate_model="dpo-shadow",
            status="completed",
            action="abort",
        ),
    )
    return AgentEpisode(
        trajectory_id="trajectory-1",
        environment_version="test",
        validator_version="v1",
        policy_name="test",
        policy_version="v1",
        initial_state={},
        steps=[
            TrajectoryStep(
                step_index=0,
                task_id="solve",
                context=_context("solve", "solve_itinerary"),
                action=action,
                observations=[
                    ObservationEnvelope(
                        ok=True,
                        tool="solve_itinerary",
                        data={"status": "OPTIMAL"},
                        source="solver",
                        confidence=1.0,
                    )
                ],
                verification={"passed": True},
                state_before_hash="before",
                state_after_hash="after",
            )
        ],
        final_state={
            "task_graph": {
                "tasks": [
                    {
                        "task_id": "solve",
                        "required": True,
                        "status": "succeeded",
                        "allowed_actions": ["solve_itinerary"],
                    },
                    {
                        "task_id": "confirm",
                        "required": True,
                        "status": "blocked",
                        "allowed_actions": ["ask_user"],
                    },
                ]
            },
            "artifacts": {
                "solver": {"artifact_type": "solver_result", "payload": {"status": "OPTIMAL"}},
                "validation": {
                    "artifact_type": "validation_report",
                    "payload": {
                        "hard_pass": True,
                        "hard_violations": [],
                        "soft_scores": {"preference_match": 0.8, "diversity": 0.6},
                    },
                },
            },
        },
        status="interrupted",
        termination_reason="awaiting_user",
        created_at=started,
        completed_at=started + timedelta(milliseconds=1500),
    )


def _run(scenario_id: str, mode: str, **updates) -> EvaluationRun:
    values = {
        "scenario_id": scenario_id,
        "mode": mode,
        "hard_pass": True,
        "validated_draft": True,
        "task_completion_rate": 1.0,
        "latency_ms": 1000,
        "tool_calls": 5,
        "policy_decisions": 1,
        "policy_route_calls": 1,
        "student_route_calls": 1,
    }
    values.update(updates)
    return EvaluationRun(**values)


def test_agent_episode_metrics_exclude_user_confirmation_from_autonomous_closure():
    run = AgenticEvaluator().from_agent_episode("case-1", _episode())

    assert run.hard_pass is True
    assert run.validated_draft is True
    assert run.task_completion_rate == 1.0
    assert run.tool_calls == 1
    assert run.solver_calls == 1
    assert run.token_usage == 12
    assert run.latency_ms == 1500
    assert run.soft_score == pytest.approx(0.7)
    assert run.fallback is False
    assert run.policy_decisions == 1
    assert run.policy_route_calls == 1
    assert run.student_route_calls == 0
    assert run.teacher_route_calls == 1
    assert run.policy_route_fallbacks == 1
    assert run.route_family_counts == {"recovery": 1}
    assert run.policy_shadow_calls == 1
    assert run.policy_shadow_failures == 0
    assert run.policy_shadow_action_divergences == 1


def test_deterministic_state_is_revalidated_and_uses_measured_costs():
    state = {
        "slots": {"travel_days": 1},
        "itinerary": [
            {
                "day_number": 1,
                "activities": [
                    {
                        "poi_id": "museum",
                        "poi_name": "Museum",
                        "category": "attraction",
                        "start_time": "09:00",
                        "end_time": "10:00",
                        "duration_min": 60,
                    }
                ],
            }
        ],
        "solve_status": "optimal",
        "execution_trace": ["retrieve", "weather", "plan"],
    }

    run = AgenticEvaluator().from_deterministic_state(
        "case-1", state, latency_ms=800, tool_calls=3, token_usage=42
    )

    assert run.hard_pass is True
    assert run.validated_draft is True
    assert run.latency_ms == 800
    assert run.tool_calls == 3
    assert run.token_usage == 42
    assert run.episode_steps == 3


def test_release_gate_passes_only_on_paired_quality_cost_and_latency():
    deterministic = [_run("a", "deterministic"), _run("b", "deterministic")]
    agent = [
        _run("a", "agent", latency_ms=1200, tool_calls=6),
        _run("b", "agent", latency_ms=1300, tool_calls=6),
    ]

    report = AgenticEvaluator().compare(
        deterministic,
        agent,
        gate=ReleaseGateConfig(minimum_paired_scenarios=2),
    )

    assert report.release_eligible is True
    assert report.paired_scenarios == 2
    assert report.agent.p95_latency_ms == 1300
    assert all(check.passed for check in report.checks)
    assert report.agent.policy_route_trace_rate == 1.0
    assert report.agent.teacher_route_share == 0.0


def test_release_gate_blocks_small_samples_and_quality_regressions():
    report = AgenticEvaluator().compare(
        [_run("a", "deterministic")],
        [
            _run(
                "a",
                "agent",
                hard_pass=False,
                validated_draft=False,
                fallback=True,
                task_completion_rate=0.5,
                latency_ms=3000,
            )
        ],
    )

    assert report.release_eligible is False
    failed = {check.code for check in report.checks if not check.passed}
    assert "MINIMUM_PAIRED_SCENARIOS" in failed
    assert "AGENT_HARD_PASS_RATE" in failed
    assert "P95_LATENCY_RATIO" in failed


def test_compare_rejects_duplicate_or_unpaired_scenarios():
    evaluator = AgenticEvaluator()
    with pytest.raises(ValueError, match="duplicate scenario"):
        evaluator.compare(
            [_run("a", "deterministic"), _run("a", "deterministic")],
            [_run("a", "agent")],
        )
    with pytest.raises(ValueError, match="scenario sets differ"):
        evaluator.compare([_run("a", "deterministic")], [_run("b", "agent")])


def test_compare_rejects_silently_dropped_agent_failures():
    with pytest.raises(ValueError, match="missing agent=.*b"):
        AgenticEvaluator().compare(
            [_run("a", "deterministic"), _run("b", "deterministic")],
            [_run("a", "agent")],
        )


def test_release_gate_blocks_missing_route_traces_and_excessive_student_fallbacks():
    report = AgenticEvaluator().compare(
        [_run("a", "deterministic"), _run("b", "deterministic")],
        [
            _run(
                "a",
                "agent",
                policy_decisions=2,
                policy_route_calls=1,
                student_route_calls=0,
                teacher_route_calls=1,
                policy_route_fallbacks=1,
            ),
            _run("b", "agent"),
        ],
        gate=ReleaseGateConfig(minimum_paired_scenarios=2),
    )

    failed = {check.code for check in report.checks if not check.passed}
    assert report.release_eligible is False
    assert report.agent.policy_route_trace_rate == pytest.approx(2 / 3)
    assert report.agent.policy_route_fallback_rate == pytest.approx(0.5)
    assert "POLICY_ROUTE_TRACE_RATE" in failed
    assert "POLICY_ROUTE_FALLBACK_RATE" in failed
