import json

from evaluation.full_agent_loop_benchmark import FullAgentLoopCase
from scripts.evaluate_full_agent_loop import (
    _action_rows,
    _episode_candidate,
    _write_episode_candidates,
    build_report,
)


def test_action_rows_preserve_policy_routing_evidence() -> None:
    route = {
        "requested_target": "student",
        "executed_target": "student",
        "family": "search",
        "reason": "verified POI candidates are ready",
        "fallback_used": False,
        "fallback_error_code": None,
    }
    rows = _action_rows(
        {
            "steps": [
                {
                    "step_index": 1,
                    "task_id": "research_evidence",
                    "action": {
                        "action": "get_poi_detail",
                        "decision_source": "policy",
                        "route_trace": route,
                    },
                }
            ]
        }
    )

    assert rows[0]["route_trace"] == route


def _record(expected_outcome: str, solver_status: str | None) -> dict:
    return {
        "expected_outcome": expected_outcome,
        "passed": True,
        "validation_hard_pass": solver_status is not None,
        "solver_status": solver_status,
        "total_tokens": 10,
        "latency_ms": 20.0,
        "policy_calls": 1,
        "tool_calls": 2,
        "agent_status": "awaiting_confirmation",
        "failures": [],
    }


def test_report_separates_initial_drafts_from_revision_plans() -> None:
    records = [
        _record("draft", "optimal"),
        _record("draft", "fallback"),
        _record("revision", "optimal"),
        _record("clarification", None),
    ]

    report = build_report([], records, policy_model="test")
    summary = report["summary"]

    assert summary["cpsat_success_rate"] == 0.5
    assert summary["solver_status_counts"] == {"optimal": 1, "fallback": 1}
    assert summary["solver_status_counts_scope"] == "expected_outcome=draft"
    assert summary["planned_count"] == 3
    assert summary["planned_hard_pass_rate"] == 1.0
    assert summary["planned_solver_status_counts"] == {
        "optimal": 2,
        "fallback": 1,
    }


def test_complete_episode_is_exported_as_auditable_sft_candidate(tmp_path) -> None:
    case = FullAgentLoopCase(
        case_id="native-react-shanghai",
        slice="ordinary",
        user_input="去上海玩两天",
    )
    episode = {
        "trajectory_id": "trajectory-native-react",
        "environment_version": "travel-agent-env.v1",
        "validator_version": "travel-validator.v1",
        "policy_name": "native-tool-agent-policy",
        "policy_version": "teacher-v1",
        "initial_state": {
            "goal": {"hard_constraints": {"destination": "上海"}},
        },
    }

    candidate = _episode_candidate(
        case,
        episode,
        rollout_id="seed-7",
        phase="initial",
    )
    output = tmp_path / "episodes.jsonl"
    _write_episode_candidates(output, [candidate])

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["scenario_id"] == "native-react-shanghai:seed-7:initial"
    assert row["template_family"] == "native-react:ordinary"
    assert row["city"] == "上海"
    assert row["episode"]["trajectory_id"] == "trajectory-native-react"
