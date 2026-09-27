import json
from types import SimpleNamespace

import scripts.evaluate_full_agent_loop as full_loop
from evaluation.full_agent_loop_benchmark import FullAgentLoopCase
from scripts.evaluate_full_agent_loop import (
    _action_rows,
    _episode_candidate,
    _failure_channels,
    _score_safe_termination,
    _write_episode_candidates,
    build_report,
)


def _safe_result(*, status: str, capability: dict | None = None) -> dict:
    return {
        "agent_status": status,
        "itinerary": [],
        "agent_ledger": {
            "budget": {"used_solver_calls": 0},
            "failures": [
                {
                    "code": "RESEARCH_EVIDENCE_INSUFFICIENT",
                    "message": "CURRENT_INFO_NOT_PLANNABLE",
                }
            ],
            "goal": {"capability": capability or {"status": "solvable"}},
        },
    }


def _policy_action(action: str, arguments: dict) -> dict:
    return {"action": action, "arguments": arguments, "source": "policy"}


def test_safe_termination_accepts_grounded_abort_when_tradeoff_is_unauthorized() -> None:
    case = FullAgentLoopCase(
        case_id="external-provider-abort",
        slice="restaurant",
        user_input="查一家仍营业的餐厅",
        expected_outcome="draft_or_safe_termination",
        safe_required_actions=["search_current_info"],
        safe_terminal_actions=["propose_tradeoff", "abort"],
        required_action_arguments={
            "search_current_info": {"info_type": "restaurant"}
        },
    )
    actions = [
        _policy_action(
            "search_current_info",
            {"query": "仍营业的餐厅", "info_type": "restaurant"},
        ),
        _policy_action("abort", {"reason": "外部营业信息无法验证"}),
    ]

    assert _score_safe_termination(case, _safe_result(status="failed"), actions) == []


def test_safe_termination_rejects_wrong_search_semantics_and_exact_repeat() -> None:
    case = FullAgentLoopCase(
        case_id="external-provider-wrong-route",
        slice="restaurant",
        user_input="查一家仍营业的餐厅",
        expected_outcome="draft_or_safe_termination",
        safe_required_actions=["search_current_info"],
        safe_terminal_actions=["propose_tradeoff", "abort"],
        required_action_arguments={
            "search_current_info": {"info_type": "restaurant"}
        },
    )
    wrong = _policy_action(
        "search_current_info",
        {"query": "仍营业的餐厅", "info_type": "event"},
    )
    actions = [wrong, dict(wrong), _policy_action("abort", {"reason": "证据不足"})]

    failures = _score_safe_termination(case, _safe_result(status="failed"), actions)

    assert "ACTION_ARGUMENT_MISMATCH:search_current_info" in failures
    assert "EXACT_POLICY_ACTION_REPEAT" in failures


def test_safe_termination_accepts_authorized_tradeoff_status() -> None:
    case = FullAgentLoopCase(
        case_id="infeasible-tradeoff",
        slice="tight_budget",
        user_input="预算不足",
        expected_outcome="draft_or_safe_termination",
        safe_required_actions=["propose_tradeoff"],
        safe_terminal_actions=["propose_tradeoff"],
    )
    result = _safe_result(
        status="awaiting_information",
        capability={
            "status": "infeasible",
            "evidence": ["预算不足以覆盖指定天数"],
            "actionable_alternatives": True,
            "alternatives": ["提高预算", "减少行程天数"],
        },
    )

    assert _score_safe_termination(
        case,
        result,
        [_policy_action("propose_tradeoff", {"reason": "预算与天数冲突"})],
    ) == []


def _terminal_result(*, failure_class: str, error_type: str) -> dict:
    return {
        "agent_status": "failed",
        "termination_reason": "policy_error_fallback",
        "agent_error": f"{error_type}: failed",
        "agent_episode": {
            "events": [
                {
                    "event_type": "episode_terminated",
                    "payload": {
                        "reason": "policy_error_fallback",
                        "error": f"{error_type}: failed",
                        "failure_class": failure_class,
                        "error_type": error_type,
                        "error_code": "TOOL_CALL_SHAPE_ERROR",
                        "policy_output_summary": {
                            "tool_call_count": 2,
                            "actions": ["search_pois"],
                        },
                    },
                }
            ]
        },
    }


def test_failure_channels_keep_model_protocol_errors_out_of_runtime_errors() -> None:
    channels = _failure_channels(
        _terminal_result(
            failure_class="model_policy_failure",
            error_type="PolicyOutputError",
        )
    )

    assert channels["runtime_error"] is None
    assert channels["model_policy_failure"]["error_code"] == "TOOL_CALL_SHAPE_ERROR"
    assert channels["model_policy_failure"]["policy_output_summary"]["tool_call_count"] == 2


def test_failure_channels_keep_provider_errors_as_runtime_errors() -> None:
    channels = _failure_channels(
        _terminal_result(failure_class="runtime_error", error_type="ConnectionError")
    )

    assert channels["model_policy_failure"] is None
    assert channels["runtime_error"]["error_type"] == "ConnectionError"


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


def test_report_exposes_actual_intent_parser_and_backend() -> None:
    row = _record("draft", "optimal")
    row.update(
        {
            "intent": {"parse_source": "llm"},
            "intent_inference": {
                "model": "deepseek-v4-flash",
                "backend": "cloud-openai-compatible",
            },
        }
    )

    summary = build_report([], [row], policy_model="test")["summary"]

    assert summary["intent_parse_source_counts"] == {"llm": 1}
    assert summary["intent_actual_model_counts"] == {"deepseek-v4-flash": 1}
    assert summary["intent_backend_counts"] == {"cloud-openai-compatible": 1}


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


def _intent_result(*, missing_required=None, clarification_questions=None):
    payload = {
        "intent": "generate_itinerary",
        "parse_source": "llm",
    }
    return SimpleNamespace(
        token_usage=7,
        missing_required=list(missing_required or []),
        clarification_questions=list(clarification_questions or []),
        slots={},
        inference_metrics={
            "model": "deepseek-v4-flash",
            "backend": "cloud-openai-compatible",
        },
        model_dump=lambda **_kwargs: payload,
    )


async def test_clarification_uses_carried_intent_inference_not_stale_global(monkeypatch) -> None:
    case = FullAgentLoopCase(
        case_id="clarification-metrics-contract",
        slice="clarification",
        user_input="我想坐高铁去成都",
        expected_outcome="clarification",
        expected_missing=["origin"],
    )
    intent = _intent_result(
        missing_required=["origin"],
        clarification_questions=["你从哪里出发？"],
    )

    async def fake_process_user_turn(*_args, **_kwargs):
        return intent

    monkeypatch.setattr(full_loop, "process_user_turn", fake_process_user_turn)
    monkeypatch.setattr(
        full_loop.llm,
        "last_request_metrics",
        SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "model": "stale-global-model",
                "backend": "stale-global-backend",
            }
        ),
    )

    result = await full_loop.evaluate_case(case, policy=object())

    assert result["passed"] is True
    assert result["intent_inference"] == intent.inference_metrics


async def test_normal_path_uses_carried_intent_inference_not_stale_global(monkeypatch) -> None:
    case = FullAgentLoopCase(
        case_id="normal-metrics-contract",
        slice="ordinary",
        user_input="去上海玩两天",
    )
    intent = _intent_result()

    async def fake_process_user_turn(*_args, **_kwargs):
        return intent

    async def fake_run_agent_branch(*_args, **_kwargs):
        return {
            "agent_status": "awaiting_confirmation",
            "agent_ledger": {"budget": {}},
            "validation_report": {"hard_pass": True},
            "itinerary": [{"day": 1}],
        }

    monkeypatch.setattr(full_loop, "process_user_turn", fake_process_user_turn)
    monkeypatch.setattr(full_loop, "initialize_agent_ledger", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(full_loop, "run_agent_branch", fake_run_agent_branch)
    monkeypatch.setattr(full_loop, "_score_draft", lambda *_args: [])
    monkeypatch.setattr(
        full_loop.llm,
        "last_request_metrics",
        SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "model": "stale-global-model",
                "backend": "stale-global-backend",
            }
        ),
    )

    result = await full_loop.evaluate_case(case, policy=object())

    assert result["passed"] is True
    assert result["intent_inference"] == intent.inference_metrics


async def test_revision_collects_initial_episode_before_revision_branch(monkeypatch) -> None:
    case = FullAgentLoopCase(
        case_id="revision-episode-contract",
        slice="user_revision",
        user_input="去上海玩两天",
        expected_outcome="revision",
        revision_input="改成三天",
    )
    intent = _intent_result()

    async def fake_process_user_turn(*_args, **_kwargs):
        return intent

    async def fake_run_agent_branch(*_args, **_kwargs):
        return {
            "agent_ledger": {
                "goal": {"goal_version": 1},
                "task_graph": {"plan_version": 1},
                "artifacts": {},
            },
            "agent_episode": {
                "trajectory_id": "revision-initial-trajectory",
                "environment_version": "travel-agent-env.v1",
                "validator_version": "travel-validator.v1",
                "policy_name": "native-tool-agent-policy",
                "policy_version": "checkpoint-32",
                "initial_state": {"goal": {"hard_constraints": {"destination": "上海"}}},
            },
        }

    async def fake_revision_result(*_args, **kwargs):
        assert kwargs["intent_inference"] == {
            "model": "deepseek-v4-flash",
            "backend": "cloud-openai-compatible",
        }
        assert [item.scenario_id for item in kwargs["episode_collector"]] == [
            "revision-episode-contract:run-1:initial"
        ]
        return {"passed": True}

    monkeypatch.setattr(full_loop, "process_user_turn", fake_process_user_turn)
    monkeypatch.setattr(full_loop, "_score_intent", lambda *_args: [])
    monkeypatch.setattr(full_loop, "initialize_agent_ledger", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(full_loop, "run_agent_branch", fake_run_agent_branch)
    monkeypatch.setattr(full_loop, "_evaluate_revision_result", fake_revision_result)
    monkeypatch.setattr(
        full_loop.llm,
        "last_request_metrics",
        SimpleNamespace(
            model_dump=lambda **_kwargs: {
                "model": "stale-global-model",
                "backend": "stale-global-backend",
            }
        ),
    )

    collector = []
    result = await full_loop.evaluate_case(
        case,
        policy=object(),
        rollout_id="run-1",
        episode_collector=collector,
    )

    assert result == {"passed": True}
    assert len(collector) == 1
