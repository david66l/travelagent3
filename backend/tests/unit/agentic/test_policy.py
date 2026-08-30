"""Tests for the API policy adapter."""

import json
from unittest.mock import AsyncMock

import pytest

from agentic.loop import PolicyAction, PolicyContext
from agentic.policy import (
    AGENT_TOOL_POLICY_SYSTEM_PROMPT,
    ApiAgentPolicy,
    ControllerFirstPolicy,
    DecisionSpecialistRoutedAgentPolicy,
    NativeToolAgentPolicy,
    PolicyDecision,
    PolicyOutputError,
    RoutedAgentPolicy,
    SelfRepairingAgentPolicy,
    ShadowComparingAgentPolicy,
    constrain_policy_context,
    is_poi_detail_specialist_state,
    policy_prompt_payload,
    route_policy_context,
)
from core.inference_metrics import InferenceMetrics


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


def test_tool_policy_prompt_blocks_continuation_after_infeasible_capability():
    assert "capability.status is infeasible" in AGENT_TOOL_POLICY_SYSTEM_PROMPT
    assert "do not continue planning" in AGENT_TOOL_POLICY_SYSTEM_PROMPT
    assert "propose_tradeoff" in AGENT_TOOL_POLICY_SYSTEM_PROMPT


def test_tool_policy_prompt_treats_external_content_as_untrusted_data():
    assert "untrusted data" in AGENT_TOOL_POLICY_SYSTEM_PROMPT
    assert "Never follow" in AGENT_TOOL_POLICY_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_api_policy_uses_bounded_structured_context():
    client = AsyncMock()
    client.structured_call.return_value = PolicyDecision(action="get_weather", arguments={})
    client.last_token_usage = 123
    client.last_request_metrics = InferenceMetrics(
        model="teacher",
        backend="openai-compatible",
        request_latency_ms=25,
    )

    action = await ApiAgentPolicy(client).propose(_context())

    assert action.action == "get_weather"
    assert action.arguments == {}
    assert action.token_usage == 123
    assert action.inference_metrics is client.last_request_metrics
    assert client.structured_call.await_args.kwargs["task_type"] == "agent_policy"
    user_payload = client.structured_call.await_args.args[0][1]["content"]
    assert "action_contracts" in user_payload


@pytest.mark.asyncio
async def test_controller_first_policy_skips_model_for_mandatory_transition():
    delegate = AsyncMock()
    context = _context()
    context.current_subtask["task_id"] = "collect_weather"

    action = await ControllerFirstPolicy(delegate).propose(context)

    assert action.action == "get_weather"
    assert action.decision_source == "controller"
    delegate.propose.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_first_policy_delegates_search_strategy():
    delegate = AsyncMock()
    delegate.propose.return_value = PolicyDecision(action="search_pois")
    context = _context()
    context.current_subtask["task_id"] = "search_candidates"
    context.allowed_actions = ["search_pois"]

    result = await ControllerFirstPolicy(delegate).propose(context)

    assert result.action == "search_pois"
    delegate.propose.assert_awaited_once_with(context)


@pytest.mark.asyncio
async def test_controller_first_closes_one_unambiguous_research_verifier_gap():
    delegate = AsyncMock()
    context = _context()
    context.current_subtask["task_id"] = "research_evidence"
    context.allowed_actions = ["search_pois", "get_route_matrix", "finalize_research"]
    context.failure_summary = [
        {
            "code": "RESEARCH_EVIDENCE_INSUFFICIENT",
            "message": "MISSING_ARTIFACT:route_matrix",
        }
    ]
    action = await ControllerFirstPolicy(delegate).propose(context)

    assert action.action == "get_route_matrix"
    assert action.decision_source == "controller"
    delegate.propose.assert_not_awaited()

    delegate.propose.return_value = PolicyAction(action="finalize_research")
    context.relevant_artifacts = [{"artifact_type": "route_matrix"}]
    resumed = await ControllerFirstPolicy(delegate).propose(context)
    assert resumed.action == "finalize_research"
    delegate.propose.assert_awaited_once_with(context)


def test_research_verifier_gaps_dynamically_narrow_policy_actions():
    context = _context()
    context.current_subtask["task_id"] = "research_evidence"
    context.allowed_actions = [
        "search_current_info",
        "get_poi_detail",
        "get_route_matrix",
        "finalize_research",
    ]
    context.failure_summary = [
        {
            "code": "RESEARCH_EVIDENCE_INSUFFICIENT",
            "message": (
                "MISSING_ARTIFACT:poi_detail_set, MISSING_ARTIFACT:route_matrix, "
                "INSUFFICIENT_POI_DETAILS:0/4"
            ),
        }
    ]

    constrained = constrain_policy_context(context)

    assert constrained.allowed_actions == ["get_poi_detail", "get_route_matrix"]


def test_low_quality_transport_evidence_reopens_transport_search():
    context = _context()
    context.current_subtask = {
        "task_id": "research_evidence",
        "action_attempt_counts": {"search_transport": 1},
    }
    context.allowed_actions = ["search_transport", "finalize_research", "propose_tradeoff", "abort"]
    context.relevant_artifacts = [{"artifact_type": "transport_search_result"}]
    context.failure_summary = [
        {
            "code": "RESEARCH_EVIDENCE_INSUFFICIENT",
            "message": "TRANSPORT_SCHEDULE_NOT_PLANNABLE",
        }
    ]

    constrained = constrain_policy_context(context)

    assert constrained.allowed_actions == ["search_transport"]


def test_research_provider_failure_uses_tradeoff_after_two_attempts():
    context = _context()
    context.current_subtask = {
        "task_id": "research_evidence",
        "action_attempt_counts": {"search_current_info": 2},
        "success_criteria": {"research_required_artifact_types": ["current_info_search"]},
    }
    context.allowed_actions = [
        "search_current_info",
        "finalize_research",
        "propose_tradeoff",
        "abort",
    ]

    constrained = constrain_policy_context(context)

    assert constrained.allowed_actions == ["propose_tradeoff"]


def test_intent_requirements_hide_irrelevant_live_tools_until_required_evidence_exists():
    context = _context()
    context.current_subtask = {
        "task_id": "research_evidence",
        "success_criteria": {
            "research_required_artifact_types": [
                "city_knowledge",
                "poi_candidate_set",
                "poi_detail_set",
                "weather_snapshot",
                "route_matrix",
            ]
        },
    }
    context.allowed_actions = [
        "retrieve_city_knowledge",
        "search_pois",
        "get_poi_detail",
        "get_weather",
        "search_current_info",
        "get_route_matrix",
        "finalize_research",
    ]
    context.relevant_artifacts = [
        {"artifact_type": "city_knowledge"},
        {"artifact_type": "poi_candidate_set"},
    ]

    constrained = constrain_policy_context(context)

    assert constrained.allowed_actions == [
        "get_poi_detail",
        "get_weather",
        "get_route_matrix",
    ]


@pytest.mark.asyncio
async def test_controller_finalizes_research_when_all_declared_evidence_is_present():
    delegate = AsyncMock()
    context = _context()
    context.current_subtask = {
        "task_id": "research_evidence",
        "success_criteria": {
            "research_required_artifact_types": [
                "city_knowledge",
                "poi_candidate_set",
                "poi_detail_set",
                "route_matrix",
            ]
        },
    }
    context.allowed_actions = [
        "retrieve_city_knowledge",
        "search_pois",
        "get_poi_detail",
        "get_route_matrix",
        "finalize_research",
    ]
    context.relevant_artifacts = [
        {"artifact_type": "city_knowledge"},
        {"artifact_type": "poi_candidate_set"},
        {"artifact_type": "poi_detail_set"},
        {"artifact_type": "route_matrix"},
    ]

    action = await ControllerFirstPolicy(delegate).propose(context)
    constrained = constrain_policy_context(context)

    assert action.action == "finalize_research"
    assert action.decision_source == "controller"
    assert constrained.allowed_actions == ["finalize_research"]
    delegate.propose.assert_not_awaited()


@pytest.mark.asyncio
async def test_event_evidence_uses_the_unified_current_information_search():
    delegate = AsyncMock()
    context = _context()
    context.original_request = "去上海看周杰伦演唱会"
    context.hard_constraints.update({"intent_kind": "event_trip", "event_query": "周杰伦演唱会"})
    context.current_subtask = {
        "task_id": "research_evidence",
        "success_criteria": {"research_required_artifact_types": ["event_search_result"]},
    }
    context.allowed_actions = ["search_current_info", "finalize_research"]

    action = await ControllerFirstPolicy(delegate).propose(context)

    assert action.action == "search_current_info"
    assert action.arguments == {"query": "周杰伦演唱会", "info_type": "event"}
    assert action.decision_source == "controller"
    delegate.propose.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_first_policy_delegates_non_solvable_capability_decision():
    delegate = AsyncMock()
    delegate.propose.return_value = PolicyDecision(action="ask_user")
    context = _context()
    context.current_subtask["task_id"] = "capability_check"
    context.allowed_actions = ["capability_check", "ask_user"]
    context.capability = {"status": "needs_user"}

    result = await ControllerFirstPolicy(delegate).propose(context)

    assert result.action == "ask_user"
    delegate.propose.assert_awaited_once_with(context)


def test_controller_constraint_removes_invalid_capability_actions_for_tradeoff():
    context = _context()
    context.current_subtask = {
        "task_id": "capability_check",
        "allowed_actions": [
            "capability_check",
            "ask_user",
            "propose_tradeoff",
            "abort",
        ],
    }
    context.capability = {"status": "infeasible", "evidence": ["budget conflict"]}
    context.allowed_actions = list(context.current_subtask["allowed_actions"])

    constrained = constrain_policy_context(context)

    assert constrained.allowed_actions == ["propose_tradeoff", "abort"]
    assert constrained.current_subtask["allowed_actions"] == [
        "propose_tradeoff",
        "abort",
    ]
    assert context.allowed_actions[0] == "capability_check"


@pytest.mark.asyncio
async def test_api_policy_rejects_action_outside_controller_allowlist():
    client = AsyncMock()
    client.structured_call.return_value = PolicyDecision(action="solve_itinerary")

    with pytest.raises(PolicyOutputError, match="allowed"):
        await ApiAgentPolicy(client).propose(_context())


@pytest.mark.asyncio
async def test_api_policy_rejects_controller_owned_arguments():
    client = AsyncMock()
    client.structured_call.return_value = PolicyDecision(
        action="get_weather", arguments={"city": "Shanghai"}
    )

    with pytest.raises(PolicyOutputError, match="invalid policy arguments"):
        await ApiAgentPolicy(client).propose(_context())


@pytest.mark.asyncio
async def test_self_repairing_policy_corrects_illegal_action_once():
    class RepairablePolicy:
        def __init__(self) -> None:
            self.contexts: list[PolicyContext] = []

        async def propose(self, context: PolicyContext) -> PolicyAction:
            self.contexts.append(context)
            if len(self.contexts) == 1:
                return PolicyAction(
                    action="solve_itinerary",
                    token_usage=17,
                )
            return PolicyAction(
                action="get_weather",
                arguments={"date": "2026-09-01"},
                token_usage=13,
            )

    delegate = RepairablePolicy()
    action = await SelfRepairingAgentPolicy(delegate).propose(_context())

    assert action.action == "get_weather"
    assert action.arguments == {"date": "2026-09-01"}
    assert action.token_usage == 30
    assert action.repair_attempts == 1
    assert action.repair_error_codes == ["ACTION_NOT_ALLOWED"]
    assert delegate.contexts[1].policy_feedback[0]["code"] == "ACTION_NOT_ALLOWED"


@pytest.mark.asyncio
async def test_self_repairing_policy_changes_known_no_progress_arguments():
    class SearchRepairPolicy:
        def __init__(self) -> None:
            self.calls = 0

        async def propose(self, context: PolicyContext) -> PolicyAction:
            self.calls += 1
            keywords = ["museum", "park"] if self.calls == 1 else ["museum"]
            return PolicyAction(action="search_pois", arguments={"keywords": keywords})

    context = _context()
    context.current_subtask = {"task_id": "search_candidates"}
    context.allowed_actions = ["search_pois"]
    context.failure_summary = [
        {
            "code": "QUERY_TOO_BROAD",
            "attempted_strategy": "search_pois",
            "attempted_arguments": {"keywords": ["museum", "park"]},
            "retryable": True,
        }
    ]
    delegate = SearchRepairPolicy()

    action = await SelfRepairingAgentPolicy(delegate).propose(context)

    assert action.arguments == {"keywords": ["museum"]}
    assert action.repair_attempts == 1
    assert action.repair_error_codes == ["REPEATED_NO_PROGRESS_ACTION"]


@pytest.mark.asyncio
async def test_self_repairing_policy_does_not_hide_provider_failure():
    class ProviderFailurePolicy:
        async def propose(self, context: PolicyContext) -> PolicyAction:
            raise TimeoutError("provider unavailable")

    with pytest.raises(TimeoutError, match="provider unavailable"):
        await SelfRepairingAgentPolicy(ProviderFailurePolicy()).propose(_context())


@pytest.mark.asyncio
async def test_native_tool_policy_uses_state_scoped_schemas():
    client = AsyncMock()
    client.tool_call.return_value = {
        "action": "get_weather",
        "arguments": {"date": "2026-08-12"},
    }
    client.last_token_usage = 41

    action = await NativeToolAgentPolicy(
        client,
        model="trained-policy",
        temperature=0.6,
        max_tokens=192,
        seed=73421,
    ).propose(_context())

    assert action.arguments == {"date": "2026-08-12"}
    assert action.token_usage == 41
    call = client.tool_call.await_args
    assert [tool["function"]["name"] for tool in call.args[1]] == ["get_weather"]
    assert call.kwargs["model_override"] == "trained-policy"
    assert call.kwargs["temperature"] == 0.6
    assert call.kwargs["max_tokens"] == 192
    assert call.kwargs["seed"] == 73421


def test_policy_prompt_projection_removes_run_specific_ids_and_timestamps():
    context = _context()
    context.current_subtask.update(
        {
            "updated_at": "2026-08-12T00:00:00Z",
            "verifier_evidence_refs": ["secret-observation"],
        }
    )
    context.relevant_fact_refs = ["trajectory-1:random-fact"]
    context.relevant_artifact_refs = ["trajectory-1:random-artifact"]
    context.relevant_facts = [{"fact_id": "random-fact", "key": "weather"}]
    context.relevant_artifacts = [
        {"artifact_id": "random-artifact", "artifact_type": "weather_snapshot"}
    ]
    context.failure_summary = [
        {
            "failure_id": "random-failure",
            "action_id": "random-action",
            "code": "TOOL_TIMEOUT",
            "created_at": "2026-08-12T00:00:00Z",
        }
    ]

    payload = policy_prompt_payload(context)

    assert payload["trajectory_id"] == "[CURRENT_TRAJECTORY]"
    assert payload["relevant_fact_refs"] == ["fact:0"]
    assert payload["relevant_artifact_refs"] == ["artifact:0"]
    assert payload["relevant_facts"][0]["fact_id"] == "fact:0"
    assert payload["relevant_artifacts"][0]["artifact_id"] == "artifact:0"
    assert payload["failure_summary"] == [{"code": "TOOL_TIMEOUT"}]
    assert "updated_at" not in payload["current_subtask"]


def test_policy_prompt_projection_compacts_legacy_artifact_summaries():
    context = _context()
    context.relevant_artifacts = [
        {
            "artifact_id": "legacy-city",
            "artifact_type": "city_knowledge",
            "payload": {
                "city": "南京",
                "topic": "博物馆",
                "record_count": 1,
                "_evidence_source": "built_in",
                "pois": [{"name": "南京博物院", "description": "x" * 5000}],
            },
        },
        {
            "artifact_id": "legacy-search",
            "artifact_type": "current_info_search",
            "source_count": 1,
            "source_urls": ["https://events.example/" + "x" * 5000],
        },
    ]

    payload = policy_prompt_payload(context)

    city, search = payload["relevant_artifacts"]
    assert city["poi_names"] == ["南京博物院"]
    assert "payload" not in city
    assert search["source_domains"] == ["events.example"]
    assert "source_urls" not in search
    assert len(json.dumps(payload, ensure_ascii=False)) < 3000


def test_react_search_failure_masks_unrelated_tools_but_keeps_argument_choice():
    context = _context()
    context.current_subtask["task_id"] = "research_evidence"
    context.current_subtask["allowed_actions"] = [
        "retrieve_city_knowledge",
        "search_pois",
        "finalize_research",
        "propose_tradeoff",
    ]
    context.allowed_actions = list(context.current_subtask["allowed_actions"])
    context.failure_summary = [
        {
            "code": "QUERY_TOO_BROAD",
            "message": "美食引入偏题候选，历史文化仍保持高精度",
            "retryable": True,
            "retry_budget_remaining": 4,
            "attempted_strategy": "search_pois",
            "attempted_arguments": {"keywords": ["历史文化", "美食"]},
        }
    ]
    context.decision_history = [
        {
            "task_id": "research_evidence",
            "action": "search_pois",
            "arguments": {"keywords": ["历史文化", "美食"]},
            "outcome_status": "failed",
            "progress_made": True,
        }
    ]

    constrained = constrain_policy_context(context)

    assert constrained.allowed_actions == ["search_pois"]
    assert constrained.current_subtask["allowed_actions"] == ["search_pois"]
    assert constrained.failure_summary == context.failure_summary


def test_react_search_failure_mask_expires_after_successful_recovery():
    context = _context()
    context.current_subtask["task_id"] = "research_evidence"
    context.current_subtask["allowed_actions"] = [
        "retrieve_city_knowledge",
        "search_pois",
        "finalize_research",
    ]
    context.allowed_actions = list(context.current_subtask["allowed_actions"])
    context.failure_summary = [
        {
            "code": "QUERY_TOO_BROAD",
            "retryable": True,
            "retry_budget_remaining": 3,
            "attempted_strategy": "search_pois",
            "attempted_arguments": {"keywords": ["历史文化", "美食"]},
        }
    ]
    context.decision_history = [
        {
            "task_id": "research_evidence",
            "action": "search_pois",
            "arguments": {"keywords": ["历史文化"]},
            "outcome_status": "completed",
            "progress_made": True,
        }
    ]

    constrained = constrain_policy_context(context)

    assert constrained.allowed_actions == context.allowed_actions


def test_tool_policy_prompt_keeps_schema_and_controller_boundaries_explicit():
    from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT

    assert "only argument keys declared" in AGENT_TOOL_POLICY_SYSTEM_PROMPT
    assert "trusted_city" in AGENT_TOOL_POLICY_SYSTEM_PROMPT
    assert "candidate_poi_ids" in AGENT_TOOL_POLICY_SYSTEM_PROMPT
    assert "injected by the controller" in AGENT_TOOL_POLICY_SYSTEM_PROMPT


def test_policy_prompt_minimizes_singleton_empty_argument_context():
    context = _context()
    context.allowed_actions = ["get_poi_detail"]
    context.current_subtask = {
        "task_id": "collect_poi_details",
        "goal": "Collect details",
        "allowed_actions": ["get_poi_detail"],
        "required_facts": ["candidate_poi_ids"],
    }
    context.relevant_facts = [
        {"fact_id": "secret-fact", "key": "candidate_poi_ids", "value": ["poi-1"]}
    ]

    payload = policy_prompt_payload(context)

    assert payload["controller_hydrates_arguments"] is True
    assert payload["allowed_actions"] == ["get_poi_detail"]
    assert "relevant_facts" not in payload
    assert "required_facts" not in payload["current_subtask"]


@pytest.mark.asyncio
async def test_shadow_comparing_policy_returns_champion_and_records_challenger():
    champion = AsyncMock()
    champion.propose.return_value = PolicyDecision(action="search_pois")
    challenger = AsyncMock()
    challenger.propose.return_value = PolicyDecision(action="ask_user")

    action = await ShadowComparingAgentPolicy(
        champion,
        challenger,
        challenger_model="dpo-shadow",
    ).propose(_context())

    assert action.action == "search_pois"
    assert action.shadow_trace is not None
    assert action.shadow_trace.status == "completed"
    assert action.shadow_trace.action == "ask_user"
    assert action.shadow_trace.candidate_model == "dpo-shadow"


@pytest.mark.asyncio
async def test_shadow_comparing_policy_fails_open_when_challenger_fails():
    champion = AsyncMock()
    champion.propose.return_value = PolicyDecision(action="search_pois")
    challenger = AsyncMock()
    challenger.propose.side_effect = RuntimeError("candidate unavailable")

    action = await ShadowComparingAgentPolicy(
        champion,
        challenger,
        challenger_model="dpo-shadow",
    ).propose(_context())

    assert action.action == "search_pois"
    assert action.shadow_trace is not None
    assert action.shadow_trace.status == "failed"
    assert action.shadow_trace.error_code == "RuntimeError"


@pytest.mark.asyncio
async def test_shadow_comparing_policy_never_hides_champion_failure():
    champion = AsyncMock()
    champion.propose.side_effect = PolicyOutputError("champion failed")
    challenger = AsyncMock()
    challenger.propose.return_value = PolicyDecision(action="search_pois")

    with pytest.raises(PolicyOutputError, match="champion failed"):
        await ShadowComparingAgentPolicy(
            champion,
            challenger,
            challenger_model="dpo-shadow",
        ).propose(_context())


def test_policy_router_sends_clarification_search_and_recovery_to_student():
    clarification = _context()
    clarification.capability = {"status": "needs_user"}
    clarification.missing_information = ["budget_range"]
    clarification.allowed_actions = ["ask_user", "capability_check"]

    search = _context()
    search.current_subtask["task_id"] = "search_candidates"
    search.allowed_actions = ["search_pois"]

    recovery = search.model_copy(deep=True)
    recovery.capability = {"status": "missing_tool"}
    recovery.failure_summary = [
        {
            "code": "EMPTY_RESULT",
            "retryable": True,
            "retry_budget_remaining": 1,
            "attempted_strategy": "search_pois",
        }
    ]

    assert route_policy_context(clarification).family == "clarification"
    assert route_policy_context(clarification).target == "student"
    assert route_policy_context(search).family == "search"
    assert route_policy_context(search).target == "student"
    assert route_policy_context(recovery).family == "recovery"
    assert route_policy_context(recovery).target == "student"


def test_policy_router_sends_bounded_exhausted_recovery_to_student():
    exhausted = _context()
    exhausted.current_subtask.update(
        {"task_id": "search_candidates", "attempts": 2, "max_attempts": 2}
    )
    exhausted.capability = {"status": "missing_tool"}
    exhausted.allowed_actions = ["search_pois", "propose_tradeoff", "abort"]
    exhausted.failure_summary = [
        {"code": "RATE_LIMIT", "retryable": True, "retry_budget_remaining": 0}
    ]

    route = route_policy_context(exhausted)
    constrained = constrain_policy_context(exhausted)

    assert route.target == "student"
    assert route.family == "tradeoff"
    assert constrained.allowed_actions == ["propose_tradeoff", "abort"]


def test_policy_router_infers_live_retry_budget_from_controller_attempts():
    recovery = _context()
    recovery.current_subtask.update(
        {"task_id": "search_candidates", "attempts": 1, "max_attempts": 2}
    )
    recovery.capability = {"status": "missing_tool"}
    recovery.allowed_actions = ["search_pois", "propose_tradeoff", "abort"]
    recovery.failure_summary = [
        {
            "code": "TIMEOUT",
            "retryable": True,
            "attempted_strategy": "search_pois",
        }
    ]

    route = route_policy_context(recovery)
    constrained = constrain_policy_context(recovery)

    assert route.target == "student"
    assert route.family == "recovery"
    assert constrained.allowed_actions == ["search_pois"]
    assert constrained.current_subtask["allowed_actions"] == ["search_pois"]


def test_policy_router_sends_bounded_recovery_without_grounded_action_to_student():
    recovery = _context()
    recovery.current_subtask.update(
        {"task_id": "search_candidates", "attempts": 1, "max_attempts": 2}
    )
    recovery.capability = {"status": "missing_tool"}
    recovery.allowed_actions = ["search_pois", "propose_tradeoff", "abort"]
    recovery.failure_summary = [{"code": "TIMEOUT", "retryable": True}]

    route = route_policy_context(recovery)
    constrained = constrain_policy_context(recovery)

    assert route.target == "student"
    assert route.family == "tradeoff"
    assert constrained.allowed_actions == ["propose_tradeoff", "abort"]


def test_controller_constraint_uses_frozen_case_action_for_retry():
    recovery = _context()
    recovery.capability = {"status": "missing_tool"}
    recovery.allowed_actions = [
        "search_pois",
        "ask_user",
        "propose_tradeoff",
        "abort",
    ]
    recovery.failure_summary = [
        {
            "action": "search_pois",
            "error_code": "STALE_DATA",
            "retryable": True,
            "retry_budget_remaining": 1,
        }
    ]
    recovery.missing_information = ["poi_location"]

    constrained = constrain_policy_context(recovery)

    assert constrained.allowed_actions == ["search_pois"]


def test_controller_constraint_asks_one_missing_user_field_at_a_time():
    clarification = _context()
    clarification.capability = {"status": "needs_user"}
    clarification.allowed_actions = ["ask_user", "capability_check"]
    clarification.missing_information = ["occupancy", "budget_range"]

    constrained = constrain_policy_context(clarification)

    assert constrained.allowed_actions == ["ask_user"]
    assert constrained.missing_information == ["occupancy"]


def test_controller_constraint_aborts_when_no_actionable_alternative_exists():
    unsupported = _context()
    unsupported.capability = {
        "status": "missing_tool",
        "actionable_alternatives": False,
        "alternatives": [],
    }
    unsupported.allowed_actions = ["propose_tradeoff", "abort"]
    unsupported.failure_summary = []

    constrained = constrain_policy_context(unsupported)
    route = route_policy_context(unsupported)

    assert constrained.allowed_actions == ["abort"]
    assert route.target == "student"
    assert route.family == "tradeoff"
    assert "student curriculum" in route.reason


def test_policy_router_sends_bounded_tradeoff_to_student_and_unknown_to_teacher():
    tradeoff = _context()
    tradeoff.capability = {"status": "infeasible"}
    tradeoff.allowed_actions = ["propose_tradeoff", "abort"]

    unknown = _context()
    unknown.allowed_actions = ["custom_complex_action"]

    assert route_policy_context(tradeoff).family == "tradeoff"
    assert route_policy_context(tradeoff).target == "student"
    assert route_policy_context(unknown).family == "complex"
    assert route_policy_context(unknown).target == "teacher"


@pytest.mark.asyncio
async def test_routed_policy_falls_back_once_when_student_output_is_invalid():
    student = AsyncMock()
    student.propose.side_effect = PolicyOutputError(
        "multiple tool calls", code="MULTIPLE_TOOL_CALLS"
    )
    teacher = AsyncMock()
    teacher.propose.return_value = PolicyDecision(action="search_pois")
    context = _context()
    context.current_subtask["task_id"] = "search_candidates"
    context.allowed_actions = ["search_pois"]
    policy = RoutedAgentPolicy(student, teacher)

    action = await policy.propose(context)

    assert action.action == "search_pois"
    student.propose.assert_awaited_once_with(context)
    teacher.propose.assert_awaited_once_with(context)
    assert policy.last_route is not None
    assert policy.last_route.fallback_used is True
    assert policy.last_route.fallback_error_code == "MULTIPLE_TOOL_CALLS"
    assert action.route_trace is not None
    assert action.route_trace.requested_target == "student"
    assert action.route_trace.executed_target == "teacher"
    assert action.route_trace.fallback_used is True
    assert action.route_trace.fallback_error_code == "MULTIPLE_TOOL_CALLS"


@pytest.mark.asyncio
async def test_routed_policy_uses_student_directly_for_bounded_tradeoff():
    student = AsyncMock()
    student.propose.return_value = PolicyDecision(action="propose_tradeoff")
    teacher = AsyncMock()
    context = _context()
    context.capability = {"status": "infeasible"}
    context.allowed_actions = ["propose_tradeoff", "abort"]
    policy = RoutedAgentPolicy(student, teacher)

    action = await policy.propose(context)

    assert action.action == "propose_tradeoff"
    student.propose.assert_awaited_once_with(context)
    teacher.propose.assert_not_awaited()
    assert policy.last_route is not None
    assert policy.last_route.target == "student"
    assert action.route_trace is not None
    assert action.route_trace.requested_target == "student"
    assert action.route_trace.executed_target == "student"
    assert action.route_trace.family == "tradeoff"


@pytest.mark.asyncio
async def test_decision_specialist_routes_only_verified_poi_detail_state():
    generalist = AsyncMock()
    specialist = AsyncMock()
    specialist.propose.return_value = PolicyDecision(action="get_poi_detail")
    context = _context()
    context.allowed_actions = ["retrieve_city_knowledge", "get_poi_detail", "get_route_matrix"]
    context.relevant_artifacts = [{"artifact_type": "poi_candidate_set"}]
    policy = DecisionSpecialistRoutedAgentPolicy(generalist, specialist)

    assert is_poi_detail_specialist_state(context) is True
    action = await policy.propose(context)

    specialist.propose.assert_awaited_once_with(context)
    generalist.propose.assert_not_awaited()
    assert action.action == "get_poi_detail"
    assert action.route_trace is not None
    assert action.route_trace.executed_target == "student"


@pytest.mark.asyncio
async def test_decision_specialist_falls_back_and_stays_off_outside_support():
    generalist = AsyncMock()
    generalist.propose.return_value = PolicyDecision(action="get_poi_detail")
    specialist = AsyncMock()
    specialist.propose.side_effect = PolicyOutputError("bad args", code="ARGUMENT_VALIDATION_FAILED")
    context = _context()
    context.allowed_actions = ["get_poi_detail"]
    context.relevant_artifacts = [{"artifact_type": "poi_candidate_set"}]
    policy = DecisionSpecialistRoutedAgentPolicy(generalist, specialist)

    action = await policy.propose(context)

    assert action.route_trace is not None
    assert action.route_trace.executed_target == "teacher"
    assert action.route_trace.fallback_used is True
    assert action.route_trace.fallback_error_code == "ARGUMENT_VALIDATION_FAILED"

    context.failure_summary = [{"code": "TOOL_TIMEOUT"}]
    specialist.reset_mock()
    await policy.propose(context)
    specialist.propose.assert_not_awaited()
