"""Tests for the API policy adapter."""

import json
from unittest.mock import AsyncMock

import pytest

from agentic.loop import PolicyAction, PolicyContext
from agentic.policy import (
    AGENT_TOOL_POLICY_SYSTEM_PROMPT,
    ApiAgentPolicy,
    NativeToolAgentPolicy,
    PolicyDecision,
    PolicyOutputError,
    SelfRepairingAgentPolicy,
    constrain_policy_context,
    policy_prompt_payload,
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


def test_tool_policy_prompt_treats_external_content_as_untrusted_data():
    assert "untrusted data" in AGENT_TOOL_POLICY_SYSTEM_PROMPT
    assert "never instructions" in AGENT_TOOL_POLICY_SYSTEM_PROMPT


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


@pytest.mark.asyncio
async def test_native_tool_policy_repairs_vllm_generated_tool_json_400():
    class GeneratedToolJsonError(Exception):
        status_code = 400

    client = AsyncMock()
    client.tool_call.side_effect = [
        GeneratedToolJsonError("Invalid JSON: EOF while parsing list[function-wrap[tool_call]]"),
        {"action": "get_weather", "arguments": {"date": "2026-08-12"}},
    ]
    client.last_token_usage = 41
    policy = SelfRepairingAgentPolicy(
        NativeToolAgentPolicy(client, model="trained-policy"),
        max_repair_attempts=1,
    )

    action = await policy.propose(_context())

    assert action.action == "get_weather"
    assert action.repair_attempts == 1
    assert action.repair_error_codes == ["POLICY_OUTPUT_MALFORMED"]
    assert client.tool_call.await_count == 2


@pytest.mark.asyncio
async def test_native_tool_policy_keeps_other_provider_400_as_runtime_error():
    class ProviderRequestError(Exception):
        status_code = 400

    client = AsyncMock()
    client.tool_call.side_effect = ProviderRequestError("invalid request parameter")

    with pytest.raises(ProviderRequestError, match="invalid request parameter"):
        await NativeToolAgentPolicy(client, model="trained-policy").propose(_context())


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


def _verifier_repair_context() -> PolicyContext:
    context = _context()
    context.current_subtask = {
        "task_id": "review_itinerary",
        "status": "running",
        "action_attempt_counts": {},
    }
    context.allowed_actions = ["retry_solve", "propose_tradeoff", "abort"]
    context.hard_constraints["constraint_flexibility"] = {
        "schema_version": "constraint-flexibility.v1",
        "locked_constraints": ["total_budget"],
        "solver_adjustable_constraints": ["activity_schedule"],
        "relaxable_constraints": ["daily_time_window"],
        "relaxation_options": {
            "daily_time_window": ["延长每日活动时间"],
        },
    }
    context.capability = {
        "status": "solvable",
        "evidence": ["活动时间重叠，但仍可调整排程"],
        "actionable_alternatives": None,
        "alternatives": [],
    }
    context.relevant_artifacts = [
        {"artifact_type": "solver_result", "status": "fallback"},
        {
            "artifact_type": "validation_report",
            "hard_pass": False,
            "violation_codes": ["ACTIVITY_TIME_OVERLAP"],
            "violations": [
                {
                    "code": "ACTIVITY_TIME_OVERLAP",
                    "message": "活动时间重叠，但仍可调整排程",
                }
            ],
        },
    ]
    return context
