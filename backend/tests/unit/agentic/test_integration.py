"""Tests for the LangGraph-facing Agent Loop integration."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentic.integration import (
    _configured_local_policy,
    _configured_policy,
    _policy_identity,
    run_agent_branch,
    summarize_policy_routing,
)
from agentic.loop import ActionOutcome, PolicyAction, PolicyContext, PolicyRouteTrace
from agentic.observations import ObservationEnvelope
from agentic.runtime import initialize_agent_ledger
from agentic.state import ArtifactRecord, FactRecord
from agentic.policy import (
    ApiAgentPolicy,
    DecisionSpecialistRoutedAgentPolicy,
    RoutedAgentPolicy,
    ShadowComparingAgentPolicy,
)
from schemas import Location, ScoredPOI, ToolResult, WeatherDay
from tools.tool_executor import ToolExecutor
from agentic.action_executor import TravelActionExecutor
from data.collectors.amap import AmapCollector
from core.settings import settings


class FirstAllowedPolicy:
    async def propose(self, context: PolicyContext) -> PolicyAction:
        if "finish" in context.allowed_actions:
            return PolicyAction(action="finish")
        return PolicyAction(action=context.allowed_actions[0])


class TracedFirstAllowedPolicy:
    async def propose(self, context: PolicyContext) -> PolicyAction:
        return PolicyAction(
            action=context.allowed_actions[0],
            route_trace=PolicyRouteTrace(
                requested_target="student",
                executed_target="student",
                family="search",
                reason="test route",
            ),
        )


class RecordingGoalDirectedPolicy:
    def __init__(self) -> None:
        self.contexts: list[PolicyContext] = []

    async def propose(self, context: PolicyContext) -> PolicyAction:
        self.contexts.append(context)
        task_id = context.current_subtask["task_id"]
        if task_id == "search_candidates":
            has_candidates = any(
                item.get("artifact_type") == "poi_candidate_set"
                for item in context.relevant_artifacts
            )
            action = "accept_candidates" if has_candidates else "search_pois"
        elif task_id == "review_itinerary":
            action = "accept_itinerary"
        else:
            action = "finish" if "finish" in context.allowed_actions else context.allowed_actions[0]
        return PolicyAction(action=action)


def test_configured_policy_loads_and_caches_local_checkpoint(monkeypatch):
    created = []

    class FakeLocalPolicy:
        def __init__(self, checkpoint, **kwargs):
            self.checkpoint = checkpoint
            created.append((checkpoint, kwargs))

    monkeypatch.setattr(settings, "agentic_policy_backend", "local_checkpoint")
    monkeypatch.setattr(settings, "agentic_local_checkpoint", "E:/models/policy")
    monkeypatch.setattr(settings, "agentic_local_load_in_4bit", True)
    monkeypatch.setattr(settings, "agentic_local_max_new_tokens", 160)
    monkeypatch.setattr(
        settings,
        "agentic_local_structured_decoding",
        "qwen_tool_envelope",
    )
    monkeypatch.setattr("agentic.local_policy.LocalCheckpointAgentPolicy", FakeLocalPolicy)
    _configured_local_policy.cache_clear()
    try:
        first = _configured_policy()
        second = _configured_policy()
    finally:
        _configured_local_policy.cache_clear()

    assert first is second
    assert created == [
        (
            "E:/models/policy",
            {
                "max_new_tokens": 160,
                "do_sample": False,
                "load_in_4bit": True,
                "structured_decoding": "qwen_tool_envelope",
            },
        )
    ]
    assert _policy_identity(first) == (
        "local-checkpoint-agent-policy",
        "E:/models/policy",
    )


def test_local_checkpoint_backend_requires_path(monkeypatch):
    monkeypatch.setattr(settings, "agentic_policy_backend", "local_checkpoint")
    monkeypatch.setattr(settings, "agentic_local_checkpoint", "")
    _configured_local_policy.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="AGENTIC_LOCAL_CHECKPOINT"):
            _configured_policy()
    finally:
        _configured_local_policy.cache_clear()


def test_configured_routed_policy_supports_split_student_teacher_endpoints(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(settings, "agentic_policy_backend", "api")
    monkeypatch.setattr(settings, "agentic_policy_protocol", "native_tool")
    monkeypatch.setattr(settings, "agentic_policy_routing_enabled", True)
    monkeypatch.setattr(settings, "agentic_student_policy_model", "student-4b")
    monkeypatch.setattr(settings, "agentic_teacher_policy_model", "teacher-8b")
    monkeypatch.setattr(settings, "agentic_student_base_url", "http://student:8000/v1")
    monkeypatch.setattr(settings, "agentic_teacher_base_url", "http://teacher:8002/v1")
    monkeypatch.setattr(settings, "agentic_challenger_shadow_enabled", False)
    monkeypatch.setattr(settings, "vllm_api_key", "test-key")
    monkeypatch.setattr("core.llm_client.LLMClient", FakeClient)

    policy = _configured_policy()

    assert isinstance(policy, RoutedAgentPolicy)
    assert policy.student.model == "student-4b"
    assert policy.teacher.model == "teacher-8b"
    assert created == [
        {
            "base_url": "http://student:8000/v1",
            "api_key": "test-key",
            "using_vllm": True,
        },
        {
            "base_url": "http://teacher:8002/v1",
            "api_key": "test-key",
            "using_vllm": True,
        },
    ]
    assert _policy_identity(policy) == (
        "routed-native-tool-agent-policy",
        "student=student-4b;teacher=teacher-8b",
    )


def test_configured_policy_builds_shared_base_decision_specialist(monkeypatch):
    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(settings, "agentic_policy_backend", "api")
    monkeypatch.setattr(settings, "agentic_policy_protocol", "native_tool")
    monkeypatch.setattr(settings, "agentic_policy_routing_enabled", False)
    monkeypatch.setattr(settings, "agentic_decision_specialist_enabled", True)
    monkeypatch.setattr(settings, "agentic_policy_model", "travel-sft")
    monkeypatch.setattr(settings, "agentic_decision_specialist_model", "travel-grpo-poi")
    monkeypatch.setattr(settings, "vllm_base_url", "http://policy:8001/v1")
    monkeypatch.setattr(settings, "vllm_api_key", "test-key")
    monkeypatch.setattr("core.llm_client.LLMClient", FakeClient)

    policy = _configured_policy()

    assert isinstance(policy, DecisionSpecialistRoutedAgentPolicy)
    assert policy.generalist.model == "travel-sft"
    assert policy.poi_detail_specialist.model == "travel-grpo-poi"
    assert created == [
        {
            "base_url": "http://policy:8001/v1",
            "api_key": "test-key",
            "using_vllm": True,
        }
    ]
    assert _policy_identity(policy) == (
        "decision-specialist-native-tool-agent-policy",
        "generalist=travel-sft;poi_detail_specialist=travel-grpo-poi",
    )


def test_configured_policy_builds_non_authoritative_challenger(monkeypatch):
    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(settings, "agentic_policy_backend", "api")
    monkeypatch.setattr(settings, "agentic_policy_protocol", "native_tool")
    monkeypatch.setattr(settings, "agentic_policy_routing_enabled", True)
    monkeypatch.setattr(settings, "agentic_student_policy_model", "student-sft")
    monkeypatch.setattr(settings, "agentic_teacher_policy_model", "teacher-8b")
    monkeypatch.setattr(settings, "agentic_student_base_url", "http://student/v1")
    monkeypatch.setattr(settings, "agentic_teacher_base_url", "http://teacher/v1")
    monkeypatch.setattr(settings, "agentic_challenger_shadow_enabled", True)
    monkeypatch.setattr(settings, "agentic_challenger_policy_model", "student-dpo")
    monkeypatch.setattr(settings, "agentic_challenger_base_url", "http://student/v1")
    monkeypatch.setattr("core.llm_client.LLMClient", FakeClient)

    policy = _configured_policy()

    assert isinstance(policy, ShadowComparingAgentPolicy)
    assert policy.challenger_model == "student-dpo"
    assert _policy_identity(policy) == (
        "shadow-comparing-routed-native-tool-agent-policy",
        "champion=student=student-sft;teacher=teacher-8b;challenger=student-dpo",
    )


def test_api_policy_identity_does_not_require_model_attribute():
    policy = ApiAgentPolicy(client=AsyncMock())

    name, version = _policy_identity(policy)

    assert name == "api-json-agent-policy"
    assert version


class SuccessfulExecutor:
    async def execute(self, *, task, action, ledger) -> ActionOutcome:
        if action.action == "accept_candidates":
            return ActionOutcome(
                artifacts=[
                    ArtifactRecord(
                        artifact_id="candidate-selection",
                        artifact_type="candidate_selection",
                        payload={"accepted_count": 1},
                        goal_version=ledger.goal.goal_version,
                        plan_version=ledger.task_graph.plan_version,
                    )
                ]
            )
        if action.action == "accept_itinerary":
            return ActionOutcome(
                artifacts=[
                    ArtifactRecord(
                        artifact_id="verified-acceptance",
                        artifact_type="verified_itinerary_acceptance",
                        payload={"hard_pass": True},
                        goal_version=ledger.goal.goal_version,
                        plan_version=ledger.task_graph.plan_version,
                    )
                ]
            )
        mapping: dict[str, tuple[str, dict[str, Any]]] = {
            "capability_check": ("capability_report", {"status": "solvable"}),
            "collect_weather": ("weather_snapshot", {"condition": "sunny"}),
            "search_candidates": (
                "poi_candidate_set",
                {"pois": [{"name": "Museum"}]},
            ),
            "collect_poi_details": ("poi_detail_set", {"details": [{}]}),
            "collect_route_matrix": (
                "route_matrix",
                {"time_minutes": [[0]], "transport_cost": [[0.0]]},
            ),
            "solve_itinerary": (
                "solver_result",
                {
                    "status": "optimal",
                    "days": [
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
                    "solve_time_ms": 5,
                },
            ),
            "validate_itinerary": (
                "validation_report",
                {"hard_pass": True, "hard_violations": []},
            ),
            "compose_draft": ("itinerary_draft", {}),
        }
        if task.task_id == "await_confirmation":
            return ActionOutcome(status="awaiting_user")
        artifact_type, payload = mapping[task.task_id]
        observations = []
        facts = []
        if task.task_id == "collect_weather":
            observations = [
                ObservationEnvelope(
                    ok=True,
                    tool="get_weather",
                    data=payload,
                    source="test",
                    confidence=1,
                    tool_call_id=action.action_id,
                )
            ]
        if task.task_id == "search_candidates":
            facts = [
                FactRecord(
                    fact_id="candidate-ids",
                    key="candidate_poi_ids",
                    value=["Museum"],
                    observation_ref=action.action_id,
                    goal_version=ledger.goal.goal_version,
                    plan_version=ledger.task_graph.plan_version,
                    source="test",
                    confidence=1,
                )
            ]
        return ActionOutcome(
            observations=observations,
            facts=facts,
            artifacts=[
                ArtifactRecord(
                    artifact_id=f"artifact-{task.task_id}",
                    artifact_type=artifact_type,
                    payload=payload,
                    goal_version=ledger.goal.goal_version,
                    plan_version=ledger.task_graph.plan_version,
                )
            ],
            loop_control="continue" if action.action == "search_pois" else None,
        )


@pytest.mark.asyncio
async def test_amap_nested_type_is_classified_as_restaurant():
    collector = AmapCollector("test")
    try:
        item = collector._normalize(
            "Shanghai",
            {
                "name": "Restaurant",
                "location": "121.47,31.23",
                "type": "餐饮服务;中餐厅",
            },
        )
    finally:
        await collector.close()

    assert item is not None
    assert item.category == "restaurant"


@pytest.mark.asyncio
async def test_agent_branch_projects_verified_solver_draft_for_legacy_output():
    initialized = initialize_agent_ledger(
        {
            "user_input": "Plan one day in Shanghai",
            "slots": {"destination": "Shanghai", "travel_days": 1},
        },
        mode="agent",
    )

    result = await run_agent_branch(
        initialized,
        policy=FirstAllowedPolicy(),
        executor=SuccessfulExecutor(),
    )

    assert result["agent_status"] == "awaiting_confirmation"
    assert result["next_action"] == "agent_draft"
    assert result["itinerary"][0]["activities"][0]["poi_name"] == "Museum"
    assert result["validation_report"]["hard_pass"] is True
    assert result["agent_episode"]["status"] == "interrupted"
    assert result["agent_episode"]["content_hash"]


@pytest.mark.asyncio
async def test_controller_first_runtime_delegates_only_real_decision_nodes(monkeypatch):
    initialized = initialize_agent_ledger(
        {
            "user_input": "Plan one day in Shanghai",
            "slots": {"destination": "Shanghai", "travel_days": 1},
        },
        mode="agent",
    )
    policy = RecordingGoalDirectedPolicy()
    monkeypatch.setattr(settings, "agentic_execution_mode", "controller_first")

    result = await run_agent_branch(
        initialized,
        policy=policy,
        executor=SuccessfulExecutor(),
    )

    assert result["agent_status"] == "awaiting_confirmation"
    assert result["agent_execution_mode"] == "controller_first"
    assert [context.current_subtask["task_id"] for context in policy.contexts] == [
        "search_candidates",
        "search_candidates",
    ]
    sources = [step["action"]["decision_source"] for step in result["agent_episode"]["steps"]]
    assert sources.count("policy") == 2
    assert sources.count("controller") == 9


@pytest.mark.asyncio
async def test_react_runtime_keeps_deterministic_gates_controller_owned():
    initialized = initialize_agent_ledger(
        {
            "user_input": "Plan one day in Shanghai",
            "slots": {"destination": "Shanghai", "travel_days": 1},
        },
        mode="agent",
    )
    policy = RecordingGoalDirectedPolicy()

    result = await run_agent_branch(
        initialized,
        policy=policy,
        executor=SuccessfulExecutor(),
        execution_mode="react",
    )

    assert result["agent_status"] == "awaiting_confirmation"
    assert result["agent_execution_mode"] == "react"
    assert [context.current_subtask["task_id"] for context in policy.contexts] == [
        "search_candidates",
        "search_candidates",
    ]
    sources = [step["action"]["decision_source"] for step in result["agent_episode"]["steps"]]
    assert sources.count("policy") == 2
    assert sources.count("controller") == 9


@pytest.mark.asyncio
async def test_policy_driven_agent_branch_delegates_every_task_action_to_model_policy():
    initialized = initialize_agent_ledger(
        {
            "user_input": "Plan one day in Shanghai",
            "slots": {"destination": "Shanghai", "travel_days": 1},
        },
        mode="agent",
    )
    policy = RecordingGoalDirectedPolicy()

    result = await run_agent_branch(
        initialized,
        policy=policy,
        executor=SuccessfulExecutor(),
        execution_mode="policy_driven",
    )

    assert result["agent_status"] == "awaiting_confirmation"
    assert result["agent_execution_mode"] == "policy_driven"
    assert len(policy.contexts) == 11
    assert [step["action"]["decision_source"] for step in result["agent_episode"]["steps"]] == [
        "policy"
    ] * 11


@pytest.mark.asyncio
async def test_single_step_branch_checkpoints_and_resumes_one_episode():
    state = initialize_agent_ledger(
        {
            "user_input": "Plan one day in Shanghai",
            "slots": {"destination": "Shanghai", "travel_days": 1},
        },
        mode="agent",
    )
    policy = RecordingGoalDirectedPolicy()

    observed_step_counts: list[int] = []
    episode_ids: set[str] = set()
    for _ in range(16):
        patch = await run_agent_branch(
            state,
            policy=policy,
            executor=SuccessfulExecutor(),
            execution_mode="policy_driven",
            single_step=True,
        )
        state = {**state, **patch}
        observed_step_counts.append(len(state["agent_episode"]["steps"]))
        episode_ids.add(state["agent_episode"]["trajectory_id"])
        if state["agent_status"] != "running":
            break

    assert state["agent_status"] == "awaiting_confirmation"
    assert observed_step_counts[-1] == 11
    assert all(
        current > previous
        for previous, current in zip(observed_step_counts, observed_step_counts[1:])
    )
    assert (
        max(
            current - previous
            for previous, current in zip(observed_step_counts, observed_step_counts[1:])
        )
        <= 2
    )  # one checkpoint may contain a safe parallel read-only batch
    assert len(episode_ids) == 1
    assert state["agent_episode"]["status"] == "interrupted"
    assert state["agent_episode"]["content_hash"]


@pytest.mark.asyncio
async def test_agent_branch_exposes_ui_safe_policy_routing_summary():
    initialized = initialize_agent_ledger(
        {
            "user_input": "Plan one day in Shanghai",
            "slots": {"destination": "Shanghai", "travel_days": 1},
        },
        mode="agent",
    )

    result = await run_agent_branch(
        initialized,
        policy=TracedFirstAllowedPolicy(),
        executor=SuccessfulExecutor(),
    )

    summary = result["agent_policy_routing"]
    assert summary["schema_version"] == "agent-policy-routing-summary.v1"
    assert summary["route_counts"]["student"] > 0
    assert summary["route_counts"]["teacher"] == 0
    assert summary["fallback_count"] == 0
    assert summary["decisions"][0]["reason"] == "test route"
    assert summarize_policy_routing is not None


@pytest.mark.asyncio
async def test_agent_branch_failure_stops_without_legacy_fallback():
    result = await run_agent_branch({}, policy=FirstAllowedPolicy(), executor=SuccessfulExecutor())

    assert result["agent_status"] == "failed"
    assert result["termination_reason"] == "AGENT_LEDGER_MISSING"


@pytest.mark.asyncio
async def test_terminal_agent_failure_preserves_replayable_episode():
    initialized = initialize_agent_ledger(
        {
            "user_input": "Plan one day in Shanghai",
            "slots": {"destination": "Shanghai", "travel_days": 1},
        },
        mode="agent",
    )

    class AlwaysFailExecutor:
        async def execute(self, *, task, action, ledger) -> ActionOutcome:
            return ActionOutcome(
                status="failed",
                error_code="TEST_FAILURE",
                error_message="intentional",
                retryable=False,
            )

    result = await run_agent_branch(
        initialized,
        policy=FirstAllowedPolicy(),
        executor=AlwaysFailExecutor(),
    )

    assert result["agent_status"] == "failed"
    assert result["agent_episode"]["status"] == "failed"
    assert result["agent_episode"]["content_hash"]


@pytest.mark.asyncio
async def test_policy_failure_exposes_terminal_error_for_observability():
    initialized = initialize_agent_ledger(
        {
            "user_input": "Plan one day in Shanghai",
            "slots": {"destination": "Shanghai", "travel_days": 1},
        },
        mode="agent",
    )

    class FailingPolicy:
        async def propose(self, context: PolicyContext) -> PolicyAction:
            raise RuntimeError("policy endpoint unavailable")

    result = await run_agent_branch(
        initialized,
        policy=FailingPolicy(),
        executor=SuccessfulExecutor(),
    )

    assert result["agent_status"] == "failed"
    assert result["termination_reason"] == "policy_error_fallback"
    assert result["agent_error"] == "RuntimeError: policy endpoint unavailable"


@pytest.mark.asyncio
async def test_agent_branch_runs_real_route_solver_and_validator_stack():
    initialized = initialize_agent_ledger(
        {
            "user_input": "Plan one day in Shanghai",
            "slots": {"destination": "Shanghai", "travel_days": 1},
        },
        mode="agent",
    )
    tools = ToolExecutor()
    candidates = [
        {
            "name": "Museum",
            "category": "attraction",
            "score": 0.9,
            "location": {"lat": 31.23, "lng": 121.47},
            "ticket_price": 0,
            "open_time": "08:00",
            "close_time": "18:00",
        },
        {
            "name": "Park",
            "category": "attraction",
            "score": 0.8,
            "location": {"lat": 31.24, "lng": 121.48},
            "ticket_price": 0,
            "open_time": "08:00",
            "close_time": "18:00",
        },
    ]
    tools._poi.run = AsyncMock(
        return_value=ToolResult(data=candidates, data_source="built_in", confidence=1)
    )
    scored = [
        ScoredPOI(
            name=item["name"],
            category=item["category"],
            score=item["score"],
            location=Location(**item["location"]),
            ticket_price=item["ticket_price"],
            open_time=item["open_time"],
            close_time=item["close_time"],
        )
        for item in candidates
    ]
    tools._poi.search_pois = AsyncMock(
        side_effect=lambda city, keywords, category=None: [
            item for item in scored if item.name in keywords
        ]
    )
    tools._weather.query = AsyncMock(
        return_value=[
            WeatherDay(
                date="2026-08-12",
                condition="sunny",
                temp_high=30,
                temp_low=24,
                precipitation_chance=0,
                data_source="built_in",
                is_fallback=True,
            )
        ]
    )

    result = await run_agent_branch(
        initialized,
        policy=FirstAllowedPolicy(),
        executor=TravelActionExecutor(tools),
    )

    assert result["agent_status"] == "awaiting_confirmation"
    assert result["solve_status"] in {"optimal", "fallback"}
    assert result["validation_report"]["hard_pass"] is True
    assert result["itinerary"]
