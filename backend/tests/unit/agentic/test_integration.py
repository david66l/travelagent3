"""Tests for the LangGraph-facing Agent Loop integration."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentic.integration import (
    _configured_local_policy,
    _configured_policy,
    _policy_identity,
    run_agent_branch,
)
from agentic.loop import ActionOutcome, PolicyAction, PolicyContext
from agentic.observations import ObservationEnvelope
from agentic.runtime import initialize_agent_ledger
from agentic.state import ArtifactRecord, FactRecord
from agentic.policy import ApiAgentPolicy
from data.collectors.amap import AmapCollector
from core.settings import settings


class FirstAllowedPolicy:
    async def propose(self, context: PolicyContext) -> PolicyAction:
        if "finish" in context.allowed_actions:
            return PolicyAction(action="finish")
        return PolicyAction(action=context.allowed_actions[0])


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
async def test_executor_exception_keeps_zero_step_runtime_terminal_episode(monkeypatch):
    initialized = initialize_agent_ledger(
        {
            "user_input": "Plan one day in Shanghai",
            "slots": {"destination": "Shanghai", "travel_days": 1},
        },
        mode="agent",
    )

    async def crashing_loop(*_args, **_kwargs):
        raise ConnectionError("tool transport unavailable")

    monkeypatch.setattr("agentic.integration.BoundedAgentLoop.run", crashing_loop)

    result = await run_agent_branch(
        initialized,
        policy=FirstAllowedPolicy(),
        executor=SuccessfulExecutor(),
    )

    assert result["termination_reason"] == "runtime_error_fallback"
    assert result["agent_error"] == "ConnectionError: tool transport unavailable"
    assert result["agent_episode"]["status"] == "failed"
    terminal = result["agent_episode"]["events"][-1]["payload"]
    assert terminal["failure_class"] == "runtime_error"
    assert terminal["error_type"] == "ConnectionError"
