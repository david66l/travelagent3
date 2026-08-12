"""Tests for the LangGraph-facing Agent Loop integration."""

from typing import Any
from unittest.mock import AsyncMock

import pytest

from agentic.integration import run_agent_branch
from agentic.loop import ActionOutcome, PolicyAction, PolicyContext
from agentic.observations import ObservationEnvelope
from agentic.runtime import initialize_agent_ledger
from agentic.state import ArtifactRecord, FactRecord
from schemas import Location, ScoredPOI, ToolResult, WeatherDay
from tools.tool_executor import ToolExecutor
from agentic.action_executor import TravelActionExecutor


class FirstAllowedPolicy:
    async def propose(self, context: PolicyContext) -> PolicyAction:
        return PolicyAction(action=context.allowed_actions[0])


class SuccessfulExecutor:
    async def execute(self, *, task, action, ledger) -> ActionOutcome:
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
async def test_agent_branch_failure_requests_legacy_fallback():
    result = await run_agent_branch({}, policy=FirstAllowedPolicy(), executor=SuccessfulExecutor())

    assert result["agent_status"] == "fallback"
    assert result["termination_reason"] == "AGENT_LEDGER_MISSING"


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
