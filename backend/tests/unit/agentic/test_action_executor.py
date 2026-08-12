"""Tests for Agent Loop to travel-tool adaptation."""

from unittest.mock import AsyncMock

import pytest

from agentic.action_executor import TravelActionExecutor
from agentic.loop import PolicyAction
from agentic.state import AgentLedgerState, ArtifactRecord, GoalLedger, TaskGraph, TaskNode


def _ledger(action: str, artifact_type: str = "poi_candidate_set") -> AgentLedgerState:
    return AgentLedgerState(
        goal=GoalLedger(original_request="Plan Shanghai"),
        task_graph=TaskGraph(
            goal_version=1,
            tasks=(
                TaskNode(
                    task_id="task",
                    goal="do it",
                    status="running",
                    allowed_actions=(action,),
                    success_criteria={"required_artifact_types": [artifact_type]},
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_capability_check_creates_controller_grounded_artifact():
    ledger = _ledger("capability_check", "capability_report")
    outcome = await TravelActionExecutor(AsyncMock()).execute(
        task=ledger.task_graph.get("task"),
        action=PolicyAction(action="capability_check"),
        ledger=ledger,
    )

    assert outcome.artifacts[0].artifact_type == "capability_report"
    assert outcome.artifacts[0].payload["status"] == "solvable"


@pytest.mark.asyncio
async def test_search_commits_candidate_ids_from_versioned_observation():
    tools = AsyncMock()
    tools.execute.return_value = [
        {
            "observation": {
                "ok": True,
                "tool": "search_pois",
                "data": [{"name": "Museum"}, {"id": "park-1", "name": "Park"}],
                "source": "built_in",
                "confidence": 0.9,
                "tool_call_id": "call-1",
            }
        }
    ]
    ledger = _ledger("search_pois")
    outcome = await TravelActionExecutor(tools).execute(
        task=ledger.task_graph.get("task"),
        action=PolicyAction(
            action_id="call-1",
            action="search_pois",
            arguments={"city": "Shanghai"},
        ),
        ledger=ledger,
    )

    assert outcome.facts[0].key == "candidate_poi_ids"
    assert outcome.facts[0].value == ["Museum", "park-1"]
    assert outcome.artifacts[0].artifact_type == "poi_candidate_set"


@pytest.mark.asyncio
async def test_tool_failure_is_preserved_as_retryable_outcome():
    tools = AsyncMock()
    tools.execute.return_value = [
        {
            "observation": {
                "ok": False,
                "tool": "get_weather",
                "source": "unavailable",
                "confidence": 0,
                "error": {
                    "code": "UPSTREAM_TIMEOUT",
                    "message": "timeout",
                    "retryable": True,
                },
            }
        }
    ]
    ledger = _ledger("get_weather", "unused")
    outcome = await TravelActionExecutor(tools).execute(
        task=ledger.task_graph.get("task"),
        action=PolicyAction(action="get_weather", arguments={"city": "Shanghai"}),
        ledger=ledger,
    )

    assert outcome.status == "failed"
    assert outcome.error_code == "UPSTREAM_TIMEOUT"
    assert outcome.retryable is True


@pytest.mark.asyncio
async def test_weather_observation_is_persisted_for_later_policy_steps():
    tools = AsyncMock()
    tools.execute.return_value = [
        {
            "observation": {
                "ok": True,
                "tool": "get_weather",
                "data": [{"date": "2026-08-12", "condition": "rain"}],
                "source": "api",
                "confidence": 0.9,
            }
        }
    ]
    ledger = _ledger("get_weather", "weather_snapshot")

    outcome = await TravelActionExecutor(tools).execute(
        task=ledger.task_graph.get("task"),
        action=PolicyAction(action="get_weather", arguments={"city": "Shanghai"}),
        ledger=ledger,
    )

    assert outcome.artifacts[0].artifact_type == "weather_snapshot"
    assert outcome.artifacts[0].payload["days"][0]["condition"] == "rain"


@pytest.mark.asyncio
async def test_solver_arguments_are_hydrated_from_trusted_artifacts():
    tools = AsyncMock()
    tools.execute.return_value = [
        {
            "observation": {
                "ok": True,
                "tool": "solve_itinerary",
                "data": {"status": "optimal", "days": []},
                "source": "built_in",
                "confidence": 1,
            }
        }
    ]
    ledger = _ledger("solve_itinerary", "solver_result")
    ledger.goal.hard_constraints = {
        "destination": "Shanghai",
        "travel_days": 2,
        "budget_range": 3000,
    }
    ledger.artifacts["candidates"] = ArtifactRecord(
        artifact_id="candidates",
        artifact_type="poi_candidate_set",
        payload={
            "pois": [
                {
                    "name": "Museum",
                    "location": {"lat": 31.23, "lng": 121.47},
                    "ticket_price": 50,
                }
            ]
        },
        goal_version=1,
        plan_version=1,
    )
    ledger.artifacts["matrix"] = ArtifactRecord(
        artifact_id="matrix",
        artifact_type="route_matrix",
        payload={"time_minutes": [[0]], "transport_cost": [[0.0]]},
        goal_version=1,
        plan_version=1,
    )

    await TravelActionExecutor(tools).execute(
        task=ledger.task_graph.get("task"),
        action=PolicyAction(action="solve_itinerary", arguments={}),
        ledger=ledger,
    )

    call = tools.execute.await_args.args[0][0]
    arguments = __import__("json").loads(call["function"]["arguments"])
    assert arguments["pois"][0]["name"] == "Museum"
    assert arguments["constraints"]["travel_days"] == 2
    assert arguments["dist_matrix"] == [[0]]


def test_validation_arguments_use_solver_artifact_not_policy_copy():
    ledger = _ledger("validate_itinerary", "validation_report")
    ledger.artifacts["solver"] = ArtifactRecord(
        artifact_id="solver",
        artifact_type="solver_result",
        payload={"days": [{"day_number": 1, "activities": []}]},
        goal_version=1,
        plan_version=1,
    )
    executor = TravelActionExecutor(AsyncMock())

    arguments = executor._hydrate_arguments(
        ledger,
        PolicyAction(
            action="validate_itinerary",
            arguments={"itinerary": [{"hallucinated": True}]},
        ),
    )

    assert arguments["itinerary"] == [{"day_number": 1, "activities": []}]


@pytest.mark.asyncio
async def test_poi_detail_action_collects_entire_candidate_set():
    tools = AsyncMock()
    tools.execute.return_value = [
        {
            "observation": {
                "ok": True,
                "tool": "get_poi_detail",
                "data": {"poi_name": name},
                "source": "built_in",
                "confidence": 1,
                "tool_call_id": f"call:{index}",
            }
        }
        for index, name in enumerate(["Museum", "Park"])
    ]
    ledger = _ledger("get_poi_detail", "poi_detail_set")
    ledger.goal.hard_constraints = {"destination": "Shanghai"}
    ledger.artifacts["candidates"] = ArtifactRecord(
        artifact_id="candidates",
        artifact_type="poi_candidate_set",
        payload={"pois": [{"name": "Museum"}, {"name": "Park"}]},
        goal_version=1,
        plan_version=1,
    )

    outcome = await TravelActionExecutor(tools).execute(
        task=ledger.task_graph.get("task"),
        action=PolicyAction(action_id="call", action="get_poi_detail"),
        ledger=ledger,
    )

    assert len(tools.execute.await_args.args[0]) == 2
    assert outcome.artifacts[0].payload["expected_count"] == 2
    assert len(outcome.artifacts[0].payload["details"]) == 2
