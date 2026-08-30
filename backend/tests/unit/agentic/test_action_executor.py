"""Tests for Agent Loop to travel-tool adaptation."""

from datetime import UTC, datetime, timedelta

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
async def test_unified_search_commits_event_evidence_and_fixed_planning_constraint():
    tools = AsyncMock()
    tools.execute.return_value = [
        {
            "observation": {
                "ok": True,
                "tool": "search_current_info",
                "data": {
                    "info_type": "event",
                    "event": {
                        "name": "周杰伦演唱会",
                        "date": "2026-09-01",
                        "start_time": "19:30",
                        "end_time": "21:30",
                        "venue": "梅赛德斯奔驰文化中心",
                        "lat": 31.19,
                        "lng": 121.49,
                        "complete": True,
                    },
                    "results": [{"url": "https://example.org/official"}],
                    "queried_at": "2026-08-29T00:00:00+00:00",
                },
                "source": "api",
                "confidence": 0.8,
                "tool_call_id": "event-search-1",
            }
        }
    ]
    ledger = _ledger("search_current_info", "event_search_result")
    ledger.goal.original_request = "去上海看周杰伦演唱会"
    ledger.goal.hard_constraints = {
        "destination": "Shanghai",
        "intent_kind": "event_trip",
        "event_query": "周杰伦演唱会",
    }

    outcome = await TravelActionExecutor(tools).execute(
        task=ledger.task_graph.get("task"),
        action=PolicyAction(
            action_id="event-search-1",
            action="search_current_info",
            arguments={"query": "周杰伦演唱会", "info_type": "event"},
        ),
        ledger=ledger,
    )

    assert [item.artifact_type for item in outcome.artifacts] == [
        "event_search_result",
        "poi_candidate_set",
    ]
    assert outcome.facts[0].key == "fixed_events"
    assert outcome.facts[0].value[0]["start_time"] == "19:30"
    assert outcome.facts[0].expires_at is not None
    assert outcome.artifacts[0].expires_at is not None
    assert outcome.facts[0].expires_at <= datetime.now(UTC) + timedelta(hours=6, seconds=5)


def test_expired_fact_and_artifact_are_not_reused():
    from agentic.state import FactRecord

    ledger = _ledger("solve_itinerary", "solver_result")
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    ledger.facts["transport"] = FactRecord(
        fact_id="transport",
        key="transport_time_windows",
        value={"daily_start_minutes": [600]},
        observation_ref="obs",
        goal_version=1,
        plan_version=1,
        source="api",
        confidence=1,
        expires_at=expired_at,
    )
    ledger.artifacts["event"] = ArtifactRecord(
        artifact_id="event",
        artifact_type="event_search_result",
        payload={"event": {"complete": True}},
        goal_version=1,
        plan_version=1,
        expires_at=expired_at,
    )

    assert TravelActionExecutor._latest_fact_value(ledger, "transport_time_windows") is None
    assert TravelActionExecutor._latest_artifact(ledger, "event_search_result") is None


def test_live_opening_and_closure_evidence_patch_exact_solver_poi():
    ledger = _ledger("solve_itinerary", "solver_result")
    ledger.goal.hard_constraints = {
        "destination": "北京",
        "travel_days": 1,
        "start_date": "2026-09-01",
    }
    ledger.artifacts["candidates"] = ArtifactRecord(
        artifact_id="candidates",
        artifact_type="poi_candidate_set",
        payload={
            "pois": [
                {"id": "forbidden-city", "name": "故宫", "category": "attraction"},
                {"id": "temple", "name": "天坛", "category": "attraction"},
            ]
        },
        goal_version=1,
        plan_version=1,
    )
    ledger.artifacts["details"] = ArtifactRecord(
        artifact_id="details",
        artifact_type="poi_detail_set",
        payload={
            "details": [{"name": "故宫"}, {"name": "天坛"}],
            "expected_count": 2,
        },
        goal_version=1,
        plan_version=1,
    )
    ledger.artifacts["live-hours"] = ArtifactRecord(
        artifact_id="live-hours",
        artifact_type="current_info_search",
        payload={
            "info_type": "closure",
            "date": "2026-09-01",
            "results": [
                {
                    "title": "故宫临时闭馆通知",
                    "snippet": "故宫将于2026-09-01临时闭馆",
                    "url": "https://official.example/closure",
                },
                {
                    "title": "天坛开放时间调整",
                    "snippet": "天坛2026-09-01开放时间10:00-16:00",
                    "url": "https://official.example/hours",
                },
            ],
        },
        goal_version=1,
        plan_version=1,
    )

    arguments = TravelActionExecutor(AsyncMock())._hydrate_arguments(
        ledger, PolicyAction(action="solve_itinerary")
    )

    forbidden_city = next(item for item in arguments["pois"] if item["name"] == "故宫")
    temple = next(item for item in arguments["pois"] if item["name"] == "天坛")
    assert forbidden_city["closed_dates"] == ["2026-09-01"]
    assert forbidden_city["availability_evidence_urls"] == ["https://official.example/closure"]
    assert temple["closed_dates"] == []
    assert temple["date_opening_hours"] == {"2026-09-01": ("10:00", "16:00")}
    assert temple["availability_evidence_urls"] == ["https://official.example/hours"]


@pytest.mark.asyncio
async def test_transport_evidence_becomes_solver_daily_time_boundaries():
    tools = AsyncMock()
    tools.execute.return_value = [
        {
            "observation": {
                "ok": True,
                "tool": "search_transport",
                "data": {
                    "legs": [
                        {
                            "direction": "inbound",
                            "date": "2026-09-01",
                            "selected_option": {
                                "service_code": "G1",
                                "departure_time": "07:00",
                                "arrival_time": "11:30",
                                "source_url": "https://rail.example/inbound",
                            },
                        },
                        {
                            "direction": "outbound",
                            "date": "2026-09-03",
                            "selected_option": {
                                "service_code": "G2",
                                "departure_time": "18:00",
                                "arrival_time": "22:30",
                                "source_url": "https://rail.example/outbound",
                            },
                        },
                    ],
                    "results": [{"url": "https://rail.example/inbound"}],
                    "queried_at": "2026-08-29T00:00:00+00:00",
                },
                "source": "api",
                "confidence": 0.8,
                "tool_call_id": "transport-1",
            }
        }
    ]
    ledger = _ledger("search_transport", "transport_search_result")
    ledger.goal.hard_constraints = {
        "origin": "北京",
        "destination": "上海",
        "travel_days": 3,
        "start_date": "2026-09-01",
        "end_date": "2026-09-03",
    }

    outcome = await TravelActionExecutor(tools).execute(
        task=ledger.task_graph.get("task"),
        action=PolicyAction(action_id="transport-1", action="search_transport"),
        ledger=ledger,
    )

    assert outcome.facts[0].key == "transport_time_windows"
    assert outcome.facts[0].value["daily_start_minutes"] == [750, 480, 480]
    assert outcome.facts[0].value["daily_end_minutes"] == [1260, 1260, 990]
    assert outcome.artifacts[0].payload["planning_constraints"]["applied"] is True


def test_search_hydration_preserves_explicit_grounded_recovery_keywords():
    ledger = _ledger("search_pois")
    ledger.goal.hard_constraints = {"destination": "Shanghai"}
    ledger.goal.soft_preferences = {"interests": ["history", "museum"]}
    executor = TravelActionExecutor(AsyncMock())

    narrowed = executor._hydrate_arguments(
        ledger,
        PolicyAction(action="search_pois", arguments={"keywords": ["museum"]}),
    )
    defaulted = executor._hydrate_arguments(
        ledger,
        PolicyAction(action="search_pois", arguments={}),
    )

    assert narrowed["keywords"] == ["museum"]
    assert defaulted["keywords"] == ["history", "museum"]
    assert narrowed["city"] == "Shanghai"
    assert narrowed["category"] is None


@pytest.mark.asyncio
async def test_search_rejects_empty_candidate_evidence():
    tools = AsyncMock()
    tools.execute.return_value = [
        {
            "observation": {
                "ok": True,
                "tool": "search_pois",
                "data": [],
                "source": "unavailable",
                "confidence": 0,
                "tool_call_id": "call-empty",
            }
        }
    ]
    ledger = _ledger("search_pois")

    outcome = await TravelActionExecutor(tools).execute(
        task=ledger.task_graph.get("task"),
        action=PolicyAction(action_id="call-empty", action="search_pois"),
        ledger=ledger,
    )

    assert outcome.status == "failed"
    assert outcome.error_code == "CANDIDATE_POIS_EMPTY"
    assert not outcome.facts
    assert not outcome.artifacts


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
    assert outcome.artifacts[0].payload["_evidence_source"] == "api"
    assert outcome.artifacts[0].payload["_is_fallback"] is False
    assert outcome.tool_calls_used == 1


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


def test_validation_uses_same_effective_landmark_window_as_solver():
    ledger = _ledger("validate_itinerary", "validation_report")
    ledger.goal.hard_constraints = {
        "destination": "上海",
        "travel_days": 1,
        "start_date": "2026-10-25",
    }
    ledger.artifacts["candidates"] = ArtifactRecord(
        artifact_id="candidates",
        artifact_type="poi_candidate_set",
        payload={
            "pois": [
                {
                    "id": "disney",
                    "name": "上海迪士尼度假区",
                    "category": "attraction",
                    "open_time": "08:00",
                    "close_time": "18:00",
                    "recommended_hours": "12",
                }
            ]
        },
        goal_version=1,
        plan_version=1,
    )
    executor = TravelActionExecutor(AsyncMock())

    arguments = executor._hydrate_arguments(ledger, PolicyAction(action="validate_itinerary"))

    assert arguments["facts"][0]["id"] == "disney"
    assert arguments["facts"][0]["duration_minutes"] == 720
    assert arguments["facts"][0]["open_time"] == "08:00"
    assert arguments["facts"][0]["close_time"] == "21:30"


def test_solver_hydration_excludes_restaurants_and_injects_meal_windows():
    ledger = _ledger("solve_itinerary", "solver_result")
    ledger.goal.hard_constraints = {
        "destination": "Shanghai",
        "travel_days": 2,
        "start_date": "2026-08-17",
    }
    ledger.artifacts["candidates"] = ArtifactRecord(
        artifact_id="candidates",
        artifact_type="poi_candidate_set",
        payload={
            "pois": [
                {"name": "Museum", "category": "attraction"},
                {"name": "Restaurant", "category": "restaurant"},
            ]
        },
        goal_version=1,
        plan_version=1,
    )

    arguments = TravelActionExecutor(AsyncMock())._hydrate_arguments(
        ledger, PolicyAction(action="solve_itinerary")
    )

    assert [poi["name"] for poi in arguments["pois"]] == ["Museum"]
    assert arguments["constraints"]["include_restaurant"] is True
    assert arguments["constraints"]["meals_per_day"] == 2
    assert arguments["constraints"]["day_weekdays"] == [0, 1]


def test_solver_hydration_resolves_must_visit_and_filters_forbidden_pois():
    ledger = _ledger("solve_itinerary", "solver_result")
    ledger.goal.hard_constraints = {
        "destination": "上海",
        "travel_days": 2,
        "must_visit": ["上海博物馆"],
        "must_not_visit": ["外滩"],
        "max_walk_minutes": 40,
    }
    ledger.artifacts["candidates"] = ArtifactRecord(
        artifact_id="candidates",
        artifact_type="poi_candidate_set",
        payload={
            "pois": [
                {"id": "bund-1", "name": "上海外滩", "category": "attraction"},
                *[
                    {
                        "id": f"scenic-{index}",
                        "name": f"普通景点{index}",
                        "category": "attraction",
                    }
                    for index in range(9)
                ],
                {"id": "museum-1", "name": "上海博物馆人民广场馆", "category": "attraction"},
            ]
        },
        goal_version=1,
        plan_version=1,
    )

    arguments = TravelActionExecutor(AsyncMock())._hydrate_arguments(
        ledger, PolicyAction(action="solve_itinerary")
    )

    assert arguments["pois"][0]["id"] == "museum-1"
    assert "bund-1" not in {poi["id"] for poi in arguments["pois"]}
    assert arguments["constraints"]["must_visit"] == ["museum-1"]
    assert arguments["constraints"]["must_not_visit"] == ["外滩"]
    assert arguments["constraints"]["max_walk_minutes"] == 40


def test_solver_uses_verified_detail_subset_instead_of_all_search_hits():
    ledger = _ledger("solve_itinerary", "solver_result")
    ledger.goal.hard_constraints = {"destination": "Shanghai", "travel_days": 2}
    ledger.artifacts["candidates"] = ArtifactRecord(
        artifact_id="candidates",
        artifact_type="poi_candidate_set",
        payload={"pois": [{"name": f"Raw {index}"} for index in range(30)]},
        goal_version=1,
        plan_version=1,
    )
    ledger.artifacts["details"] = ArtifactRecord(
        artifact_id="details",
        artifact_type="poi_detail_set",
        payload={
            "details": [
                {
                    "name": f"Raw {index}",
                    "category": "attraction",
                    "ticket_price": 42,
                }
                for index in range(8)
            ],
            "expected_count": 8,
        },
        goal_version=1,
        plan_version=1,
    )

    arguments = TravelActionExecutor(AsyncMock())._hydrate_arguments(
        ledger, PolicyAction(action="solve_itinerary")
    )

    assert len(arguments["pois"]) == 8
    assert arguments["pois"][0]["name"] == "Raw 0"
    assert arguments["pois"][0]["ticket_price"] == 42


def test_details_and_planning_skip_non_plannable_candidates_before_limit():
    ledger = _ledger("solve_itinerary", "solver_result")
    restaurants = [
        {"id": f"r-{index}", "name": f"Restaurant {index}", "category": "restaurant"}
        for index in range(10)
    ]
    attractions = [
        {"id": "a-1", "name": "Museum", "category": "attraction"},
        {"id": "a-2", "name": "Park", "category": "attraction"},
    ]
    ledger.artifacts["candidates"] = ArtifactRecord(
        artifact_id="candidates",
        artifact_type="poi_candidate_set",
        payload={"pois": [*restaurants, *attractions]},
        goal_version=1,
        plan_version=1,
    )
    ledger.artifacts["details"] = ArtifactRecord(
        artifact_id="details",
        artifact_type="poi_detail_set",
        payload={
            "details": [{"name": "Museum"}, {"name": "Park"}],
            "expected_count": 2,
        },
        goal_version=1,
        plan_version=1,
    )

    arguments = TravelActionExecutor(AsyncMock())._hydrate_arguments(
        ledger, PolicyAction(action="solve_itinerary")
    )

    assert [item["id"] for item in arguments["pois"]] == ["a-1", "a-2"]


def test_route_matrix_and_solver_keep_identical_candidate_identity_and_order():
    ledger = _ledger("solve_itinerary", "solver_result")
    raw = [
        {"id": "a", "name": "Museum", "category": "attraction"},
        {"id": "r", "name": "Lunch", "category": "restaurant"},
        {"id": "b", "name": "Park", "category": "attraction"},
    ]
    ledger.artifacts["candidates"] = ArtifactRecord(
        artifact_id="candidates",
        artifact_type="poi_candidate_set",
        payload={"pois": raw},
        goal_version=1,
        plan_version=1,
    )
    executor = TravelActionExecutor(AsyncMock())
    route_arguments = executor._hydrate_arguments(ledger, PolicyAction(action="get_route_matrix"))
    ledger.artifacts["details"] = ArtifactRecord(
        artifact_id="details",
        artifact_type="poi_detail_set",
        payload={
            "details": [
                {"name": item["name"], "category": "attraction", "ticket_price": 10} for item in raw
            ],
            "expected_count": 3,
        },
        goal_version=1,
        plan_version=1,
    )
    solve_arguments = executor._hydrate_arguments(ledger, PolicyAction(action="solve_itinerary"))

    assert [item["id"] for item in route_arguments["pois"]] == ["a", "b"]
    assert [item["id"] for item in solve_arguments["pois"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_poi_detail_action_collects_candidate_set_within_budget():
    tools = AsyncMock()
    tools.execute.return_value = [
        {
            "observation": {
                "ok": True,
                "tool": "get_poi_detail",
                "data": {"name": name},
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


@pytest.mark.asyncio
async def test_poi_detail_action_reserves_tool_budget_for_mandatory_gates(monkeypatch):
    monkeypatch.setattr("agentic.action_executor.settings.agentic_poi_detail_limit", 8)
    monkeypatch.setattr("agentic.action_executor.settings.agentic_reserved_gate_tool_calls", 3)
    tools = AsyncMock()
    tools.execute.side_effect = lambda calls, guard_context: [
        {
            "observation": {
                "ok": True,
                "tool": "get_poi_detail",
                "data": {"name": f"POI {index}"},
                "source": "built_in",
                "confidence": 1,
                "tool_call_id": call["id"],
            }
        }
        for index, call in enumerate(calls)
    ]
    ledger = _ledger("get_poi_detail", "poi_detail_set")
    ledger.goal.hard_constraints = {"destination": "Shanghai"}
    ledger.artifacts["candidates"] = ArtifactRecord(
        artifact_id="candidates",
        artifact_type="poi_candidate_set",
        payload={"pois": [{"name": f"POI {index}"} for index in range(20)]},
        goal_version=1,
        plan_version=1,
    )
    # search + weather already consumed two calls from the default budget 16.
    ledger.budget = ledger.budget.consume(tool_calls=2)

    outcome = await TravelActionExecutor(tools).execute(
        task=ledger.task_graph.get("task"),
        action=PolicyAction(action_id="details", action="get_poi_detail"),
        ledger=ledger,
    )

    assert outcome.tool_calls_used == 8
    assert outcome.artifacts[0].payload["expected_count"] == 8


@pytest.mark.asyncio
async def test_poi_detail_action_rejects_cross_entity_evidence():
    tools = AsyncMock()
    tools.execute.return_value = [
        {
            "observation": {
                "ok": True,
                "tool": "get_poi_detail",
                "data": {"name": "中山陵"},
                "source": "built_in",
                "confidence": 1,
                "tool_call_id": "details:0",
            }
        },
        {
            "observation": {
                "ok": True,
                "tool": "get_poi_detail",
                "data": {"name": "中山陵"},
                "source": "built_in",
                "confidence": 1,
                "tool_call_id": "details:1",
            }
        },
    ]
    ledger = _ledger("get_poi_detail", "poi_detail_set")
    ledger.goal.hard_constraints = {"destination": "南京"}
    ledger.artifacts["candidates"] = ArtifactRecord(
        artifact_id="candidates",
        artifact_type="poi_candidate_set",
        payload={"pois": [{"name": "中山陵"}, {"name": "明孝陵"}]},
        goal_version=1,
        plan_version=1,
    )

    outcome = await TravelActionExecutor(tools).execute(
        task=ledger.task_graph.get("task"),
        action=PolicyAction(action_id="details", action="get_poi_detail"),
        ledger=ledger,
    )

    assert outcome.status == "failed"
    assert outcome.error_code == "POI_DETAIL_IDENTITY_MISMATCH"
    assert not outcome.artifacts
