"""Unit tests for shared LangGraph node implementations."""

from unittest.mock import AsyncMock, patch

import pytest

from graph.node_impl import _planner_async, _rag_async


def _make_poi_dict(spot_id: str, name: str) -> dict:
    return {
        "spot_id": spot_id,
        "spot_name": name,
        "spot_type": "attraction",
        "city": "北京",
        "lat": 39.9,
        "lng": 116.4,
        "ticket_price": 60.0,
        "tags": ["历史"],
        "duration_minutes": 180,
        "walk_intensity": 3,
        "need_reservation": False,
        "reservation_reminder": False,
    }


@pytest.mark.asyncio
async def test_rag_retrieval_node_returns_poi_state():
    state = {
        "slots": {"destination": "北京", "travel_days": 3, "interests": ["历史"]},
        "profile": {"interests": ["文化"]},
    }

    mock_result = {
        "poi_candidates": [_make_poi_dict("p1", "故宫"), _make_poi_dict("p2", "天坛")],
        "retrieval_query": "北京 历史 文化 旅游景点",
        "retrieval_empty": False,
        "retrieval_stats": {
            "structured_count": 2,
            "vector_count": 0,
            "bm25_count": 0,
            "merged_count": 2,
        },
    }

    with patch("agents.rag_retrieval.TravelRetrievalRAGAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.retrieve = AsyncMock(return_value=mock_result)
        result = await _rag_async(state)

    assert result["retrieval_empty"] is False
    assert result["retrieval_query"]
    assert len(result["poi_candidates"]) == 2
    assert len(result["knowledge_results"]) == 2
    assert result["knowledge_results"][0]["name"] == "故宫"
    instance.retrieve.assert_awaited_once()


@pytest.mark.asyncio
async def test_rag_retrieval_node_empty_when_no_destination():
    state = {"slots": {"travel_days": 3}, "profile": {}}
    result = await _rag_async(state)
    assert result["retrieval_empty"] is True
    assert result["poi_candidates"] == []
    assert result["knowledge_results"] == []


@pytest.mark.asyncio
async def test_rag_retrieval_node_graceful_on_agent_failure():
    state = {
        "slots": {"destination": "北京", "travel_days": 3},
        "profile": {},
    }

    with patch("agents.rag_retrieval.TravelRetrievalRAGAgent") as MockAgent:
        instance = MockAgent.return_value
        instance.retrieve = AsyncMock(side_effect=RuntimeError("db down"))
        result = await _rag_async(state)

    assert result["retrieval_empty"] is True
    assert result["poi_candidates"] == []
    assert result["knowledge_results"] == []


@pytest.mark.asyncio
async def test_planner_node_calls_vrp_service_and_returns_itinerary():
    from vrp_solver_service.models import SolverResponse, DayPlanOutput, ActivityOutput

    state = {
        "slots": {"destination": "北京", "travel_days": 2, "must_visit": ["故宫"]},
        "profile": {"trip": {"destination": "北京", "travel_days": 2}},
        "knowledge_results": [
            _make_poi_dict("p1", "故宫"),
            _make_poi_dict("p2", "天坛"),
        ],
    }

    mock_response = SolverResponse(
        status="optimal",
        days=[
            DayPlanOutput(
                day_number=1,
                activities=[
                    ActivityOutput(
                        poi_id="p1",
                        poi_name="故宫",
                        category="attraction",
                        start_time="08:00",
                        end_time="11:00",
                        duration_min=180,
                        ticket_price=60.0,
                        lat=39.9,
                        lng=116.4,
                        tags=["历史"],
                    ),
                ],
                total_cost=60.0,
            ),
            DayPlanOutput(day_number=2, activities=[]),
        ],
        solve_time_ms=120,
    )

    with patch(
        "vrp_solver_service.client.VRPSolverClient.solve", new=AsyncMock(return_value=mock_response)
    ):
        result = await _planner_async(state)

    assert result["stage"] == "planned"
    assert result["next_action"] == "fact_check"
    assert result["solve_status"] == "optimal"
    assert len(result["itinerary"]) == 2
    assert result["itinerary"][0]["activities"][0]["poi_name"] == "故宫"


@pytest.mark.asyncio
async def test_planner_node_falls_back_when_vrp_service_fails():
    state = {
        "slots": {"destination": "北京", "travel_days": 1},
        "profile": {"trip": {"destination": "北京", "travel_days": 1}},
        "knowledge_results": [
            _make_poi_dict("p1", "故宫"),
        ],
    }

    with patch(
        "vrp_solver_service.client.VRPSolverClient.solve",
        new=AsyncMock(side_effect=RuntimeError("connection refused")),
    ):
        result = await _planner_async(state)

    assert result["stage"] == "planned"
    assert result["next_action"] == "fact_check"
    assert result["solve_status"] == "local_fallback"
