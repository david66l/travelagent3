"""Tests for the TravelAgent LangGraph orchestration layer."""

from unittest.mock import AsyncMock, patch

import pytest

from graph.exceptions import DegradationLevel, NodeException, classify_error
from graph.graph import build_graph
from graph.routers import (
    route_after_apply_change,
    route_after_confirm_gate,
    route_after_factcheck,
    route_after_gathering,
    route_after_hallucination,
    route_after_output,
    route_after_profile,
    route_after_tool_call,
)
from graph.session_manager import SessionManager
from models.travel_slots import SlotParseOutput, TravelSlots


def _make_parse_output(**overrides) -> SlotParseOutput:
    return SlotParseOutput(
        intent=overrides.get("intent", "generate_itinerary"),
        confidence=overrides.get("confidence", 0.9),
        sentiment=overrides.get("sentiment", "neutral"),
        slots=overrides.get("slots", TravelSlots(destination="北京", travel_days=3)),
        missing_slots=overrides.get("missing_slots", []),
        clarifying_question=overrides.get("clarifying_question"),
        disambiguation=overrides.get("disambiguation"),
    )


def test_graph_compiles():
    graph = build_graph(checkpointer=None)
    assert graph is not None


@pytest.mark.asyncio
async def test_gathering_turn_node_with_error_handling():
    from graph.gathering import gathering_turn_node

    state = {"user_input": "hi", "messages": [], "profile": {}}
    with patch(
        "graph.gathering.process_user_turn",
        new=AsyncMock(side_effect=RuntimeError("llm timeout")),
    ):
        result = await gathering_turn_node(state)

    assert result["error_node"] == "gathering_turn"
    assert result["next_action"] == "retry"
    assert "llm timeout" in result["error_message"]


@pytest.mark.asyncio
async def test_plan_node_with_error_handling():
    from graph.nodes import plan_node

    state = {"slots": {}, "profile": {}, "knowledge_results": []}
    with patch(
        "graph.node_impl._planner_async", new=AsyncMock(side_effect=RuntimeError("vrp down"))
    ):
        result = await plan_node(state)

    assert result["error_node"] == "plan"
    assert result["next_action"] == "fallback"


def test_router_after_gathering():
    assert route_after_gathering({"next_action": "clarify"}) == "clarify"
    assert route_after_gathering({"next_action": "respond"}) == "respond"
    assert route_after_gathering({"next_action": "infeasible"}) == "infeasible"
    # Planning now enters at profile_recall, which fans out into retrieve ∥ weather_check.
    assert route_after_gathering({"next_action": "plan"}) == "profile_recall"


def test_router_after_profile_writeback():
    # Planning path fans out into equal-length parallel branches re-joining at plan.
    assert route_after_profile({"stage": "memory_loaded"}) == ["retrieve", "weather_check"]
    assert route_after_profile({"stage": "memory_updated"}) == "__end__"


def test_router_after_confirm_gate():
    assert route_after_confirm_gate({"confirm_decision": "confirm"}) == "tool_call"
    assert route_after_confirm_gate({"confirm_decision": "modify"}) == "apply_single_change"
    assert route_after_confirm_gate({"confirm_decision": None}) == "plan"


def test_router_after_apply_change():
    assert route_after_apply_change({"next_action": "planner"}) == "plan"
    assert route_after_apply_change({"next_action": "fact_check"}) == "factcheck"


def test_router_after_tool_call():
    assert route_after_tool_call({"next_action": "fact_check"}) == "factcheck"
    assert route_after_tool_call({"next_action": "clarify"}) == "output"


def test_router_after_factcheck_is_pure():
    # The router no longer mutates state; the factcheck node owns the loop counter.
    state = {"next_action": "planner", "loop_count": 0}
    assert route_after_factcheck(state) == "plan"
    assert state["loop_count"] == 0  # router must not mutate
    assert route_after_factcheck({"next_action": "factcheck_done"}) == "hallucination"
    assert route_after_factcheck({"stage": "fact_check_done"}) == "hallucination"


@pytest.mark.asyncio
async def test_factcheck_node_owns_loop_guard():
    from graph.node_impl import _fact_check_async

    class _FakeResult:
        def mappings(self):
            return self

        def first(self):
            return {
                "status": "active",
                "ticket_price": 999.0,
                "open_time": "08:00",
                "close_time": "18:00",
            }

    class _FakeDB:
        async def execute(self, *a, **k):
            return _FakeResult()

    class _FakeMaker:
        def __call__(self):
            return self

        async def __aenter__(self):
            return _FakeDB()

        async def __aexit__(self, *a):
            return False

    itinerary = [{"activities": [{"poi_name": "故宫", "ticket_price": 10}]}]
    with patch("core.database.async_session_maker", new=_FakeMaker()):
        # Within budget → replan and advance the counter (node, not router).
        r0 = await _fact_check_async({"itinerary": itinerary, "loop_count": 0, "max_loops": 3})
        assert r0["next_action"] == "planner"
        assert r0["loop_count"] == 1

        # Budget exhausted → give up, keep plan, surface the unresolved conflict.
        r3 = await _fact_check_async({"itinerary": itinerary, "loop_count": 3, "max_loops": 3})
        assert r3["next_action"] == "factcheck_done"
        assert any("未解决" in w for w in r3["warnings"])


def test_router_after_hallucination():
    assert route_after_hallucination({"next_action": "respond"}) == "output"
    assert route_after_hallucination({"next_action": "clarify"}) == "output"


def test_router_after_output():
    # Non-planning turns end here.
    assert route_after_output({"next_action": "clarify"}) == "__end__"
    assert route_after_output({"next_action": "respond"}) == "__end__"
    assert route_after_output({"next_action": "infeasible"}) == "__end__"
    # Draft (no decision yet) / post-modify must pause for an explicit decision.
    assert route_after_output({"next_action": "fact_check"}) == "confirm_gate"
    assert route_after_output({"confirm_decision": "modify"}) == "confirm_gate"
    # Only an explicitly confirmed plan proceeds to booking.
    assert route_after_output({"confirm_decision": "confirm"}) == "booking"


@pytest.mark.asyncio
async def test_constraint_change_updates_profile_and_requires_fresh_plan():
    from core.conversation_state import flatten_profile
    from graph.nodes import apply_single_change_node

    state = {
        "profile": {"destination": "成都", "travel_days": 3},
        "slots": {"destination": "成都", "travel_days": 3},
        "itinerary": [{"day_number": 1, "activities": []}],
        "pending_change": {"action": "set_budget", "value": 6000},
    }
    result = await apply_single_change_node(state)

    assert flatten_profile(result["profile"])["budget_range"] == 6000
    assert result["slots"]["total_budget"] == 6000
    assert result["next_action"] == "planner"
    assert result["confirm_decision"] is None


def test_replan_closure_replaces_the_closed_poi_in_same_slot():
    from graph.nodes import _trace_replan_local

    itinerary = [
        {
            "day_number": 1,
            "activities": [{"poi_name": "宽窄巷子", "start_time": "09:00", "end_time": "11:00"}],
        }
    ]
    candidates = [
        {
            "spot_name": "杜甫草堂",
            "category": "attraction",
            "ticket_price": 50,
            "lat": 30.66,
            "lng": 104.03,
        }
    ]

    changed, note = _trace_replan_local(
        {"type": "closure", "poi": "宽窄巷子"}, itinerary, candidates
    )

    activity = changed[0]["activities"][0]
    assert activity["poi_name"] == "杜甫草堂"
    assert activity["start_time"] == "09:00"
    assert "替换" in note
    assert itinerary[0]["activities"][0]["poi_name"] == "宽窄巷子"


def test_replan_weather_reorders_and_delay_shifts_real_times():
    from graph.nodes import _trace_replan_local

    itinerary = [
        {
            "day_number": 1,
            "activities": [
                {"poi_name": "成都博物馆", "start_time": "09:00", "end_time": "11:00"},
                {"poi_name": "人民公园", "start_time": "11:30", "end_time": "13:00"},
            ],
        }
    ]
    weather, _ = _trace_replan_local({"type": "weather", "detail": "下午有雨"}, itinerary)
    assert [a["poi_name"] for a in weather[0]["activities"]] == ["人民公园", "成都博物馆"]
    assert weather[0]["activities"][0]["start_time"] == "09:00"

    delayed, note = _trace_replan_local(
        {"type": "delay", "detail": "晚到1.5小时", "day_number": 1}, itinerary
    )
    assert delayed[0]["activities"][0]["start_time"] == "10:30"
    assert delayed[0]["activities"][1]["end_time"] == "14:30"
    assert "90 分钟" in note


def test_error_classification():
    exc = classify_error("plan", RuntimeError("LLM call timed out"))
    assert exc.level == DegradationLevel.RETRY

    exc = classify_error("gathering_turn", RuntimeError("content filter refusal"))
    assert exc.level == DegradationLevel.ESCALATE


def test_node_exception_carries_state_patch():
    exc = NodeException(
        "retrieve",
        "db down",
        level=DegradationLevel.FALLBACK,
        state_patch={"retrieval_empty": True},
    )
    assert exc.state_patch["retrieval_empty"] is True


@pytest.mark.asyncio
async def test_session_manager_create_and_load():
    sm = SessionManager(ttl_seconds=60)
    with patch("graph.session_manager.memory_manager.hot_set", new=AsyncMock()) as mock_set:
        state = await sm.create("s1", "u1", "北京3天")
        assert state["user_id"] == "u1"
        mock_set.assert_awaited_once()

    with patch(
        "graph.session_manager.memory_manager.hot_get",
        new=AsyncMock(return_value={"stage": "planned"}),
    ):
        loaded = await sm.load("s1")
        assert loaded["stage"] == "planned"
