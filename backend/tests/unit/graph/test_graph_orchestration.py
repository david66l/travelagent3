"""Tests for the TravelAgent LangGraph orchestration layer."""

from unittest.mock import AsyncMock, patch

import pytest

from graph.exceptions import DegradationLevel, NodeException, classify_error
from graph.graph import build_graph
from graph.routers import (
    route_after_factcheck,
    route_after_gathering,
    route_after_hallucination,
    route_after_output,
    route_after_plan,
    route_after_profile,
    route_after_retrieve,
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
    with patch("graph.node_impl._planner_async", new=AsyncMock(side_effect=RuntimeError("vrp down"))):
        result = await plan_node(state)

    assert result["error_node"] == "plan"
    assert result["next_action"] == "fallback"


def test_router_after_gathering():
    assert route_after_gathering({"next_action": "clarify"}) == "clarify"
    assert route_after_gathering({"next_action": "respond"}) == "respond"
    assert route_after_gathering({"next_action": "plan"}) == "planning"


def test_router_after_profile_writeback():
    assert route_after_profile({"stage": "memory_loaded"}) == "retrieve"
    assert route_after_profile({"stage": "memory_updated"}) == "__end__"


def test_router_after_plan():
    assert route_after_plan({"next_action": "fact_check"}) == "tool_call"
    assert route_after_plan({"next_action": "clarify"}) == "output"


def test_router_after_tool_call():
    assert route_after_tool_call({"next_action": "fact_check"}) == "factcheck"
    assert route_after_tool_call({"next_action": "clarify"}) == "output"


def test_router_after_factcheck_loop_guard():
    state = {"next_action": "planner", "loop_count": 0}
    assert route_after_factcheck(state) == "plan"
    assert state["loop_count"] == 1

    state["loop_count"] = 3
    assert route_after_factcheck(state) == "hallucination"


def test_router_after_hallucination():
    assert route_after_hallucination({"next_action": "respond"}) == "output"
    assert route_after_hallucination({"next_action": "clarify"}) == "output"


def test_router_after_output():
    assert route_after_output({"next_action": "clarify"}) == "__end__"
    assert route_after_output({"next_action": "respond"}) == "booking"


def test_error_classification():
    exc = classify_error("plan", RuntimeError("LLM call timed out"))
    assert exc.level == DegradationLevel.RETRY

    exc = classify_error("gathering_turn", RuntimeError("content filter refusal"))
    assert exc.level == DegradationLevel.ESCALATE


def test_node_exception_carries_state_patch():
    exc = NodeException("retrieve", "db down", level=DegradationLevel.FALLBACK, state_patch={"retrieval_empty": True})
    assert exc.state_patch["retrieval_empty"] is True


@pytest.mark.asyncio
async def test_session_manager_create_and_load():
    sm = SessionManager(ttl_seconds=60)
    with patch("graph.session_manager.memory_manager.hot_set", new=AsyncMock()) as mock_set:
        state = await sm.create("s1", "u1", "北京3天")
        assert state["user_id"] == "u1"
        mock_set.assert_awaited_once()

    with patch("graph.session_manager.memory_manager.hot_get", new=AsyncMock(return_value={"stage": "planned"})):
        loaded = await sm.load("s1")
        assert loaded["stage"] == "planned"
