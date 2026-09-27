"""Tests for the TravelAgent LangGraph orchestration layer."""

from unittest.mock import AsyncMock, patch

import pytest

from graph.exceptions import DegradationLevel, NodeException, classify_error
from graph.graph import build_graph
from graph.routers import (
    route_after_agent_loop,
    route_after_confirm_gate,
    route_after_gathering,
    route_after_output,
    route_after_profile,
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


def test_router_after_gathering():
    assert route_after_gathering({"next_action": "clarify"}) == "clarify"
    assert route_after_gathering({"next_action": "respond"}) == "respond"
    assert route_after_gathering({"next_action": "infeasible"}) == "infeasible"
    # Planning now enters at profile_recall, which fans out into retrieve ∥ weather_check.
    assert route_after_gathering({"next_action": "plan"}) == "profile_recall"


def test_router_after_profile_writeback():
    # Planning path fans out into equal-length parallel branches re-joining at plan.
    assert route_after_profile({"stage": "memory_loaded"}) == "agent_loop"
    assert route_after_profile({"stage": "memory_updated"}) == "__end__"
    assert route_after_profile({"policy_mode": "agent"}) == "agent_loop"


def test_router_after_agent_loop_checkpoints_or_stops_without_legacy_fallback():
    assert (
        route_after_agent_loop({"agent_status": "awaiting_confirmation", "itinerary": [{}]})
        == "output"
    )
    assert route_after_agent_loop({"agent_status": "awaiting_information"}) == "output"
    assert route_after_agent_loop({"agent_status": "running"}) == "agent_loop"
    assert route_after_agent_loop({"agent_status": "failed"}) == "output"


def test_router_after_confirm_gate():
    assert route_after_confirm_gate({"confirm_decision": "confirm"}) == "output"
    assert route_after_confirm_gate({"confirm_decision": "modify"}) == "agent_loop"
    assert route_after_confirm_gate({"confirm_decision": None}) == "output"
    assert route_after_confirm_gate({"confirm_decision": "reject_needs_reason"}) == "output"
    assert route_after_confirm_gate({"confirm_decision": "reject_with_reason"}) == "agent_loop"


@pytest.mark.asyncio
async def test_agent_rejection_without_reason_asks_before_replanning():
    from graph.nodes import confirm_gate_node

    state = {
        "policy_mode": "agent",
        "agent_ledger": {"present": True},
        "itinerary": [{"day_number": 1, "activities": []}],
    }
    with patch("langgraph.types.interrupt", return_value={"action": "reject"}):
        result = await confirm_gate_node(state)

    assert result["agent_status"] == "awaiting_revision_reason"
    assert result["next_action"] == "clarify"
    assert result["clarification_questions"]


@pytest.mark.asyncio
async def test_agent_confirmation_closes_global_completion_gate():
    from tests.unit.agentic.test_harness_loop import ledger, artifact, seed_evidence
    from graph.nodes import confirm_gate_node

    state = ledger()
    seed_evidence(state)
    artifact(state, "solver_result", {"days": []})
    artifact(state, "validation_report", {"hard_pass": True, "hard_violations": []})
    artifact(state, "finish", {})
    state.task_graph = state.task_graph.model_copy(
        update={"tasks": (state.task_graph.tasks[0].model_copy(update={"status": "blocked"}),)}
    )
    with patch("langgraph.types.interrupt", return_value={"action": "confirm"}):
        result = await confirm_gate_node(
            {
                "policy_mode": "agent",
                "agent_status": "awaiting_confirmation",
                "agent_ledger": state.model_dump(mode="json"),
                "itinerary": [],
            }
        )
    assert result["agent_status"] == "finished"
    assert result["next_action"] == "agent_final"


@pytest.mark.asyncio
async def test_fallback_draft_confirmation_does_not_close_failed_agent_ledger():
    from graph.nodes import confirm_gate_node

    state = {
        "policy_mode": "agent",
        "agent_status": "fallback",
        "agent_ledger": {"not": "a valid ledger"},
        "itinerary": [{"day_number": 1, "activities": []}],
    }

    with patch("langgraph.types.interrupt", return_value={"action": "confirm"}):
        result = await confirm_gate_node(state)

    assert result["agent_status"] == "failed"
    assert result["next_action"] == "agent_error"
    assert "agent_ledger" not in result


def test_router_after_output():
    # Non-planning turns end here.
    assert route_after_output({"next_action": "clarify"}) == "__end__"
    assert route_after_output({"next_action": "respond"}) == "__end__"
    assert route_after_output({"next_action": "infeasible"}) == "__end__"
    assert route_after_output({"next_action": "agent_draft"}) == "confirm_gate"
    # Draft (no decision yet) / post-modify must pause for an explicit decision.
    assert route_after_output({"next_action": "fact_check"}) == "confirm_gate"
    assert route_after_output({"confirm_decision": "modify"}) == "confirm_gate"
    # Only an explicitly confirmed plan proceeds to booking.
    assert route_after_output({"confirm_decision": "confirm"}) == "booking"


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
