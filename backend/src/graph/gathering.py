"""Gathering-phase LangGraph subgraph — intent recognition and slot filling."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from langgraph.graph import END, StateGraph

from core.conversation_state import (
    default_conversation_state,
    flatten_profile,
    is_profile_ready,
)
from core.conversation_turn import process_user_turn
from core.langsmith_trace import traceable_step
from graph.exceptions import with_error_handling
from graph.models import AgentState

logger = logging.getLogger(__name__)

_PLANNING_INTENTS = frozenset({"generate_itinerary", "modify_itinerary", "confirm_itinerary"})

INTENT_READY_MESSAGE = "意图识别已完成，接下来将进行大致的规划。"


def conversation_state_from_agent(agent: dict[str, Any]) -> dict[str, Any]:
    """Rebuild WS-style conversation state from graph checkpoint + session payload."""
    conv = default_conversation_state()
    embedded = agent.get("_conversation_state")
    if isinstance(embedded, dict):
        conv.update(deepcopy(embedded))
    elif agent.get("profile"):
        conv["profile"] = deepcopy(agent.get("profile") or {})

    for key in (
        "turn",
        "recent_messages",
        "last_intent",
        "missing_required",
        "phase",
        "user_id",
        "user_role",
        "slots",
        "inferred_slots",
        "feasibility_report",
    ):
        if key in agent and agent[key] is not None:
            conv[key] = agent[key]
    return conv


def conversation_sync_from_state(conv: dict[str, Any], *, next_action: str) -> dict[str, Any]:
    """Fields that must be persisted back to Redis / memory after a gathering turn."""
    phase = "gathering" if next_action == "clarify" else "planning"
    return {
        "turn": conv.get("turn", 0),
        "recent_messages": conv.get("recent_messages", []),
        "last_intent": conv.get("last_intent"),
        "missing_required": conv.get("missing_required", []),
        "phase": phase,
        "slots": conv.get("slots"),
        "inferred_slots": conv.get("inferred_slots"),
        "feasibility_report": conv.get("feasibility_report"),
    }


async def gathering_turn_impl(state: dict[str, Any]) -> dict[str, Any]:
    """Run the same path as ``process_user_turn`` inside LangGraph."""
    return await _gathering_turn_traced(state)


@traceable_step("intent/gathering_turn", run_type="chain")
async def _gathering_turn_traced(state: dict[str, Any]) -> dict[str, Any]:
    conv = conversation_state_from_agent(state)
    if state.get("user_id"):
        conv["user_id"] = state["user_id"]
    if state.get("user_role"):
        conv["user_role"] = state["user_role"]

    was_gathering = conv.get("phase") == "gathering"
    result = await process_user_turn(conv, state.get("user_input", ""))

    merged_flat = flatten_profile(conv.get("profile") or {})
    profile_ready = is_profile_ready(merged_flat)

    if result.intent in _PLANNING_INTENTS:
        if not profile_ready:
            next_action = "clarify"
        elif (result.feasibility_report or {}).get("issues"):
            # Hard feasibility conflicts → gate before planning.
            next_action = "infeasible"
        else:
            next_action = "plan"
    else:
        next_action = "respond"

    stage = "gathering" if next_action == "clarify" else "demand_parsed"

    patch: dict[str, Any] = {
        "profile": conv.get("profile"),
        "intent": result.intent,
        "confidence": result.confidence,
        "sentiment": result.sentiment,
        "slots": result.slots,
        "missing_slots": result.missing_required,
        "clarification_questions": result.clarification_questions,
        "feasibility_report": result.feasibility_report,
        "next_action": next_action,
        "stage": stage,
        "conversation_sync": conversation_sync_from_state(conv, next_action=next_action),
        "_conversation_state": conv,
    }
    if state.get("session_id"):
        patch["session_id"] = state["session_id"]
    if next_action == "plan" and was_gathering:
        patch["intent_ready_message"] = INTENT_READY_MESSAGE
    return patch


@with_error_handling("gathering_turn")
async def gathering_turn_node(state: dict[str, Any]) -> dict[str, Any]:
    return await gathering_turn_impl(state)


def build_gathering_subgraph():
    """Compile the first-phase subgraph: one turn of intent/slot gathering."""
    builder = StateGraph(AgentState)
    builder.add_node("gathering_turn", gathering_turn_node)
    builder.set_entry_point("gathering_turn")
    builder.add_edge("gathering_turn", END)
    return builder.compile()
