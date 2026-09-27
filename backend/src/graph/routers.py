"""Conditional edge routers for the TravelAgent LangGraph."""

from __future__ import annotations

from typing import Any


MAX_LOOPS = 3


def route_after_gathering(state: dict[str, Any]) -> str:
    """After gathering: clarify/respond/infeasible, or enter planning.

    Planning starts at profile_recall, which then fans out to retrieve AND
    weather_check in parallel. The fan-out is deliberately placed *after*
    profile_recall so both parallel branches are the same length (one hop to
    ``plan``); equal-length branches re-join in a single LangGraph superstep,
    avoiding the duplicate ``plan``/``output`` execution and the resulting
    INVALID_CONCURRENT_GRAPH_UPDATE on the ``itinerary`` channel.
    """
    next_action = state.get("next_action", "")
    if next_action == "clarify":
        return "clarify"
    if next_action == "respond":
        return "respond"
    if next_action == "infeasible":
        return "infeasible"
    return "profile_recall"


def route_after_profile(state):
    return "__end__" if state.get("stage") in {"memory_updated", "completed"} else "agent_loop"


def route_after_agent_loop(state: dict[str, Any]) -> str:
    """Checkpoint every Agent batch and never hide failure with a second planner."""
    if state.get("agent_status") == "running":
        return "agent_loop"
    if state.get("agent_status") == "awaiting_information":
        return "output"
    if state.get("agent_status") in {"awaiting_confirmation", "finished"} and state.get(
        "itinerary"
    ):
        return "output"
    return "output"


def route_after_confirm_gate(state):
    return (
        "agent_loop"
        if state.get("confirm_decision") in {"modify", "reject_with_reason"}
        else "output"
    )


def route_after_output(state: dict[str, Any]) -> str:
    """After output: pause drafts for HITL; finalize only confirmed plans."""
    next_action = state.get("next_action", "")
    if next_action in ("clarify", "respond", "infeasible", "agent_error"):
        return "__end__"
    if next_action == "agent_draft":
        return "confirm_gate"
    if state.get("confirm_decision") == "confirm":
        return "booking"  # final output after enrichment
    return "confirm_gate"  # initial / modified draft must be accepted first


def route_after_booking(state: dict[str, Any]) -> str:
    """After booking: write back memory and end."""
    return "profile_recall"
