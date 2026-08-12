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


def route_after_profile(state: dict[str, Any]) -> str | list[str]:
    """After profile recall: write-back ends the flow; otherwise fan out.

    The planning path fans out into retrieve (personalised RAG, depends on the
    recalled profile) and weather_check (needs only the gathered slots). Both
    are single hops to ``plan`` so they re-join synchronously.
    """
    if state.get("stage") in ("memory_updated", "completed"):
        return "__end__"
    return ["retrieve", "weather_check"]


def route_after_retrieve(state: dict[str, Any]) -> str:
    """After RAG: always proceed to plan (planner handles empty via fallback)."""
    return "plan"


def route_after_weather(state: dict[str, Any]) -> str:
    """After weather check: always proceed to plan."""
    return "plan"


def route_after_confirm_gate(state: dict[str, Any]) -> str:
    """After the confirm interrupt resumes: enrich, modify, or re-solve."""
    decision = state.get("confirm_decision")
    if decision == "modify":
        return "apply_single_change"
    if decision is None:
        return "plan"  # reject → re-solve a fresh draft
    return "tool_call"  # confirm → deep enrichment


def route_after_apply_change(state: dict[str, Any]) -> str:
    """Constraint changes need a fresh solve; POI edits only need validation."""
    if state.get("next_action") == "planner":
        return "plan"
    return "factcheck"


def route_after_tool_call(state: dict[str, Any]) -> str:
    """After tool execution: proceed to fact check."""
    if state.get("next_action") == "clarify":
        return "output"
    return "factcheck"


def route_after_factcheck(state: dict[str, Any]) -> str:
    """After fact check: pure router. The node owns the loop counter + warnings.

    ``next_action == "planner"`` means the factcheck node decided a replan is
    worthwhile (conflicts found and loop budget not exhausted).
    """
    if state.get("next_action") == "clarify":
        return "output"
    if state.get("next_action") == "planner":
        return "plan"
    return "hallucination"


def route_after_hallucination(state: dict[str, Any]) -> str:
    """After hallucination check: proceed to output."""
    if state.get("next_action") == "clarify":
        return "output"
    return "output"


def route_after_output(state: dict[str, Any]) -> str:
    """After output: pause drafts for HITL; finalize only confirmed plans."""
    next_action = state.get("next_action", "")
    if next_action in ("clarify", "respond", "infeasible"):
        return "__end__"
    if state.get("confirm_decision") == "confirm":
        return "booking"  # final output after enrichment
    return "confirm_gate"  # initial / modified draft must be accepted first


def route_after_booking(state: dict[str, Any]) -> str:
    """After booking: write back memory and end."""
    return "profile_recall"
