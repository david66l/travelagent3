"""Conditional edge routers for the TravelAgent LangGraph."""

from __future__ import annotations

from typing import Any


MAX_LOOPS = 3


def route_after_gathering(state: dict[str, Any]) -> str:
    """After gathering subgraph: clarify, direct respond, or enter planning."""
    next_action = state.get("next_action", "")
    if next_action == "clarify":
        return "clarify"
    if next_action == "respond":
        return "respond"
    return "planning"


def route_after_profile(state: dict[str, Any]) -> str:
    """After profile recall: write-back ends the flow; otherwise retrieve."""
    if state.get("stage") in ("memory_updated", "completed"):
        return "__end__"
    return "retrieve"


def route_after_retrieve(state: dict[str, Any]) -> str:
    """After RAG: if empty, fallback to output; otherwise plan."""
    if state.get("retrieval_empty") and not (state.get("poi_candidates") or []):
        state["warnings"] = (state.get("warnings") or []) + ["未检索到合适景点，将使用兜底数据生成行程。"]
        return "plan"
    return "plan"


def route_after_weather(state: dict[str, Any]) -> str:
    """After weather check: always proceed to plan."""
    return "plan"


def route_after_plan(state: dict[str, Any]) -> str:
    """After planner: show draft to user, or proceed to deep enrichment."""
    next_action = state.get("next_action", "")
    if next_action == "clarify":
        return "output"
    if next_action == "confirm_draft":
        return "output"       # Phase 1: show draft, wait for confirmation
    if next_action == "enrich":
        return "tool_call"    # Phase 2: user confirmed → deep lookup
    return "tool_call"


def route_after_tool_call(state: dict[str, Any]) -> str:
    """After tool execution: proceed to fact check."""
    if state.get("next_action") == "clarify":
        return "output"
    return "factcheck"


def route_after_factcheck(state: dict[str, Any]) -> str:
    """After fact check: loop back to plan if conflicts and loops remain."""
    if state.get("next_action") == "clarify":
        return "output"
    if state.get("next_action") == "planner":
        loops = state.get("loop_count", 0)
        if loops < MAX_LOOPS:
            state["loop_count"] = loops + 1
            return "plan"
        state["warnings"] = (state.get("warnings") or []) + ["事实校验冲突未解决，仍按当前方案输出。"]
    return "hallucination"


def route_after_hallucination(state: dict[str, Any]) -> str:
    """After hallucination check: proceed to output."""
    if state.get("next_action") == "clarify":
        return "output"
    return "output"


def route_after_output(state: dict[str, Any]) -> str:
    """After output: clarify → end; draft → wait for user; final → booking."""
    next_action = state.get("next_action", "")
    if next_action == "clarify":
        return "__end__"
    if next_action == "confirm_draft":
        return "__end__"       # Phase 1: wait for user to confirm/modify
    if next_action == "completed":
        return "booking"       # Phase 2: final output, now do booking
    return "booking"


def route_after_booking(state: dict[str, Any]) -> str:
    """After booking: write back memory and end."""
    return "profile_recall"
