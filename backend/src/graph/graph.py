"""LangGraph StateGraph assembly for TravelAgent Step 6."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from graph.gathering import build_gathering_subgraph
from graph.models import AgentState
from graph.nodes import (
    agent_loop_node,
    apply_single_change_node,
    booking_node,
    confirm_gate_node,
    error_handler_node,
    factcheck_node,
    hallucination_check_node,
    human_interrupt_node,
    output_node,
    plan_node,
    profile_node,
    replan_local_node,
    retrieve_node,
    tool_call_node,
    weather_check_node,
)
from graph.routers import (
    route_after_agent_loop,
    route_after_booking,
    route_after_apply_change,
    route_after_confirm_gate,
    route_after_factcheck,
    route_after_gathering,
    route_after_hallucination,
    route_after_output,
    route_after_profile,
    route_after_retrieve,
    route_after_tool_call,
    route_after_weather,
)

logger = logging.getLogger(__name__)


def build_graph(checkpointer: Optional[Any] = None) -> StateGraph:
    """Build and compile the TravelAgent StateGraph.

    Flow:
        gathering -> (clarify/respond -> output | planning -> profile -> retrieve -> ...)
    """
    builder = StateGraph(AgentState)

    gathering_subgraph = build_gathering_subgraph()
    builder.add_node("gathering", gathering_subgraph)
    builder.add_node("profile_recall", profile_node)
    builder.add_node("agent_loop", agent_loop_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("weather_check", weather_check_node)
    builder.add_node("plan", plan_node)
    builder.add_node("confirm_gate", confirm_gate_node)
    builder.add_node("tool_call", tool_call_node)
    builder.add_node("factcheck", factcheck_node)
    builder.add_node("hallucination", hallucination_check_node)
    builder.add_node("output", output_node)
    builder.add_node("booking", booking_node)
    builder.add_node("apply_single_change", apply_single_change_node)
    builder.add_node("replan_local", replan_local_node)
    builder.add_node("human_interrupt", human_interrupt_node)
    builder.add_node("error_handler", error_handler_node)

    builder.set_entry_point("gathering")

    builder.add_conditional_edges(
        "gathering",
        route_after_gathering,
        {
            "clarify": "output",
            "respond": "output",
            "infeasible": "output",
            # Planning enters at profile_recall, which then fans out.
            "profile_recall": "profile_recall",
        },
    )

    # profile_recall fans out into retrieve ∥ weather_check (equal-length branches
    # that re-join at plan in a single superstep), or ends on memory write-back.
    builder.add_conditional_edges(
        "profile_recall",
        route_after_profile,
        {
            "agent_loop": "agent_loop",
            "retrieve": "retrieve",
            "weather_check": "weather_check",
            "__end__": END,
        },
    )

    builder.add_conditional_edges(
        "agent_loop",
        route_after_agent_loop,
        {
            "agent_loop": "agent_loop",
            "output": "output",
        },
    )

    # retrieve and weather_check are parallel branches that both re-join at plan;
    # plan runs once, after both have completed.
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"plan": "plan"},
    )

    builder.add_conditional_edges(
        "weather_check",
        route_after_weather,
        {"plan": "plan"},
    )

    # plan → output (always show the draft); output then routes to confirm_gate
    builder.add_edge("plan", "output")

    # confirm_gate pauses (interrupt); on resume → enrich / modify / re-solve
    builder.add_conditional_edges(
        "confirm_gate",
        route_after_confirm_gate,
        {
            "tool_call": "tool_call",
            "apply_single_change": "apply_single_change",
            "plan": "plan",
            "agent_loop": "agent_loop",
            "output": "output",
        },
    )

    # Phase 2: tool_call (tickets/hotels/restaurants) → factcheck → hallucination → output → booking
    builder.add_conditional_edges(
        "tool_call",
        route_after_tool_call,
        {"factcheck": "factcheck", "output": "output"},
    )

    builder.add_conditional_edges(
        "factcheck",
        route_after_factcheck,
        {"plan": "plan", "hallucination": "hallucination", "output": "output"},
    )

    builder.add_conditional_edges(
        "hallucination",
        route_after_hallucination,
        {"output": "output"},
    )

    # output → booking (confirmed final) | confirm_gate (draft/post-modify) | END
    builder.add_conditional_edges(
        "output",
        route_after_output,
        {"booking": "booking", "confirm_gate": "confirm_gate", "__end__": END},
    )

    builder.add_conditional_edges(
        "booking",
        route_after_booking,
        {"profile_recall": "profile_recall"},
    )

    builder.add_conditional_edges(
        "apply_single_change",
        route_after_apply_change,
        {"plan": "plan", "factcheck": "factcheck"},
    )
    builder.add_edge("replan_local", "output")
    builder.add_edge("human_interrupt", END)
    builder.add_edge("error_handler", END)

    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


_graph: Optional[StateGraph] = None


def set_graph(graph: StateGraph) -> None:
    """Inject a lifespan-managed compiled graph (e.g. with PostgresSaver)."""
    global _graph
    _graph = graph


async def get_graph() -> StateGraph:
    """Return the compiled graph; fall back to in-memory if not injected."""
    global _graph
    if _graph is not None:
        return _graph

    _graph = build_graph(checkpointer=None)
    logger.warning(
        "TravelAgent graph compiled with in-memory checkpointer; "
        "persistent AsyncPostgresSaver was not injected via lifespan"
    )
    return _graph


__all__ = ["build_graph", "get_graph", "set_graph"]
