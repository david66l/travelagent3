"""LangGraph StateGraph assembly for TravelAgent Step 6."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from graph.gathering import build_gathering_subgraph
from graph.models import AgentState
from graph.nodes import (
    apply_single_change_node,
    booking_node,
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
    route_after_booking,
    route_after_factcheck,
    route_after_gathering,
    route_after_hallucination,
    route_after_output,
    route_after_plan,
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
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("weather_check", weather_check_node)
    builder.add_node("plan", plan_node)
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
            "planning": "profile_recall",
        },
    )

    builder.add_conditional_edges(
        "profile_recall",
        route_after_profile,
        {"retrieve": "retrieve", "__end__": END},
    )

    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"plan": "weather_check"},  # weather before plan
    )

    builder.add_conditional_edges(
        "weather_check",
        route_after_weather,
        {"plan": "plan"},
    )

    # plan → output (show draft) or tool_call (Phase 2 enrichment)
    builder.add_conditional_edges(
        "plan",
        route_after_plan,
        {"tool_call": "tool_call", "output": "output"},
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

    # output → booking (final) or END (draft awaiting confirmation)
    builder.add_conditional_edges(
        "output",
        route_after_output,
        {"booking": "booking", "__end__": END},
    )

    builder.add_conditional_edges(
        "booking",
        route_after_booking,
        {"profile_recall": "profile_recall"},
    )

    builder.add_edge("apply_single_change", "factcheck")
    builder.add_edge("replan_local", "plan")
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
