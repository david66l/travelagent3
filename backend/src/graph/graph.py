"""LangGraph StateGraph assembly for TravelAgent Step 6."""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import END, StateGraph

from graph.gathering import build_gathering_subgraph
from graph.models import AgentState
from graph.nodes import agent_loop_node, booking_node, confirm_gate_node, output_node, profile_node
from graph.routers import (
    route_after_agent_loop,
    route_after_booking,
    route_after_confirm_gate,
    route_after_gathering,
    route_after_output,
    route_after_profile,
)

logger = logging.getLogger(__name__)


def build_graph(checkpointer: Optional[Any] = None) -> StateGraph:
    """Transport/checkpoint shell around one agent loop; no alternative planner."""
    builder = StateGraph(AgentState)
    builder.add_node("gathering", build_gathering_subgraph())
    for name, handler in {
        "profile_recall": profile_node,
        "agent_loop": agent_loop_node,
        "output": output_node,
        "confirm_gate": confirm_gate_node,
        "booking": booking_node,
    }.items():
        builder.add_node(name, handler)
    builder.set_entry_point("gathering")
    builder.add_conditional_edges(
        "gathering",
        route_after_gathering,
        {
            "clarify": "output",
            "respond": "output",
            "infeasible": "output",
            "profile_recall": "profile_recall",
        },
    )
    builder.add_conditional_edges(
        "profile_recall", route_after_profile, {"agent_loop": "agent_loop", "__end__": END}
    )
    builder.add_conditional_edges(
        "agent_loop", route_after_agent_loop, {"agent_loop": "agent_loop", "output": "output"}
    )
    builder.add_conditional_edges(
        "output",
        route_after_output,
        {"booking": "booking", "confirm_gate": "confirm_gate", "__end__": END},
    )
    builder.add_conditional_edges(
        "confirm_gate", route_after_confirm_gate, {"agent_loop": "agent_loop", "output": "output"}
    )
    builder.add_conditional_edges(
        "booking", route_after_booking, {"profile_recall": "profile_recall"}
    )
    return builder.compile(checkpointer=checkpointer) if checkpointer else builder.compile()


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
