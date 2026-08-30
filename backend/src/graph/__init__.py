"""TravelAgent LangGraph orchestration layer.

Public objects are loaded lazily. Importing a pure helper such as
``graph.node_impl`` must not initialize the database-backed graph, checkpoint
store and every optional orchestration dependency.
"""

from __future__ import annotations

from typing import Any

__all__ = ["AgentState", "SessionManager", "build_graph", "get_graph", "set_graph"]


def __getattr__(name: str) -> Any:
    if name in {"build_graph", "get_graph", "set_graph"}:
        from graph import graph as graph_module

        return getattr(graph_module, name)
    if name == "AgentState":
        from graph.models import AgentState

        return AgentState
    if name == "SessionManager":
        from graph.session_manager import SessionManager

        return SessionManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
