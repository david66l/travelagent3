"""LangSmith helpers for intent recognition and other non-LangChain code paths."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import UUID

F = TypeVar("F", bound=Callable[..., Any])

logger = logging.getLogger(__name__)

try:
    from langsmith import traceable as _langsmith_traceable
except ImportError:
    _langsmith_traceable = None  # type: ignore[misc, assignment]


def langsmith_enabled() -> bool:
    return bool(os.environ.get("LANGSMITH_API_KEY", "")) and _langsmith_traceable is not None


def traceable_step(name: str, *, run_type: str = "chain") -> Callable[[F], F]:
    """Apply LangSmith ``@traceable`` when configured; otherwise no-op."""
    if langsmith_enabled() and _langsmith_traceable is not None:
        return _langsmith_traceable(name=name, run_type=run_type)  # type: ignore[return-value]
    def _identity(fn: F) -> F:
        return fn
    return _identity


def _interrupt_outputs(exc: BaseException) -> dict[str, Any]:
    """Build LangSmith-friendly outputs for a LangGraph human-in-the-loop pause."""
    from langgraph.errors import GraphInterrupt

    outputs: dict[str, Any] = {"status": "awaiting_confirm"}
    if isinstance(exc, GraphInterrupt) and exc.args:
        batch = exc.args[0]
        if batch:
            first = batch[0]
            value = getattr(first, "value", None)
            if isinstance(value, dict):
                outputs["interrupt_type"] = value.get("type")
                if value.get("itinerary") is not None:
                    outputs["itinerary_days"] = len(value["itinerary"])
    return outputs


def patch_langsmith_run_for_interrupt(run_id: str | UUID, exc: BaseException) -> None:
    """Clear LangSmith error status for expected LangGraph ``interrupt()`` pauses."""
    if not langsmith_enabled():
        return
    try:
        from langsmith import Client

        Client().update_run(
            str(run_id),
            error=None,
            outputs=_interrupt_outputs(exc),
            end_time=datetime.now(timezone.utc),
            tags=["graph_interrupt", "awaiting_confirm"],
        )
    except Exception:
        logger.debug("Failed to patch LangSmith run %s for interrupt", run_id, exc_info=True)


def patch_langsmith_root_run_for_pause(exc: BaseException) -> None:
    """Patch the active root LangSmith run after ``GraphBubbleUp`` bubbles out."""
    if not langsmith_enabled():
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        run = get_current_run_tree()
        if run is None:
            return
        run.error = None
        run.end(outputs=_interrupt_outputs(exc))
        run.patch()
    except Exception:
        logger.debug("Failed to patch LangSmith root run for interrupt", exc_info=True)


def graph_langsmith_callbacks() -> list[Any]:
    """Callbacks to attach to LangGraph invocations when tracing is enabled."""
    if not langsmith_enabled():
        return []
    return [GraphInterruptLangSmithHandler()]


try:
    from langchain_core.callbacks.base import BaseCallbackHandler as _BaseCallbackHandler
except ImportError:
    _BaseCallbackHandler = object  # type: ignore[misc, assignment]


class GraphInterruptLangSmithHandler(_BaseCallbackHandler):
    """Rewrite node-level LangSmith failures caused by ``interrupt()`` pauses."""

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        from langgraph.errors import GraphBubbleUp

        if isinstance(error, GraphBubbleUp):
            patch_langsmith_run_for_interrupt(run_id, error)
