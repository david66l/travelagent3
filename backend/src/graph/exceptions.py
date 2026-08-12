"""LangGraph node exceptions and 3-level degradation strategy."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Awaitable, Callable

from langgraph.errors import GraphBubbleUp

logger = logging.getLogger(__name__)


class DegradationLevel(str, Enum):
    """Three-level degradation per blueprint."""

    RETRY = "retry"  # Level 1: retry the same node with adjusted input
    FALLBACK = "fallback"  # Level 2: use a simpler local implementation
    ESCALATE = "escalate"  # Level 3: return to user with clarification / error


class NodeException(Exception):
    """Exception raised by a LangGraph node with degradation metadata."""

    def __init__(
        self,
        node: str,
        message: str,
        level: DegradationLevel = DegradationLevel.FALLBACK,
        original: Exception | None = None,
        state_patch: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.node = node
        self.message = message
        self.level = level
        self.original = original
        self.state_patch = state_patch or {}


class NodeErrors:
    """Known node error taxonomies."""

    LLM_TIMEOUT = "llm_timeout"
    LLM_REFUSAL = "llm_refusal"
    DB_UNAVAILABLE = "db_unavailable"
    VRP_SERVICE_ERROR = "vrp_service_error"
    RETRIEVAL_EMPTY = "retrieval_empty"
    FACT_CONFLICT = "fact_conflict"
    EXTERNAL_API_ERROR = "external_api_error"


ERROR_DEGRADATION: dict[str, DegradationLevel] = {
    NodeErrors.LLM_TIMEOUT: DegradationLevel.RETRY,
    NodeErrors.LLM_REFUSAL: DegradationLevel.ESCALATE,
    NodeErrors.DB_UNAVAILABLE: DegradationLevel.FALLBACK,
    NodeErrors.VRP_SERVICE_ERROR: DegradationLevel.FALLBACK,
    NodeErrors.RETRIEVAL_EMPTY: DegradationLevel.FALLBACK,
    NodeErrors.FACT_CONFLICT: DegradationLevel.RETRY,
    NodeErrors.EXTERNAL_API_ERROR: DegradationLevel.FALLBACK,
}


def classify_error(node: str, exc: Exception) -> NodeException:
    """Classify a raw exception into a NodeException with degradation level."""
    msg = str(exc).lower()
    if "timeout" in msg or "timed out" in msg:
        error_type = NodeErrors.LLM_TIMEOUT
    elif "refusal" in msg or "content_filter" in msg:
        error_type = NodeErrors.LLM_REFUSAL
    elif "connection" in msg or "unavailable" in msg:
        error_type = (
            NodeErrors.DB_UNAVAILABLE if "postgres" in msg else NodeErrors.EXTERNAL_API_ERROR
        )
    elif "vrp" in msg or "solver" in msg:
        error_type = NodeErrors.VRP_SERVICE_ERROR
    else:
        error_type = NodeErrors.EXTERNAL_API_ERROR

    level = ERROR_DEGRADATION.get(error_type, DegradationLevel.FALLBACK)
    return NodeException(node=node, message=str(exc), level=level, original=exc)


async def global_error_handler(
    state: dict[str, Any],
    exc: Exception,
    node_name: str,
) -> dict[str, Any]:
    """Handle node exceptions and produce a degraded state update."""
    if isinstance(exc, NodeException):
        node_exc = exc
    else:
        node_exc = classify_error(node_name, exc)

    logger.warning(
        "Node %s failed (%s): %s",
        node_exc.node,
        node_exc.level,
        node_exc.message,
        exc_info=node_exc.original,
    )

    patch = {
        "error_node": node_exc.node,
        "error_message": node_exc.message,
        # fallback_used is a reducer field → return only this node's delta.
        "fallback_used": [f"{node_exc.node}:{node_exc.level.value}"],
        **node_exc.state_patch,
    }

    if node_exc.level == DegradationLevel.RETRY:
        patch["retry_count"] = (state.get("retry_count") or 0) + 1
        patch["next_action"] = "retry"
    elif node_exc.level == DegradationLevel.FALLBACK:
        patch["next_action"] = "fallback"
    else:  # escalate
        patch["next_action"] = "clarify"
        patch["clarification_questions"] = (state.get("clarification_questions") or []) + [
            f"[{node_exc.node}] 处理异常，请稍后重试或调整需求。"
        ]

    return patch


def with_error_handling(node_name: str):
    """Decorator that wraps a node coroutine with global error handling."""

    def decorator(func: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]):
        async def wrapper(state: dict[str, Any]) -> dict[str, Any]:
            try:
                return await func(state)
            except GraphBubbleUp:
                # LangGraph control flow (interrupt / Command bubble-up) must
                # propagate untouched — never degrade it as a node error.
                raise
            except Exception as exc:
                return await global_error_handler(state, exc, node_name)

        return wrapper

    return decorator
