"""LangSmith helpers for intent recognition and other non-LangChain code paths."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

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
