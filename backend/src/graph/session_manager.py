"""LangGraph session lifecycle: create, resume, timeout."""

from __future__ import annotations

import logging
from typing import Any

from core.memory import memory_manager

logger = logging.getLogger(__name__)

DEFAULT_TTL_SECONDS = 1800


class SessionManager:
    """Manage LangGraph execution sessions with hot/warm/cold memory recovery."""

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds

    async def create(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
        messages: list[dict[str, Any]] | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a fresh AgentState for a new turn."""
        state: dict[str, Any] = {
            "user_id": user_id,
            "user_input": user_input,
            "messages": messages or [],
            "attachments": attachments or [],
            "loop_count": 0,
            "max_loops": 3,
            "version": 1,
            "execution_trace": [],
            "fallback_used": [],
            "stage": "created",
        }
        await memory_manager.hot_set(session_id, state, ttl=self.ttl_seconds)
        return state

    async def load(self, session_id: str) -> dict[str, Any] | None:
        """Load the latest checkpoint from hot memory."""
        try:
            state = await memory_manager.hot_get(session_id)
            if state is None:
                state = await memory_manager.warm_get(session_id)
            if state:
                state.setdefault("loop_count", 0)
                state.setdefault("max_loops", 3)
                state.setdefault("version", 1)
            return state
        except Exception as exc:
            logger.warning("Failed to load session %s: %s", session_id, exc)
            return None

    async def save(self, session_id: str, state: dict[str, Any]) -> None:
        """Persist updated state to hot memory."""
        state["execution_trace"] = (state.get("execution_trace") or []) + [state.get("stage", "unknown")]
        await memory_manager.hot_set(session_id, state, ttl=self.ttl_seconds)

    async def is_expired(self, session_id: str) -> bool:
        """Check whether the session has exceeded TTL without activity."""
        try:
            state = await memory_manager.hot_get(session_id)
            return state is None
        except Exception:
            return True

    async def renew(self, session_id: str) -> None:
        """Renew session TTL."""
        state = await self.load(session_id)
        if state:
            await self.save(session_id, state)

    async def close(self, session_id: str, user_id: str | None = None) -> None:
        """Close session: archive to cold storage and remove hot state."""
        try:
            await memory_manager.archive_to_cold(session_id, user_id=user_id)
        except Exception as exc:
            logger.warning("Failed to archive session %s: %s", session_id, exc)
        try:
            await memory_manager.hot_delete(session_id)
        except Exception as exc:
            logger.warning("Failed to delete hot session %s: %s", session_id, exc)
