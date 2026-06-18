"""Three-tier memory manager for conversation state.

Hot  -> Redis  -> session:{id}:state   (active sessions, short TTL)
Warm -> Redis  -> session:{id}:warm    (recent snapshots, multi-day TTL)
Cold -> Postgres -> conversations.state_snapshot (authoritative long-term archive)

State updates are protected by a Redlock quorum lock (multi-master when configured).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Optional, cast
from uuid import UUID

from sqlalchemy import select

from core.conversation_state import default_conversation_state
from core.database import async_session_maker
from core.redlock import redlock
from core.redis_client import redis_client
from models import Conversation

logger = logging.getLogger(__name__)

HOT_TTL_SECONDS = 1800  # 30 minutes
WARM_TTL_SECONDS = 24 * 3600  # 24 hours (PRD warm layer)
CONV_ID_TTL_SECONDS = 90 * 24 * 3600  # 90 days pointer for cold recovery
LOCK_TTL_SECONDS = 10
LOCK_PREFIX = "session"
HOT_SUFFIX = "state"
WARM_SUFFIX = "warm"
LOCK_SUFFIX = "lock"
CONV_ID_SUFFIX = "conv_id"


def _hot_key(session_id: str) -> str:
    return f"{LOCK_PREFIX}:{session_id}:{HOT_SUFFIX}"


def _warm_key(session_id: str) -> str:
    return f"{LOCK_PREFIX}:{session_id}:{WARM_SUFFIX}"


def _lock_key(session_id: str) -> str:
    return f"{LOCK_PREFIX}:{session_id}:{LOCK_SUFFIX}"


def _conv_id_key(session_id: str) -> str:
    return f"{LOCK_PREFIX}:{session_id}:{CONV_ID_SUFFIX}"


def _compact_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Return a storable snapshot without redundant runtime fields."""
    snapshot = deepcopy(state)
    # Keep the trimmed recent_messages already stored in state.
    snapshot.setdefault("updated_at", 0)
    return snapshot


class MemoryManager:
    """Manage hot/warm/cold state tiers and distributed locks."""

    # --------------------------------------------------------------------- #
    # Hot layer
    # --------------------------------------------------------------------- #

    async def hot_get(self, session_id: str) -> Optional[dict[str, Any]]:
        """Fetch state from the hot Redis layer."""
        return cast(Optional[dict[str, Any]], await redis_client.get_json(_hot_key(session_id)))

    async def hot_set(
        self, session_id: str, state: dict[str, Any], *, ttl: int = HOT_TTL_SECONDS
    ) -> None:
        """Write state to the hot Redis layer and refresh the warm snapshot."""
        await redis_client.set_json(_hot_key(session_id), state, ttl=ttl)
        await self.warm_set(session_id, state)

    async def hot_delete(self, session_id: str) -> None:
        """Remove the hot layer entry (e.g., after explicit logout)."""
        await redis_client.delete(_hot_key(session_id))

    # --------------------------------------------------------------------- #
    # Warm layer
    # --------------------------------------------------------------------- #

    async def warm_get(self, session_id: str) -> Optional[dict[str, Any]]:
        """Fetch the latest warm snapshot from Redis."""
        return cast(Optional[dict[str, Any]], await redis_client.get_json(_warm_key(session_id)))

    async def warm_set(
        self, session_id: str, state: dict[str, Any], *, ttl: int = WARM_TTL_SECONDS
    ) -> None:
        """Store a compacted snapshot in the warm Redis layer."""
        snapshot = _compact_snapshot(state)
        await redis_client.set_json(_warm_key(session_id), snapshot, ttl=ttl)

    async def warm_delete(self, session_id: str) -> None:
        """Remove the warm layer entry after successful cold archive."""
        await redis_client.delete(_warm_key(session_id))

    # --------------------------------------------------------------------- #
    # Cold layer
    # --------------------------------------------------------------------- #

    async def archive_to_cold(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
    ) -> bool:
        """Persist the warm snapshot to PostgreSQL.

        Guests do not get cold storage (PRD §4.5). Authenticated users get a
        Conversation row; anonymous sessions remain in planning_jobs only.
        """
        if user_role == "guest":
            logger.debug("Skipping cold archive for guest session %s", session_id)
            await self.warm_delete(session_id)
            return False

        snapshot = await self.warm_get(session_id)
        if not snapshot:
            logger.debug("No warm snapshot to archive for session %s", session_id)
            return False

        user_uuid = self._parse_uuid(user_id)
        if user_uuid is None:
            logger.debug("Skipping cold archive for anonymous session %s", session_id)
            return False

        conversation_id = self._parse_uuid(snapshot.get("conversation_id"))

        async with async_session_maker() as db:
            conv: Optional[Conversation] = None
            if conversation_id is not None:
                result = await db.execute(
                    select(Conversation).where(Conversation.id == conversation_id)
                )
                conv = result.scalar_one_or_none()

            if conv is None:
                conv = Conversation(
                    user_id=user_uuid,
                    state_snapshot=snapshot,
                    title=snapshot.get("last_intent") or "旅行规划",
                )
                db.add(conv)
                await db.flush()
                await db.refresh(conv)
                # Remember the conversation id in the warm snapshot so future
                # archives update the same row.
                snapshot["conversation_id"] = str(conv.id)
                await self.warm_set(session_id, snapshot)
            else:
                conv.state_snapshot = snapshot

            await db.commit()

        await redis_client.set(
            _conv_id_key(session_id),
            str(conv.id),
            ttl=CONV_ID_TTL_SECONDS,
        )

        logger.info("Archived session %s to conversation %s", session_id, conv.id)
        return True

    async def cold_get_by_conversation_id(self, conversation_id: UUID) -> Optional[dict[str, Any]]:
        """Load archived snapshot from PostgreSQL."""
        async with async_session_maker() as db:
            result = await db.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
            conv = result.scalar_one_or_none()
            if conv and conv.state_snapshot:
                return cast(dict[str, Any], conv.state_snapshot)
        return None

    @staticmethod
    def compress_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        """Compress aged cold snapshots to summary-only form (PRD 90-day policy)."""
        if snapshot.get("_compressed"):
            return snapshot
        recent = snapshot.get("recent_messages") or []
        summary = {
            "last_intent": snapshot.get("last_intent"),
            "destination": snapshot.get("destination"),
            "travel_days": snapshot.get("travel_days"),
            "user_id": snapshot.get("user_id"),
            "conversation_id": snapshot.get("conversation_id"),
            "recent_messages": recent[-5:],
            "message_count": len(recent),
            "_compressed": True,
            "compressed_at": int(datetime.utcnow().timestamp()),
        }
        return summary

    # --------------------------------------------------------------------- #
    # Combined recovery
    # --------------------------------------------------------------------- #

    async def load_state(
        self,
        session_id: str,
        *,
        promote_to_hot: bool = False,
    ) -> dict[str, Any]:
        """Recover state: hot -> warm -> cold -> default empty state."""
        state = await self.hot_get(session_id)
        if state is not None:
            return state

        state = await self.warm_get(session_id)
        if state is not None:
            if promote_to_hot:
                await self.hot_set(session_id, state)
            return state

        conv_id_raw = await redis_client.get(_conv_id_key(session_id))
        conv_id = self._parse_uuid(conv_id_raw)
        if conv_id is not None:
            cold = await self.cold_get_by_conversation_id(conv_id)
            if cold is not None:
                if promote_to_hot:
                    await self.hot_set(session_id, cold)
                return cold

        # session_id is typically the conversation UUID — recover without Redis pointer.
        session_conv_id = self._parse_uuid(session_id)
        if session_conv_id is not None and session_conv_id != conv_id:
            cold = await self.cold_get_by_conversation_id(session_conv_id)
            if cold is not None:
                if promote_to_hot:
                    await self.hot_set(session_id, cold)
                return cold

        return cast(dict[str, Any], default_conversation_state())

    # --------------------------------------------------------------------- #
    # Distributed lock (Redlock-style single-instance)
    # --------------------------------------------------------------------- #

    @asynccontextmanager
    async def acquire_lock(
        self,
        session_id: str,
        *,
        ttl: int = LOCK_TTL_SECONDS,
        blocking: bool = True,
        blocking_timeout: float = 2.0,
    ) -> AsyncIterator[str]:
        """Acquire a Redis-backed lock for the session.

        Yields only after successfully acquiring the lock.  The lock is
        released automatically on context exit unless the caller has already
        lost ownership (token mismatch).
        """
        key = _lock_key(session_id)
        token: Optional[str] = None
        lock_task: Optional[asyncio.Task[None]] = None

        async def _extend() -> None:
            try:
                while True:
                    await asyncio.sleep(ttl / 2)
                    if token and await redlock.extend(key, token, ttl):
                        continue
                    break
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("Lock extend task died for session %s", session_id)

        deadline = asyncio.get_event_loop().time() + blocking_timeout
        while True:
            token = await redlock.acquire(key, ttl, blocking=False, blocking_timeout=0)
            if token:
                break
            if not blocking:
                raise RuntimeError(f"Could not acquire lock for session {session_id}")
            if asyncio.get_event_loop().time() >= deadline:
                raise RuntimeError(f"Timeout acquiring lock for session {session_id}")
            await asyncio.sleep(0.05)

        try:
            lock_task = asyncio.create_task(_extend())
            yield token
        finally:
            if lock_task is not None:
                lock_task.cancel()
                try:
                    await lock_task
                except asyncio.CancelledError:
                    pass
            if token:
                try:
                    await redlock.release(key, token)
                except Exception:
                    logger.debug("Failed to release lock for session %s", session_id)

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #

    @staticmethod
    def _parse_uuid(value: Optional[str]) -> Optional[UUID]:
        if not value:
            return None
        try:
            return UUID(value)
        except (ValueError, TypeError):
            return None


# Global singleton
memory_manager = MemoryManager()
