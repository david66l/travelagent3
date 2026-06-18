"""Celery background tasks for the memory tier lifecycle."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Coroutine, Optional, TypeVar, cast

from celery import signals
from sqlalchemy import select

from core.celery_app import celery_app
from core.database import async_session_maker
from core.memory import WARM_SUFFIX, memory_manager
from core.redis_client import redis_client
from models import Conversation

logger = logging.getLogger(__name__)

DISCONNECT_ARCHIVE_DELAY_SECONDS = 30 * 60  # PRD: archive 30min after disconnect
ACTIVE_ARCHIVE_INTERVAL_SECONDS = 300  # PRD: active session incremental archive every 5min
COMPRESS_AGE_DAYS = 90
ARCHIVE_AGE_SECONDS = 24 * 3600  # archive warm snapshots older than 24h

_worker_loop: Optional[asyncio.AbstractEventLoop] = None
_worker_loop_thread: Optional[threading.Thread] = None
_loop_ready = threading.Event()


def _start_worker_loop() -> None:
    """Background thread entry point that owns the worker event loop."""
    global _worker_loop
    loop = asyncio.new_event_loop()
    _worker_loop = loop
    _loop_ready.set()
    loop.run_forever()


@signals.worker_process_init.connect  # type: ignore[untyped-decorator]
def init_worker_process(**kwargs: Any) -> None:
    """Start a dedicated asyncio event loop thread for this Celery worker process.

    Keeping a single event loop alive for the lifetime of the process lets
    async resources (Redis connection pool, SQLAlchemy async engine) bind to
    one loop and avoids ``Event loop is closed`` errors when tasks run.
    """
    global _worker_loop_thread
    _worker_loop_thread = threading.Thread(
        target=_start_worker_loop, name="celery-async-loop", daemon=True
    )
    _worker_loop_thread.start()
    if not _loop_ready.wait(timeout=5):
        logger.warning("Celery worker async loop did not start in time")


@signals.worker_process_shutdown.connect  # type: ignore[untyped-decorator]
def shutdown_worker_process(**kwargs: Any) -> None:
    """Stop the background event loop thread when the worker shuts down."""
    global _worker_loop, _worker_loop_thread
    loop = _worker_loop
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if _worker_loop_thread is not None:
        _worker_loop_thread.join(timeout=5)
        _worker_loop_thread = None
    _worker_loop = None
    _loop_ready.clear()


async def _ensure_redis() -> None:
    """Ensure Redis is connected in the current event loop."""
    try:
        result = redis_client._ensure_client()
        # In unit tests redis_client is an AsyncMock and returns a coroutine.
        if asyncio.iscoroutine(result):
            await result
    except RuntimeError:
        await redis_client.connect()


_T = TypeVar("_T")


def _run_async(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run an async coroutine from a synchronous Celery task body.

    In a real Celery worker a background event loop thread is used so all
    async I/O runs on the same loop. Outside the worker (e.g. unit tests) we
    fall back to ``asyncio.run``.
    """
    loop = _worker_loop
    if loop is not None and loop.is_running():
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result()
    return asyncio.run(coro)


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)  # type: ignore[untyped-decorator]
def archive_session(
    self: Any,
    session_id: str,
    user_id: str | None = None,
    user_role: str | None = None,
) -> bool:
    """Archive a single session's warm snapshot to cold storage."""
    try:
        _run_async(_ensure_redis())
        return cast(
            bool,
            _run_async(memory_manager.archive_to_cold(session_id, user_id, user_role)),
        )
    except Exception as exc:
        logger.exception("archive_session failed for %s", session_id)
        raise self.retry(exc=exc)


@celery_app.task  # type: ignore[untyped-decorator]
def schedule_delayed_archive(
    session_id: str,
    user_id: str | None = None,
    delay_seconds: int = DISCONNECT_ARCHIVE_DELAY_SECONDS,
) -> str:
    """Enqueue final archive after disconnect grace period."""
    archive_session.apply_async(args=[session_id, user_id], countdown=delay_seconds)
    return session_id


@celery_app.task(bind=True, max_retries=2, default_retry_delay=120)  # type: ignore[untyped-decorator]
def archive_active_sessions(self: Any) -> dict[str, int]:
    """Incrementally archive active sessions with authenticated users."""
    _run_async(_ensure_redis())

    archived = 0
    skipped = 0

    async def _archive_active() -> dict[str, int]:
        nonlocal archived, skipped
        cursor = 0
        while True:
            cursor, keys = await redis_client.scan(
                cursor=cursor, match="session:*:state", count=100
            )
            for key in keys:
                parts = key.split(":")
                if len(parts) < 3 or parts[0] != "session":
                    continue
                session_id = parts[1]
                snapshot = await memory_manager.hot_get(session_id)
                if not snapshot:
                    skipped += 1
                    continue
                user_id = snapshot.get("user_id")
                user_role = snapshot.get("user_role")
                if not user_id or user_role == "guest":
                    skipped += 1
                    continue
                ok = await memory_manager.archive_to_cold(session_id, user_id, user_role)
                if ok:
                    archived += 1
                else:
                    skipped += 1
            if cursor == 0:
                break
        return {"archived": archived, "skipped": skipped}

    try:
        return _run_async(_archive_active())
    except Exception as exc:
        logger.exception("archive_active_sessions failed")
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=600)  # type: ignore[untyped-decorator]
def cleanup_expired_memories(self: Any) -> dict[str, int]:
    """Compress cold snapshots older than 90 days."""
    compressed = 0

    async def _cleanup() -> dict[str, int]:
        nonlocal compressed
        cutoff = datetime.utcnow() - timedelta(days=COMPRESS_AGE_DAYS)
        async with async_session_maker() as db:
            result = await db.execute(
                select(Conversation).where(Conversation.updated_at < cutoff)
            )
            for conv in result.scalars():
                snapshot = conv.state_snapshot or {}
                if snapshot.get("_compressed"):
                    continue
                conv.state_snapshot = memory_manager.compress_snapshot(snapshot)
                conv.archived_at = conv.archived_at or datetime.utcnow()
                compressed += 1
            await db.commit()
        return {"compressed": compressed}

    try:
        return _run_async(_cleanup())
    except Exception as exc:
        logger.exception("cleanup_expired_memories failed")
        raise self.retry(exc=exc)


@celery_app.task(bind=True, max_retries=2, default_retry_delay=300)  # type: ignore[untyped-decorator]
def compact_stale_sessions(self: Any) -> dict[str, int]:
    """Scan warm-layer snapshots and archive stale ones to cold storage."""
    _run_async(_ensure_redis())

    archived = 0
    skipped = 0

    async def _compact() -> dict[str, int]:
        nonlocal archived, skipped
        cursor = 0
        now = time.time()
        while True:
            cursor, keys = await redis_client.scan(
                cursor=cursor, match=f"*{WARM_SUFFIX}", count=100
            )
            for key in keys:
                if not key.endswith(f":{WARM_SUFFIX}"):
                    continue
                body = key[:-len(f":{WARM_SUFFIX}")]
                if not body.startswith("session:"):
                    continue
                session_id = body[len("session:") :]
                snapshot = await memory_manager.warm_get(session_id)
                if not snapshot:
                    continue
                updated_at = snapshot.get("updated_at", 0)
                if now - updated_at < ARCHIVE_AGE_SECONDS:
                    skipped += 1
                    continue
                user_id = snapshot.get("user_id")
                user_role = snapshot.get("user_role")
                if not user_id or user_role == "guest":
                    skipped += 1
                    continue
                ok = await memory_manager.archive_to_cold(session_id, user_id, user_role)
                if ok:
                    await memory_manager.warm_delete(session_id)
                    archived += 1
                    logger.info("Compacted session %s to cold storage", session_id)
                else:
                    skipped += 1
            if cursor == 0:
                break
        return {"archived": archived, "skipped": skipped}

    try:
        return _run_async(_compact())
    except Exception as exc:
        logger.exception("compact_stale_sessions failed")
        raise self.retry(exc=exc)
