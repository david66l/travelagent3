"""Planning dead-letter queue in Redis (PRD §4.7.4 — queue planning_dead_letter)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from core.redis_client import redis_client
from core.settings import settings

DLQ_LIST_KEY = "dlq:planning_dead_letter"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def push_dead_letter(
    *,
    task_name: str,
    job_id: str | None,
    exception: BaseException,
    traceback_text: str,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
    task_id: str | None = None,
) -> str:
    """Append a dead-letter record to the Redis list."""
    message_id = task_id or str(uuid.uuid4())
    payload = {
        "task_id": message_id,
        "task_name": task_name,
        "job_id": job_id,
        "failed_at": _now_iso(),
        "exception": type(exception).__name__,
        "traceback": traceback_text,
        "args": args or [],
        "kwargs": kwargs or {},
    }
    await redis_client.lpush(DLQ_LIST_KEY, json.dumps(payload, ensure_ascii=False))
    return message_id


async def list_dead_letters(limit: int = 100) -> list[dict[str, Any]]:
    raw_items = await redis_client.lrange(DLQ_LIST_KEY, 0, limit - 1)
    items: list[dict[str, Any]] = []
    for raw in raw_items:
        try:
            items.append(json.loads(raw))
        except json.JSONDecodeError:
            items.append({"raw": raw, "parse_error": True})
    return items


async def remove_dead_letter_by_task_id(task_id: str) -> bool:
    """Remove one DLQ entry matching task_id (manual retry / dismiss)."""
    raw_items = await redis_client.lrange(DLQ_LIST_KEY, 0, -1)
    for raw in raw_items:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if data.get("task_id") == task_id:
            await redis_client.lrem(DLQ_LIST_KEY, 1, raw)
            return True
    return False


async def archive_stale_dead_letters(days: int | None = None) -> int:
    """Move entries older than ``days`` to PostgreSQL and remove from Redis."""
    from repositories.dead_letter_archive import DeadLetterArchiveRepository

    threshold_days = days if days is not None else settings.dead_letter_archive_days
    cutoff = datetime.now(timezone.utc).timestamp() - threshold_days * 86400
    archived = 0

    raw_items = await redis_client.lrange(DLQ_LIST_KEY, 0, -1)
    for raw in raw_items:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        failed_at = data.get("failed_at")
        if not failed_at:
            continue
        try:
            failed_ts = datetime.fromisoformat(failed_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        if failed_ts > cutoff:
            continue

        from core.database import async_session_maker

        async with async_session_maker() as db:
            repo = DeadLetterArchiveRepository(db)
            await repo.create_from_dlq(data)
            await db.commit()
        await redis_client.lrem(DLQ_LIST_KEY, 1, raw)
        archived += 1

    return archived
