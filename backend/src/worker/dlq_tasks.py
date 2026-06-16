"""Celery tasks for dead-letter queue inspection and archival (PRD §4.7.4)."""

from __future__ import annotations

import logging
from typing import Any

from core.celery_app import celery_app
from core.dead_letter import archive_stale_dead_letters, list_dead_letters
from worker.memory_tasks import _ensure_redis, _run_async

logger = logging.getLogger(__name__)


@celery_app.task(name="worker.dlq_tasks.inspect_dead_letters")  # type: ignore[untyped-decorator]
def inspect_dead_letters() -> dict[str, int]:
    """Daily dead-letter inspection — logs summary (alert hook placeholder)."""
    _run_async(_ensure_redis())
    items = _run_async(list_dead_letters(limit=500))
    logger.warning(
        "Planning dead-letter daily report: %d entries pending manual review",
        len(items),
    )
    for item in items[:20]:
        logger.warning(
            "DLQ task_id=%s job_id=%s exception=%s failed_at=%s",
            item.get("task_id"),
            item.get("job_id"),
            item.get("exception"),
            item.get("failed_at"),
        )
    return {"pending": len(items)}


@celery_app.task(name="worker.dlq_tasks.archive_stale_dead_letters")  # type: ignore[untyped-decorator]
def archive_stale_dead_letters_task() -> dict[str, int]:
    """Archive dead letters older than configured retention."""
    _run_async(_ensure_redis())
    archived = _run_async(archive_stale_dead_letters())
    logger.info("Archived %d stale dead-letter entries to PostgreSQL", archived)
    return {"archived": archived}


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)  # type: ignore[untyped-decorator]
def retry_dead_letter(self: Any, task_id: str, job_id: str) -> str:
    """Re-dispatch a planning job from dead-letter manual retry."""
    from core.dead_letter import remove_dead_letter_by_task_id
    from worker.planning_tasks import execute_planning_job

    removed = _run_async(remove_dead_letter_by_task_id(task_id))
    if not removed:
        logger.warning("Dead letter task_id=%s not found for retry", task_id)
    execute_planning_job.apply_async(args=[job_id], queue="planning")
    return job_id
