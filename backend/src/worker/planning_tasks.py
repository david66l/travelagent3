"""Celery tasks for async planning job execution (M2)."""

from __future__ import annotations

import logging
import traceback
import uuid
from typing import Any, cast

from sqlalchemy import update

from core.celery_app import celery_app
from core.database import async_session_maker
from core.dead_letter import push_dead_letter
from core.metrics import incr
from core.settings import settings
from core.task_retry import (
    compute_retry_delay,
    increment_retry,
    is_retryable,
    reset_retry,
)
from models.planning_job import PlanningJob
from repositories.planning_job import PlanningJobRepository
from worker.memory_tasks import _ensure_redis, _run_async
from worker.planning_worker import PlanningWorker

logger = logging.getLogger(__name__)

TASK_NAME = "worker.planning_tasks.execute_planning_job"


async def _execute_planning_job_async(job_id: str) -> bool:
    await _ensure_redis()
    worker_id = f"celery-{uuid.uuid4().hex[:8]}"
    worker = PlanningWorker(worker_id)
    return await worker.execute_job_by_id(job_id)


async def _mark_retrying(job_id: str, error: str) -> None:
    async with async_session_maker() as db:
        repo = PlanningJobRepository(db)
        await repo.mark_retrying(job_id, error)
        await db.commit()


async def _finalize_dead_letter(
    job_id: str,
    exc: BaseException,
    task_id: str,
) -> None:
    tb = traceback.format_exc()
    await push_dead_letter(
        task_name=TASK_NAME,
        job_id=job_id,
        exception=exc,
        traceback_text=tb,
        kwargs={"job_id": job_id},
        task_id=task_id,
    )
    incr("planning_jobs_failed_total")
    async with async_session_maker() as db:
        repo = PlanningJobRepository(db)
        await repo.update_status(job_id, "failed")
        await db.execute(
            update(PlanningJob)
            .where(PlanningJob.id == job_id)
            .values(last_error=f"{type(exc).__name__}: {exc}")
        )
        await db.commit()


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True,
    name=TASK_NAME,
    max_retries=10,
    acks_late=True,
)
def execute_planning_job(self: Any, job_id: str) -> bool:
    """Claim and run one planning job by id (PRD: generate_itinerary_task)."""
    try:
        ok = cast(bool, _run_async(_execute_planning_job_async(job_id)))
        if ok:
            _run_async(reset_retry(TASK_NAME, job_id))
            return True

        logger.warning(
            "Job %s was not claimed (missing, max attempts, or already running)",
            job_id,
        )
        return False
    except Exception as exc:
        logger.exception("execute_planning_job failed for %s", job_id)

        if not is_retryable(exc):
            _run_async(_finalize_dead_letter(job_id, exc, str(self.request.id)))
            return False

        attempt = _run_async(increment_retry(TASK_NAME, job_id))
        if attempt > settings.planning_task_max_retries:
            _run_async(_finalize_dead_letter(job_id, exc, str(self.request.id)))
            return False

        _run_async(_mark_retrying(job_id, f"{type(exc).__name__}: {exc}"))
        delay = compute_retry_delay(attempt)
        raise self.retry(exc=exc, countdown=delay)


@celery_app.task(name="worker.planning_tasks.enforce_cancel_deadline")  # type: ignore[untyped-decorator]
def enforce_cancel_deadline(job_id: str) -> bool:
    """Force-cancel a job stuck in cancelling (PRD §4.7)."""

    async def _enforce() -> bool:
        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)
            ok = await repo.force_cancel(job_id)
            if ok:
                await repo.add_event(
                    job_id=job_id,
                    stage="force_cancelled",
                    event_type="force_cancelled",
                )
            await db.commit()
            return ok

    try:
        forced = cast(bool, _run_async(_enforce()))
        if forced:
            logger.warning("Force-cancelled job %s after cancel deadline", job_id)
            incr("planning_jobs_force_cancelled_total")
        return forced
    except Exception:
        logger.exception("enforce_cancel_deadline failed for %s", job_id)
        return False
