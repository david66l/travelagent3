"""Celery tasks for async planning job execution (M2)."""

from __future__ import annotations

import logging
import traceback
import uuid
from datetime import timedelta
from typing import Any, cast

from sqlalchemy import select, update

from core.celery_app import celery_app
from core.clock import utc_now_naive
from core.database import async_session_maker
from core.dead_letter import push_dead_letter
from core.metrics import incr, record_planning_task_retry
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
REDISPATCH_TASK_NAME = "worker.planning_tasks.redispatch_pending_planning_jobs"
_worker_checkpointer_context: Any | None = None
_worker_graph_ready = False


async def _ensure_worker_graph() -> None:
    """Install a process-lifetime Postgres checkpointer in Celery workers."""
    global _worker_checkpointer_context, _worker_graph_ready
    if _worker_graph_ready:
        return
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    from graph.graph import build_graph, set_graph

    context = AsyncPostgresSaver.from_conn_string(settings.database_url_sync)
    checkpointer = await context.__aenter__()
    try:
        await checkpointer.setup()
        set_graph(build_graph(checkpointer=checkpointer))
    except Exception:
        await context.__aexit__(None, None, None)
        raise
    _worker_checkpointer_context = context
    _worker_graph_ready = True
    logger.info("Celery planning worker graph compiled with AsyncPostgresSaver")


async def close_worker_graph() -> None:
    """Close the process-lifetime checkpointer connection pool on shutdown."""
    global _worker_checkpointer_context, _worker_graph_ready
    context = _worker_checkpointer_context
    _worker_checkpointer_context = None
    _worker_graph_ready = False
    if context is not None:
        await context.__aexit__(None, None, None)


async def _execute_planning_job_async(job_id: str) -> bool:
    await _ensure_redis()
    await _ensure_worker_graph()
    worker_id = f"celery-{uuid.uuid4().hex[:8]}"
    worker = PlanningWorker(worker_id)
    return await worker.execute_job_by_id(job_id)


async def _mark_retrying(job_id: str, error: str) -> None:
    async with async_session_maker() as db:
        repo = PlanningJobRepository(db)
        await repo.mark_retrying(job_id, error)
        await db.commit()


async def _dispatchable_job_ids(limit: int) -> list[str]:
    """Return durable jobs that may have missed their original broker publish."""
    cutoff = utc_now_naive() - timedelta(seconds=15)
    async with async_session_maker() as db:
        result = await db.execute(
            select(PlanningJob.id)
            .where(
                PlanningJob.status.in_(("pending", "retrying")),
                PlanningJob.updated_at <= cutoff,
            )
            .order_by(PlanningJob.created_at.asc())
            .limit(limit)
        )
        return [str(job_id) for job_id in result.scalars().all()]


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
            record_planning_task_retry("nonretryable")
            _run_async(_finalize_dead_letter(job_id, exc, str(self.request.id)))
            return False

        attempt = _run_async(increment_retry(TASK_NAME, job_id))
        if attempt > settings.planning_task_max_retries:
            record_planning_task_retry("exhausted")
            _run_async(_finalize_dead_letter(job_id, exc, str(self.request.id)))
            return False

        _run_async(_mark_retrying(job_id, f"{type(exc).__name__}: {exc}"))
        record_planning_task_retry("scheduled")
        delay = compute_retry_delay(attempt)
        raise self.retry(exc=exc, countdown=delay)


@celery_app.task(name=REDISPATCH_TASK_NAME)  # type: ignore[untyped-decorator]
def redispatch_pending_planning_jobs(limit: int = 100) -> int:
    """Compensate the DB-commit/broker-publish window.

    Duplicate deliveries are safe: the worker's atomic lease allows only one
    message to claim a given PlanningJob.
    """
    job_ids = cast(list[str], _run_async(_dispatchable_job_ids(limit)))
    for job_id in job_ids:
        execute_planning_job.apply_async(
            args=[job_id],
            queue=settings.celery_planning_queue,
        )
    if job_ids:
        logger.info("Redispatched %d durable planning jobs", len(job_ids))
    return len(job_ids)


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
