"""Background worker that pulls planning jobs from DB and executes them."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional


from core.database import async_session_maker
from core.redis_client import redis_client
from models.planning_job import PlanningJob
from repositories.planning_job import PlanningJobRepository

logger = logging.getLogger(__name__)


class PlanningWorker:
    """Lease-based worker that executes planning jobs asynchronously."""

    HEARTBEAT_INTERVAL = 10  # seconds
    LEASE_DURATION = 60      # seconds
    POLL_INTERVAL = 1        # seconds when no jobs

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self._running = False
        self._current_job_id: Optional[str] = None
        self._cancelled_jobs: set[str] = set()
        self._cancel_events: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    async def run(self):
        """Main loop: acquire jobs, execute pipeline, handle errors."""
        self._running = True
        logger.info(f"Worker {self.worker_id} started")

        while self._running:
            job: Optional[PlanningJob] = None
            async with async_session_maker() as db:
                repo = PlanningJobRepository(db)
                job = await repo.acquire_job(
                    self.worker_id,
                    lease_seconds=self.LEASE_DURATION,
                )
                if job is not None:
                    await db.commit()

            if job is None:
                await asyncio.sleep(self.POLL_INTERVAL)
                continue

            self._current_job_id = job.id
            logger.info(f"Worker {self.worker_id} acquired job {job.id}")

            try:
                await self._execute_job(job)
            except Exception as e:
                logger.exception(f"Job {job.id} failed: {e}")
                async with async_session_maker() as db:
                    repo = PlanningJobRepository(db)
                    await repo.release(
                        job.id, self.worker_id, "failed", str(e)
                    )
                    await db.commit()

            self._current_job_id = None

    def stop(self):
        """Signal the worker to stop after current job."""
        self._running = False
        logger.info(f"Worker {self.worker_id} stopping...")

    # ------------------------------------------------------------------ #
    # Job execution
    # ------------------------------------------------------------------ #

    async def _execute_job(self, job: PlanningJob):
        """Execute the planning pipeline for a single job."""
        cancel_event = asyncio.Event()
        self._cancel_events[job.id] = cancel_event

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(job.id))

        # Start cancel watcher
        cancel_task = asyncio.create_task(self._cancel_watcher(job.id))

        try:
            # Import pipeline here to avoid circular imports
            from pipeline.planning_pipeline import PlanningPipeline

            pipeline = PlanningPipeline(worker=self)
            await pipeline.run(job)
        finally:
            self._cancel_events.pop(job.id, None)
            heartbeat_task.cancel()
            cancel_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            try:
                await cancel_task
            except asyncio.CancelledError:
                pass

    async def _heartbeat_loop(self, job_id: str):
        """Renew lease every HEARTBEAT_INTERVAL seconds."""
        while True:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            async with async_session_maker() as db:
                repo = PlanningJobRepository(db)
                ok = await repo.heartbeat(
                    job_id,
                    self.worker_id,
                    lease_seconds=self.LEASE_DURATION,
                )
                if not ok:
                    logger.warning(
                        f"Worker {self.worker_id} lost lease for job {job_id}"
                    )
                    event = self._cancel_events.get(job_id)
                    if event:
                        event.set()
                    break
                await db.commit()

    async def _cancel_watcher(self, job_id: str):
        """Watch Redis for cancel signals."""
        pubsub = redis_client._client.pubsub()
        await pubsub.subscribe(f"job:cancel:{job_id}")
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    self._cancelled_jobs.add(job_id)
                    event = self._cancel_events.get(job_id)
                    if event:
                        event.set()
                    break
        finally:
            await pubsub.unsubscribe(f"job:cancel:{job_id}")

    # ------------------------------------------------------------------ #
    # Cancellation API
    # ------------------------------------------------------------------ #

    async def check_cancelled(self, job_id: str) -> bool:
        """Check if job has been cancelled (DB is source of truth)."""
        event = self._cancel_events.get(job_id)
        if event and event.is_set():
            return True
        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)
            return await repo.is_cancelled(job_id)

    async def mark_cancelled(self, job_id: str) -> bool:
        """Confirm cancellation and release lock."""
        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)
            ok = await repo.confirm_cancel(job_id, self.worker_id)
            if ok:
                await repo.add_event(
                    job_id=job_id,
                    stage="cancelled",
                    event_type="cancelled",
                )
            await db.commit()
            return ok

    # ------------------------------------------------------------------ #
    # Stage recording + Redis notification
    # ------------------------------------------------------------------ #

    async def record_stage(
        self,
        job: PlanningJob,
        stage: str,
        payload: Optional[dict] = None,
    ):
        """Record stage completion to DB and notify via Redis."""
        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)

            # Update job status
            ok = await repo.update_stage(
                job.id,
                stage,
                payload,
                worker_id=self.worker_id,
            )
            if not ok:
                await db.rollback()
                return False

            # Add event log
            await repo.add_event(
                job_id=job.id,
                stage=stage,
                event_type="completed",
                payload=payload,
            )

            await db.commit()

        # Redis fast notification
        elapsed = (datetime.utcnow() - job.created_at).total_seconds()
        await redis_client.set_json(
            f"job:status:{job.id}",
            {
                "type": "stage",
                "stage": stage,
                "elapsed": round(elapsed, 1),
                "job_id": job.id,
            },
            ttl=300,
        )
        await redis_client._client.publish(
            f"job:status:{job.id}",
            json.dumps(
                {
                    "type": "stage",
                    "stage": stage,
                    "elapsed": round(elapsed, 1),
                    "job_id": job.id,
                },
                ensure_ascii=False,
            ),
        )
        return True


# Global worker instance (started in main.py lifespan)
planning_worker: Optional[PlanningWorker] = None


async def start_worker():
    """Start the planning worker in a background task."""
    global planning_worker
    import uuid as uuid_mod

    worker_id = f"worker-{uuid_mod.uuid4().hex[:8]}"
    planning_worker = PlanningWorker(worker_id)
    asyncio.create_task(planning_worker.run())
    logger.info(f"Planning worker {worker_id} started")


async def stop_worker():
    """Signal the worker to stop gracefully."""
    global planning_worker
    if planning_worker:
        planning_worker.stop()
        # Give a short grace period
        await asyncio.sleep(0.5)
        planning_worker = None
