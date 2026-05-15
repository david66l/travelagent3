"""Planning job repository — DB operations with lease coordination."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from models.planning_job import PlanningJob, PlanningJobEvent


class PlanningJobRepository:
    """CRUD + lease operations for planning jobs."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ------------------------------------------------------------------ #
    # Create / Read
    # ------------------------------------------------------------------ #

    async def create(
        self,
        session_id: str,
        user_id: str,
        user_input: str,
    ) -> PlanningJob:
        job = PlanningJob(
            session_id=session_id,
            user_id=user_id,
            user_input=user_input,
            status="pending",
        )
        self.db.add(job)
        await self.db.flush()
        await self.db.refresh(job)
        return job

    async def get(self, job_id: str) -> Optional[PlanningJob]:
        result = await self.db.execute(
            select(PlanningJob).where(PlanningJob.id == job_id)
        )
        return result.scalar_one_or_none()

    async def get_by_session(
        self, session_id: str, limit: int = 10
    ) -> list[PlanningJob]:
        result = await self.db.execute(
            select(PlanningJob)
            .where(PlanningJob.session_id == session_id)
            .order_by(PlanningJob.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    # ------------------------------------------------------------------ #
    # Lease + Worker
    # ------------------------------------------------------------------ #

    async def acquire_job(
        self,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> Optional[PlanningJob]:
        """Atomically claim the oldest pending / expired-running job."""
        from sqlalchemy import text

        # Use raw SQL for atomic UPDATE ... RETURNING with SKIP LOCKED
        # NOTE: bind params cannot be embedded inside INTERVAL literals.
        stmt = text(
            """
            UPDATE planning_jobs
            SET status = 'running',
                locked_by = :worker_id,
                lock_expires_at = NOW() + make_interval(secs => :lease_seconds),
                heartbeat_at = NOW(),
                attempt_count = attempt_count + 1
            WHERE id = (
                SELECT id FROM planning_jobs
                WHERE (
                    status = 'pending'
                    OR (status = 'running' AND lock_expires_at < NOW())
                )
                AND attempt_count < max_attempts
                ORDER BY created_at
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
            """
        )
        result = await self.db.execute(
            stmt,
            {"worker_id": worker_id, "lease_seconds": lease_seconds},
        )
        row = result.mappings().one_or_none()
        if row is None:
            return None

        # Hydrate ORM instance from raw row
        job = PlanningJob(**dict(row))
        await self.db.flush()
        return job

    async def heartbeat(
        self,
        job_id: str,
        worker_id: str,
        lease_seconds: int = 60,
    ) -> bool:
        """Renew lease. Returns True if still owned, False if lease was taken."""
        result = await self.db.execute(
            update(PlanningJob)
            .where(
                PlanningJob.id == job_id,
                PlanningJob.locked_by == worker_id,
            )
            .values(
                heartbeat_at=func.now(),
                lock_expires_at=func.now() + timedelta(seconds=lease_seconds),
            )
        )
        return result.rowcount > 0

    async def release(
        self,
        job_id: str,
        worker_id: str,
        status: str,
        error: Optional[str] = None,
    ) -> bool:
        """Release lock and set final status."""
        values = {
            "status": status,
            "locked_by": None,
            "lock_expires_at": None,
            "updated_at": func.now(),
        }
        if error:
            values["last_error"] = error
        if status in ("completed", "failed", "cancelled"):
            values["completed_at"] = func.now()

        result = await self.db.execute(
            update(PlanningJob)
            .where(
                PlanningJob.id == job_id,
                PlanningJob.locked_by == worker_id,
            )
            .values(**values)
        )
        return result.rowcount > 0

    # ------------------------------------------------------------------ #
    # Cancellation
    # ------------------------------------------------------------------ #

    async def request_cancel(self, job_id: str) -> bool:
        """User requests cancellation. DB is source of truth."""
        result = await self.db.execute(
            update(PlanningJob)
            .where(PlanningJob.id == job_id)
            .values(status="cancelling", updated_at=func.now())
        )
        return result.rowcount > 0

    async def confirm_cancel(
        self,
        job_id: str,
        worker_id: str,
    ) -> bool:
        """Worker confirms cancellation and releases lock."""
        result = await self.db.execute(
            update(PlanningJob)
            .where(
                PlanningJob.id == job_id,
                PlanningJob.locked_by == worker_id,
            )
            .values(
                status="cancelled",
                locked_by=None,
                lock_expires_at=None,
                updated_at=func.now(),
            )
        )
        return result.rowcount > 0

    async def is_cancelled(self, job_id: str) -> bool:
        result = await self.db.execute(
            select(PlanningJob.status).where(PlanningJob.id == job_id)
        )
        status = result.scalar_one_or_none()
        return status in ("cancelling", "cancelled")

    # ------------------------------------------------------------------ #
    # Stage updates
    # ------------------------------------------------------------------ #

    async def update_stage(
        self,
        job_id: str,
        stage: str,
        payload: Optional[dict] = None,
        worker_id: Optional[str] = None,
    ) -> bool:
        """Update job status and append stage timing."""
        job = await self.get(job_id)
        if job is None:
            return False

        timings = job.stage_timings or {}
        elapsed = (datetime.utcnow() - job.created_at).total_seconds()
        timings[stage] = round(elapsed, 1)

        values = {
            "status": stage,
            "stage_timings": timings,
            "updated_at": func.now(),
        }
        if payload is not None:
            # Store payload in the corresponding field
            if stage == "intent_ready":
                values["intent_result"] = payload
            elif stage == "strategy_ready":
                values["strategy"] = payload.get("strategy", payload)
            elif stage == "draft_ready":
                values["itinerary_draft"] = payload.get("itinerary_draft", payload)
            elif stage == "itinerary_final":
                values["itinerary_final"] = payload.get("itinerary_final", payload)
            elif stage == "completed":
                values["proposal_text"] = payload.get("proposal_text")

        stmt = update(PlanningJob).where(PlanningJob.id == job_id)
        if worker_id is not None:
            stmt = stmt.where(PlanningJob.locked_by == worker_id)

        result = await self.db.execute(stmt.values(**values))
        return result.rowcount > 0

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    async def add_event(
        self,
        job_id: str,
        stage: str,
        event_type: str,
        payload: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> PlanningJobEvent:
        event = PlanningJobEvent(
            job_id=job_id,
            stage=stage,
            event_type=event_type,
            payload=payload,
            error=error,
        )
        self.db.add(event)
        await self.db.flush()
        await self.db.refresh(event)
        return event

    async def get_events_after(
        self,
        job_id: str,
        after_id: int = 0,
    ) -> list[PlanningJobEvent]:
        result = await self.db.execute(
            select(PlanningJobEvent)
            .where(
                PlanningJobEvent.job_id == job_id,
                PlanningJobEvent.id > after_id,
            )
            .order_by(PlanningJobEvent.id)
        )
        return list(result.scalars().all())
