"""Integration tests for PlanningJob repository with real PostgreSQL."""

import asyncio
import time
import pytest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import async_session_maker
from models.planning_job import PlanningJob
from models import User
from repositories.planning_job import PlanningJobRepository
from tests.support.planning_feedback import feedback_with_trip


class TestPlanningJobRepository:
    """Real DB tests for PlanningJobRepository."""

    @pytest.mark.asyncio
    async def test_create_job(self, db: AsyncSession):
        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        assert job.id is not None
        assert job.status == "pending"
        assert job.session_id == "sess-1"
        assert job.user_input == "北京3天"

        # Verify in DB
        result = await db.execute(select(PlanningJob).where(PlanningJob.id == job.id))
        fetched = result.scalar_one()
        assert fetched.status == "pending"

    @pytest.mark.asyncio
    async def test_get_job(self, db: AsyncSession):
        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        fetched = await repo.get(job.id)
        assert fetched is not None
        assert fetched.id == job.id

    @pytest.mark.asyncio
    async def test_guest_quota_counts_verified_itineraries_not_clarification_turns(
        self, db: AsyncSession
    ):
        user = User(role="guest")
        db.add(user)
        await db.flush()
        repo = PlanningJobRepository(db)

        clarification = await repo.create(
            session_id="quota-clarify",
            user_id=str(user.id),
            user_uuid=user.id,
            user_input="国庆去上海看演唱会",
        )
        clarification.status = "completed"
        clarification.user_feedback = {
            "agent_status": "awaiting_information",
            "itinerary": [],
        }
        await db.commit()
        assert await repo.count_completed_for_user(user.id) == 0

        itinerary = await repo.create(
            session_id="quota-success",
            user_id=str(user.id),
            user_uuid=user.id,
            user_input="杭州一天",
        )
        itinerary.status = "completed"
        itinerary.user_feedback = {
            "agent_status": "awaiting_confirmation",
            "itinerary": [{"day_number": 1, "activities": []}],
        }
        await db.commit()
        assert await repo.count_completed_for_user(user.id) == 1

    @pytest.mark.asyncio
    async def test_idempotency_advisory_lock_serializes_first_writer_race(self, db: AsyncSession):
        user = User(role="guest")
        db.add(user)
        await db.commit()
        user_id = user.id
        key = "concurrent-integration-key"
        first_locked = asyncio.Event()
        first_job_id: str | None = None

        async def first_writer():
            nonlocal first_job_id
            async with async_session_maker() as session:
                repo = PlanningJobRepository(session)
                await repo.acquire_idempotency_lock(user_id, key)
                first_locked.set()
                job = await repo.create(
                    session_id="idempotency-race",
                    user_id=str(user_id),
                    user_uuid=user_id,
                    user_input="上海一天",
                    idempotency_key=key,
                )
                first_job_id = job.id
                await asyncio.sleep(0.2)
                await session.commit()

        async def second_writer():
            await first_locked.wait()
            started = time.monotonic()
            async with async_session_maker() as session:
                repo = PlanningJobRepository(session)
                await repo.acquire_idempotency_lock(user_id, key)
                existing = await repo.get_by_idempotency_key(user_id, key)
                waited = time.monotonic() - started
                await session.commit()
                return existing, waited

        (_, second_result) = await asyncio.gather(first_writer(), second_writer())
        existing, waited = second_result
        assert existing is not None
        assert existing.id == first_job_id
        assert waited >= 0.15

    @pytest.mark.asyncio
    async def test_acquire_job_pending(self, db: AsyncSession):
        """Worker can acquire a pending job."""
        repo = PlanningJobRepository(db)
        await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        acquired = await repo.acquire_job("worker-1", lease_seconds=60)
        await db.commit()

        assert acquired is not None
        assert acquired.status == "running"
        assert acquired.locked_by == "worker-1"
        assert acquired.lock_expires_at is not None
        assert acquired.attempt_count == 1

    @pytest.mark.asyncio
    async def test_acquire_job_by_id(self, db: AsyncSession):
        """Worker can claim a specific pending job."""
        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-by-id",
            user_id="user-1",
            user_input="上海2天",
        )
        await db.commit()

        acquired = await repo.acquire_job_by_id(job.id, "worker-2", lease_seconds=60)
        await db.commit()

        assert acquired is not None
        assert acquired.id == job.id
        assert acquired.locked_by == "worker-2"

        # Second claim on same job should fail while running
        again = await repo.acquire_job_by_id(job.id, "worker-3", lease_seconds=60)
        assert again is None

    @pytest.mark.asyncio
    async def test_acquire_no_pending_jobs(self, db: AsyncSession):
        """Returns None when no pending jobs."""
        repo = PlanningJobRepository(db)
        acquired = await repo.acquire_job("worker-1")
        await db.commit()
        assert acquired is None

    @pytest.mark.asyncio
    async def test_heartbeat_renews_lease(self, db: AsyncSession):
        """Heartbeat extends lease expiration."""
        repo = PlanningJobRepository(db)
        await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        acquired = await repo.acquire_job("worker-1", lease_seconds=60)
        await db.commit()

        old_expires = acquired.lock_expires_at

        ok = await repo.heartbeat(acquired.id, "worker-1", lease_seconds=120)
        await db.commit()

        assert ok is True

        # Expire cache to get fresh data from DB
        db.expire_all()
        result = await db.execute(select(PlanningJob).where(PlanningJob.id == acquired.id))
        updated = result.scalar_one()
        assert updated.lock_expires_at > old_expires

    @pytest.mark.asyncio
    async def test_heartbeat_fails_when_lease_taken(self, db: AsyncSession):
        """Heartbeat returns False if another worker took the lease."""
        repo = PlanningJobRepository(db)
        await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        acquired = await repo.acquire_job("worker-1", lease_seconds=60)
        await db.commit()

        # Simulate another worker taking over (manual update)
        await db.execute(select(PlanningJob).where(PlanningJob.id == acquired.id).with_for_update())
        await db.execute(
            update(PlanningJob).where(PlanningJob.id == acquired.id).values(locked_by="worker-2")
        )
        await db.commit()

        ok = await repo.heartbeat(acquired.id, "worker-1", lease_seconds=60)
        await db.commit()
        assert ok is False

    @pytest.mark.asyncio
    async def test_release_job(self, db: AsyncSession):
        """Release job marks status and clears lock."""
        repo = PlanningJobRepository(db)
        await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        acquired = await repo.acquire_job("worker-1", lease_seconds=60)
        await db.commit()

        ok = await repo.release(acquired.id, "worker-1", "completed")
        await db.commit()

        assert ok is True

        db.expire_all()
        result = await db.execute(select(PlanningJob).where(PlanningJob.id == acquired.id))
        updated = result.scalar_one()
        assert updated.status == "completed"
        assert updated.locked_by is None
        assert updated.lock_expires_at is None

    @pytest.mark.asyncio
    async def test_release_fails_when_lease_taken(self, db: AsyncSession):
        """Release returns False if lease was taken by another worker."""
        repo = PlanningJobRepository(db)
        await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        acquired = await repo.acquire_job("worker-1", lease_seconds=60)
        await db.commit()

        # Another worker takes over
        await db.execute(
            update(PlanningJob).where(PlanningJob.id == acquired.id).values(locked_by="worker-2")
        )
        await db.commit()

        ok = await repo.release(acquired.id, "worker-1", "completed")
        await db.commit()
        assert ok is False

    @pytest.mark.asyncio
    async def test_update_stage_fails_when_lease_taken(self, db: AsyncSession):
        """A stale worker cannot update status or payload after losing lease."""
        repo = PlanningJobRepository(db)
        await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        acquired = await repo.acquire_job("worker-1", lease_seconds=60)
        await db.commit()

        await db.execute(
            update(PlanningJob).where(PlanningJob.id == acquired.id).values(locked_by="worker-2")
        )
        await db.commit()

        ok = await repo.update_stage(
            acquired.id,
            "completed",
            {"proposal_text": "stale result"},
            worker_id="worker-1",
        )
        await db.commit()

        assert ok is False
        db.expire_all()
        result = await db.execute(select(PlanningJob).where(PlanningJob.id == acquired.id))
        updated = result.scalar_one()
        assert updated.status == "running"
        assert updated.locked_by == "worker-2"
        assert updated.proposal_text is None

    @pytest.mark.asyncio
    async def test_cancel_job(self, db: AsyncSession):
        """Request cancel sets status to cancelling."""
        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        ok = await repo.request_cancel(job.id)
        await db.commit()

        assert ok is True
        fetched = await repo.get(job.id)
        assert fetched.status == "cancelling"

    @pytest.mark.asyncio
    async def test_confirm_cancel(self, db: AsyncSession):
        """Worker confirms cancellation and releases lock."""
        repo = PlanningJobRepository(db)
        await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        acquired = await repo.acquire_job("worker-1", lease_seconds=60)
        await db.commit()

        ok = await repo.confirm_cancel(acquired.id, "worker-1")
        await db.commit()

        assert ok is True
        db.expire_all()
        fetched = await repo.get(acquired.id)
        assert fetched.status == "cancelled"
        assert fetched.locked_by is None

    @pytest.mark.asyncio
    async def test_is_cancelled(self, db: AsyncSession):
        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        assert await repo.is_cancelled(job.id) is False

        await repo.request_cancel(job.id)
        await db.commit()
        assert await repo.is_cancelled(job.id) is True

    @pytest.mark.asyncio
    async def test_acquire_expired_job(self, db: AsyncSession):
        """Expired running job can be acquired by another worker."""
        repo = PlanningJobRepository(db)
        await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        # Worker-1 acquires
        acquired1 = await repo.acquire_job("worker-1", lease_seconds=60)
        await db.commit()

        # Simulate lease expiration (set lock_expires_at in the past)
        await db.execute(
            update(PlanningJob)
            .where(PlanningJob.id == acquired1.id)
            .values(lock_expires_at=datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=1))
        )
        await db.commit()

        # Worker-2 should be able to acquire
        acquired2 = await repo.acquire_job("worker-2", lease_seconds=60)
        await db.commit()

        assert acquired2 is not None
        assert acquired2.id == acquired1.id
        assert acquired2.locked_by == "worker-2"
        assert acquired2.attempt_count == 2

    @pytest.mark.asyncio
    async def test_add_event(self, db: AsyncSession):
        """Events are recorded with payload."""
        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        event = await repo.add_event(
            job_id=job.id,
            stage="running",
            event_type="completed",
            payload={"itinerary": [{"day": 1}]},
        )
        await db.commit()

        assert event.id is not None
        assert event.job_id == job.id
        assert event.stage == "running"
        assert event.payload == {"itinerary": [{"day": 1}]}

    @pytest.mark.asyncio
    async def test_get_events_after(self, db: AsyncSession):
        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-1",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        e1 = await repo.add_event(job.id, "running", "completed")
        e2 = await repo.add_event(job.id, "completed", "completed")
        await db.commit()

        # Use a fresh session to avoid identity map caching
        async with async_session_maker() as db2:
            repo2 = PlanningJobRepository(db2)
            events = await repo2.get_events_after(job.id, after_id=e1.id)
            assert len(events) == 1
            assert events[0].id == e2.id


class TestWorkerE2E:
    """End-to-end tests for worker acquiring and completing jobs."""

    @pytest.mark.asyncio
    async def test_worker_acquires_and_completes_job(self, db: AsyncSession):
        """Worker picks up a pending job and runs it to completion."""
        from worker.planning_worker import PlanningWorker

        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-e2e",
            user_id="user-1",
            user_input="北京3天",
            user_feedback=feedback_with_trip("北京", 3, "2026-05-01"),
        )
        await db.commit()

        acquired = await repo.acquire_job("test-worker", lease_seconds=60)
        await db.commit()

        assert acquired is not None
        assert acquired.id == job.id

        worker = PlanningWorker("test-worker")
        final_payload = {
            "content": "北京3日行程方案",
            "itinerary": [{"day_number": 1, "activities": []}],
        }

        async def _fake_stream(*args, **kwargs):
            yield {"type": "thinking", "stage": "gathering", "payload": {}}
            yield {"type": "final", "stage": "completed", "payload": final_payload}

        cancel_event = asyncio.Event()
        with patch("graph.runner.stream_graph_events", side_effect=_fake_stream):
            status = await worker._run_graph_for_job(acquired, cancel_event)

        assert status == "completed"

        job_id = job.id
        db.expire_all()
        result = await db.execute(select(PlanningJob).where(PlanningJob.id == job_id))
        updated = result.scalar_one()
        assert updated.status == "completed"

        events = await repo.get_events_after(job_id, 0)
        assert events[0].stage == "running"
        assert events[-1].stage == "completed"
        assert events[-1].payload["content"] == "北京3日行程方案"

    @pytest.mark.asyncio
    async def test_worker_cancels_job(self, db: AsyncSession):
        """Cancel request stops job execution."""
        from worker.planning_worker import PlanningWorker

        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-cancel",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        await repo.acquire_job("test-worker", lease_seconds=60)
        await db.commit()

        # Request cancel
        await repo.request_cancel(job.id)
        await db.commit()

        worker = PlanningWorker("test-worker")
        assert await worker.check_cancelled(job.id) is True

        # Confirm cancel
        ok = await worker.mark_cancelled(job.id)
        assert ok is True

        await db.commit()
        job_id = job.id
        db.expire_all()
        result = await db.execute(select(PlanningJob).where(PlanningJob.id == job_id))
        updated = result.scalar_one()
        assert updated.status == "cancelled"
