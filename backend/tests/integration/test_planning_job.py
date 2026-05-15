"""Integration tests for PlanningJob repository with real PostgreSQL."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import async_session_maker
from models.planning_job import PlanningJob
from repositories.planning_job import PlanningJobRepository

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
        result = await db.execute(
            select(PlanningJob).where(PlanningJob.id == job.id)
        )
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
        result = await db.execute(
            select(PlanningJob).where(PlanningJob.id == acquired.id)
        )
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
        await db.execute(
            select(PlanningJob)
            .where(PlanningJob.id == acquired.id)
            .with_for_update()
        )
        await db.execute(
            update(PlanningJob)
            .where(PlanningJob.id == acquired.id)
            .values(locked_by="worker-2")
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
        result = await db.execute(
            select(PlanningJob).where(PlanningJob.id == acquired.id)
        )
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
            update(PlanningJob)
            .where(PlanningJob.id == acquired.id)
            .values(locked_by="worker-2")
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
            update(PlanningJob)
            .where(PlanningJob.id == acquired.id)
            .values(locked_by="worker-2")
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
        result = await db.execute(
            select(PlanningJob).where(PlanningJob.id == acquired.id)
        )
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
            .values(lock_expires_at=datetime.utcnow() - timedelta(seconds=1))
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
        from schemas import IntentResult, ScoredPOI, WeatherDay

        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-e2e",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        # Mock intent recognition and data collection
        mock_intent = AsyncMock(return_value=IntentResult(
            intent="generate_itinerary",
            confidence=0.95,
            user_entities={
                "destination": "北京",
                "travel_days": 3,
                "travel_dates": "2026-05-01",
            },
        ))
        mock_pois = AsyncMock(return_value=[
            ScoredPOI(name="故宫", category="attraction", score=0.9, area="东城区"),
            ScoredPOI(name="天坛", category="attraction", score=0.85, area="东城区"),
            ScoredPOI(name="颐和园", category="attraction", score=0.85, area="海淀区"),
        ])
        mock_weather = AsyncMock(return_value=[
            WeatherDay(date="2026-05-01", condition="晴", temp_high=25, temp_low=15, precipitation_chance=0),
            WeatherDay(date="2026-05-02", condition="多云", temp_high=24, temp_low=14, precipitation_chance=10),
            WeatherDay(date="2026-05-03", condition="晴", temp_high=26, temp_low=16, precipitation_chance=0),
        ])

        # Mock Redis publish
        with patch("core.redis_client.redis_client._client.publish", AsyncMock()):
            with patch("agents.intent_recognition.IntentRecognitionAgent.recognize", mock_intent):
                with patch("agents.realtime_query.RealtimeQueryAgent.query_pois", mock_pois):
                    with patch("agents.realtime_query.RealtimeQueryAgent.query_weather", mock_weather):
                        worker = PlanningWorker("test-worker")
                        # Run one iteration manually
                        acquired = await repo.acquire_job("test-worker", lease_seconds=60)
                        await db.commit()

                        assert acquired is not None
                        assert acquired.id == job.id

                        # Run pipeline
                        from pipeline.planning_pipeline import PlanningPipeline
                        pipeline = PlanningPipeline(worker=worker)
                        await pipeline.run(acquired)

                        # Verify job completed (use fresh session to avoid identity map cache)
                        job_id = job.id
                        db.expire_all()
                        result = await db.execute(
                            select(PlanningJob).where(PlanningJob.id == job_id)
                        )
                        updated = result.scalar_one()
                        assert updated.status == "completed"
                        assert updated.locked_by is None
                        assert updated.proposal_text is not None
                        assert "北京" in updated.proposal_text

                        events = await repo.get_events_after(job_id, 0)
                        assert events[0].stage == "running"
                        assert events[-1].stage == "completed"
                        assert "北京" in events[-1].payload["proposal_text"]

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
        result = await db.execute(
            select(PlanningJob).where(PlanningJob.id == job_id)
        )
        updated = result.scalar_one()
        assert updated.status == "cancelled"

    @pytest.mark.asyncio
    async def test_pipeline_cancels_running_graph(self, db: AsyncSession):
        """A cancellation request interrupts a long-running pipeline task."""
        from pipeline.planning_pipeline import PlanningPipeline
        from worker.planning_worker import PlanningWorker
        from schemas import IntentResult

        intent_started = asyncio.Event()

        async def slow_intent(*args, **kwargs):
            intent_started.set()
            await asyncio.Event().wait()
            return IntentResult(
                intent="generate_itinerary",
                confidence=0.95,
                user_entities={"destination": "北京", "travel_days": 3},
            )

        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-cancel-running",
            user_id="user-1",
            user_input="北京3天",
        )
        await db.commit()

        acquired = await repo.acquire_job("test-worker", lease_seconds=60)
        await db.commit()
        job_id = job.id

        with patch("core.redis_client.redis_client._client.publish", AsyncMock()):
            with patch("agents.intent_recognition.IntentRecognitionAgent.recognize", slow_intent):
                worker = PlanningWorker("test-worker")
                pipeline = PlanningPipeline(worker=worker)
                task = asyncio.create_task(pipeline.run(acquired))

                await asyncio.wait_for(intent_started.wait(), timeout=2)
                task.cancel()

                try:
                    await asyncio.wait_for(task, timeout=7)
                except asyncio.CancelledError:
                    pass

        db.expire_all()
        result = await db.execute(
            select(PlanningJob).where(PlanningJob.id == job_id)
        )
        updated = result.scalar_one()
        assert updated.status == "cancelled"

        events = await repo.get_events_after(job_id, 0)
        assert events[-1].stage == "cancelled"
        assert events[-1].event_type == "cancelled"
