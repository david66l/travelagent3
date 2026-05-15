"""Isolated fallback behavior tests."""
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from models.planning_job import PlanningJob
from repositories.planning_job import PlanningJobRepository
from pipeline.planning_pipeline import PlanningPipeline
from worker.planning_worker import PlanningWorker
from schemas import IntentResult


class TestFallbackIsolated:
    @pytest.mark.asyncio
    async def test_ping(self, db: AsyncSession):
        """Sanity check that the fixture works in this class."""
        result = await db.execute(text("SELECT 1"))
        assert result.fetchone() == (1,)

    @pytest.mark.asyncio
    async def test_fallback_on_slow_poi_search(self, db: AsyncSession):
        repo = PlanningJobRepository(db)
        job = await repo.create(session_id="sess-fallback", user_id="user-1", user_input="上海3天")
        await db.commit()

        mock_intent = AsyncMock(return_value=IntentResult(
            intent="generate_itinerary",
            confidence=0.95,
            user_entities={
                "destination": "上海",
                "travel_days": 3,
                "travel_dates": "2026-05-01",
                "travelers_count": 2,
            },
            missing_required=[],
        ))

        async def _timeout(*a, **k):
            raise asyncio.TimeoutError
        mock_pois = MagicMock(side_effect=_timeout)

        mock_weather = AsyncMock(return_value=[])

        with patch("agents.intent_recognition.IntentRecognitionAgent.recognize", mock_intent):
            with patch("agents.realtime_query.RealtimeQueryAgent.query_pois", mock_pois):
                with patch("agents.realtime_query.RealtimeQueryAgent.query_weather", mock_weather):
                    worker = PlanningWorker("test-worker")
                    acquired = await repo.acquire_job("test-worker", lease_seconds=60)
                    await db.commit()

                    pipeline = PlanningPipeline(worker=worker)
                    await pipeline.run(acquired)

        job_id = job.id
        db.expire_all()
        result = await db.execute(select(PlanningJob).where(PlanningJob.id == job_id))
        updated = result.scalar_one()
        assert updated.status == "completed"
        assert updated.itinerary_draft is not None
