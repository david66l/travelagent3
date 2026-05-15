"""Integration tests for Planning Core (Phase 2A)."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.planning_job import PlanningJob
from repositories.planning_job import PlanningJobRepository
from schemas import IntentResult, ScoredPOI, WeatherDay


class TestPlanningCoreE2E:
    """End-to-end tests for Planning Core pipeline."""

    @pytest.mark.asyncio
    async def test_full_job_flow_to_completed(self, db: AsyncSession):
        """Job runs through all 2A stages and completes."""
        from worker.planning_worker import PlanningWorker
        from pipeline.planning_pipeline import PlanningPipeline

        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-core-e2e",
            user_id="user-1",
            user_input="上海3天",
        )
        await db.commit()

        mock_intent = AsyncMock(return_value=IntentResult(
            intent="generate_itinerary",
            confidence=0.95,
            user_entities={
                "destination": "上海",
                "travel_days": 3,
                "travel_dates": "2026-05-01",
            },
        ))
        mock_pois = AsyncMock(return_value=[
            ScoredPOI(name="外滩", category="attraction", score=0.9, area="外滩"),
            ScoredPOI(name="东方明珠", category="attraction", score=0.9, area="陆家嘴"),
            ScoredPOI(name="豫园", category="attraction", score=0.85, area="城隍庙"),
        ])
        mock_weather = AsyncMock(return_value=[
            WeatherDay(date="2026-05-01", condition="晴", temp_high=25, temp_low=15, precipitation_chance=0),
            WeatherDay(date="2026-05-02", condition="多云", temp_high=24, temp_low=14, precipitation_chance=10),
            WeatherDay(date="2026-05-03", condition="晴", temp_high=26, temp_low=16, precipitation_chance=0),
        ])

        with patch("core.redis_client.redis_client._client.publish", AsyncMock()):
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
        assert updated.locked_by is None
        assert updated.itinerary_draft is not None
        assert updated.itinerary_final is not None
        assert updated.proposal_text is not None
        assert "上海" in updated.proposal_text

        events = await repo.get_events_after(job_id, 0)
        stages = [e.stage for e in events]
        assert "running" in stages
        assert "draft_ready" in stages
        assert "itinerary_final" in stages
        assert "writing" in stages
        assert "completed" in stages

    @pytest.mark.asyncio
    async def test_fallback_on_slow_poi_search(self, db: AsyncSession):
        """If POI search > 3s, fallback draft is emitted immediately."""
        from worker.planning_worker import PlanningWorker
        from pipeline.planning_pipeline import PlanningPipeline

        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-fallback",
            user_id="user-1",
            user_input="上海3天",
        )
        await db.commit()

        mock_intent = AsyncMock(return_value=IntentResult(
            intent="generate_itinerary",
            confidence=0.95,
            user_entities={
                "destination": "上海",
                "travel_days": 3,
                "travel_dates": "2026-05-01",
            },
        ))

        async def _slow_pois(*args, **kwargs):
            await asyncio.sleep(10)
        slow_pois = AsyncMock(side_effect=_slow_pois)

        mock_weather = AsyncMock(return_value=[])

        with patch("core.redis_client.redis_client._client.publish", AsyncMock()):
            with patch("agents.intent_recognition.IntentRecognitionAgent.recognize", mock_intent):
                with patch("agents.realtime_query.RealtimeQueryAgent.query_pois", slow_pois):
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
        # Fallback should use CITY_DEFAULTS
        assert updated.itinerary_draft is not None
        draft = updated.itinerary_draft
        assert isinstance(draft, list)
        assert len(draft) > 0

    @pytest.mark.asyncio
    async def test_event_sequence_normal_flow(self, db: AsyncSession):
        """Verify exact event order for successful job."""
        from worker.planning_worker import PlanningWorker
        from pipeline.planning_pipeline import PlanningPipeline

        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id="sess-events",
            user_id="user-1",
            user_input="北京2天",
        )
        await db.commit()

        mock_intent = AsyncMock(return_value=IntentResult(
            intent="generate_itinerary",
            confidence=0.95,
            user_entities={
                "destination": "北京",
                "travel_days": 2,
                "travel_dates": "2026-05-01",
            },
        ))
        mock_pois = AsyncMock(return_value=[
            ScoredPOI(name="故宫", category="attraction", score=0.9, area="东城区"),
            ScoredPOI(name="天坛", category="attraction", score=0.85, area="东城区"),
        ])
        mock_weather = AsyncMock(return_value=[
            WeatherDay(date="2026-05-01", condition="晴", temp_high=25, temp_low=15, precipitation_chance=0),
            WeatherDay(date="2026-05-02", condition="多云", temp_high=24, temp_low=14, precipitation_chance=10),
        ])

        with patch("core.redis_client.redis_client._client.publish", AsyncMock()):
            with patch("agents.intent_recognition.IntentRecognitionAgent.recognize", mock_intent):
                with patch("agents.realtime_query.RealtimeQueryAgent.query_pois", mock_pois):
                    with patch("agents.realtime_query.RealtimeQueryAgent.query_weather", mock_weather):
                        worker = PlanningWorker("test-worker")
                        acquired = await repo.acquire_job("test-worker", lease_seconds=60)
                        await db.commit()

                        pipeline = PlanningPipeline(worker=worker)
                        await pipeline.run(acquired)

        events = await repo.get_events_after(job.id, 0)
        stages = [e.stage for e in events]

        expected = ["running", "draft_ready", "itinerary_final", "writing", "completed"]
        assert stages == expected, f"Expected {expected}, got {stages}"
