"""Integration tests for P3 — disconnect / reconnect recovery."""
import asyncio
import json
import pytest
from unittest.mock import patch, AsyncMock

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import async_session_maker
from models.planning_job import PlanningJob
from repositories.planning_job import PlanningJobRepository
from schemas import IntentResult, ScoredPOI, WeatherDay


class TestReconnect:
    @pytest.mark.asyncio
    async def test_state_restored_on_reconnect(self, db: AsyncSession):
        """After a completed job, reconnect restores profile + revision."""
        repo = PlanningJobRepository(db)
        mock_intent = AsyncMock(return_value=IntentResult(
            intent="generate_itinerary", confidence=0.95,
            user_entities={"destination": "上海", "travel_days": 2},
        ))
        mock_pois = AsyncMock(return_value=[
            ScoredPOI(name="外滩", category="attraction", score=0.9, area="外滩"),
        ])
        mock_weather = AsyncMock(return_value=[
            WeatherDay(date="2026-06-01", condition="晴", temp_high=25, temp_low=15, precipitation_chance=0),
        ])

        from pipeline.planning_pipeline import PlanningPipeline
        from worker.planning_worker import PlanningWorker

        # 1. First job — generate itinerary
        with patch("agents.intent_recognition.IntentRecognitionAgent.recognize", mock_intent):
            with patch("agents.realtime_query.RealtimeQueryAgent.query_pois", mock_pois):
                with patch("agents.realtime_query.RealtimeQueryAgent.query_weather", mock_weather):
                    job = await repo.create(
                        session_id="reconnect-1",
                        user_id="u1",
                        user_input="上海2天",
                        user_feedback={
                            "schema_version": 1,
                            "recent_messages": [
                                {"role": "user", "content": "上海2天", "ts": 1000}
                            ],
                            "profile": {"destination": "上海", "travel_days": 2,
                                        "interests": ["历史"]},
                            "phase": "planning",
                            "turn": 1,
                            "revision": 1,
                        },
                    )
                    await db.commit()

                    worker = PlanningWorker("w1")
                    acquired = await repo.acquire_job("w1", lease_seconds=60)
                    await db.commit()

                    pipeline = PlanningPipeline(worker=worker)
                    await pipeline.run(acquired)

        # 2. Verify job completed (fresh session to avoid MissingGreenlet)
        async with async_session_maker() as check_db:
            result = await check_db.execute(
                select(PlanningJob).where(PlanningJob.id == job.id)
            )
            updated = result.scalar_one()
            assert updated.status == "completed"

        # 3. Simulate reconnect: load state from latest job
        async with async_session_maker() as fresh_db:
            fresh_repo = PlanningJobRepository(fresh_db)
            latest = await fresh_repo.get_by_session("reconnect-1", limit=1)
            assert len(latest) == 1
            state = latest[0].user_feedback
            assert state is not None
            assert state.get("profile", {}).get("destination") == "上海"
            assert state.get("profile", {}).get("travel_days") == 2
            assert state.get("revision") == 1
            assert len(state.get("recent_messages", [])) >= 1

    @pytest.mark.asyncio
    async def test_reconnect_after_revision_bump(self, db: AsyncSession):
        """Reconnect after revision bump sees the new revision number."""
        # Create state with two completed revisions
        job1 = await PlanningJobRepository(db).create(
            session_id="reconnect-rev",
            user_id="u1",
            user_input="成都4天",
            user_feedback={
                "schema_version": 1,
                "profile": {"destination": "成都", "travel_days": 4, "pace": "moderate"},
                "phase": "completed",
                "turn": 4,
                "revision": 2,
                "recent_messages": [
                    {"role": "user", "content": "成都4天", "ts": 1000},
                    {"role": "assistant", "content": "行程已生成", "ts": 1001},
                    {"role": "user", "content": "改轻松一点", "ts": 1002},
                    {"role": "assistant", "content": "行程已生成", "ts": 1003},
                ],
            },
        )
        await db.commit()

        # Simulate reconnect
        async with async_session_maker() as fresh_db:
            fresh_repo = PlanningJobRepository(fresh_db)
            latest = await fresh_repo.get_by_session("reconnect-rev", limit=1)
            state = latest[0].user_feedback
            assert state["revision"] == 2
            assert state["profile"]["pace"] == "moderate"
            assert state["phase"] == "completed"
            assert len(state["recent_messages"]) == 4
