"""Integration tests for P3 — disconnect / reconnect recovery."""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from core.database import async_session_maker
from models.planning_job import PlanningJob
from repositories.planning_job import PlanningJobRepository
from tests.support.planning_feedback import feedback_with_trip


class TestReconnect:
    @pytest.mark.asyncio
    async def test_state_restored_on_reconnect(self, db: AsyncSession):
        """After a completed job, reconnect restores profile + revision."""
        repo = PlanningJobRepository(db)
        feedback = feedback_with_trip("上海", 2, user_input="上海2天")
        feedback["phase"] = "completed"
        feedback["revision"] = 1
        feedback["recent_messages"] = [
            {"role": "user", "content": "上海2天", "ts": 1000},
            {"role": "assistant", "content": "行程已生成", "ts": 1001},
        ]

        job = await repo.create(
            session_id="reconnect-1",
            user_id="u1",
            user_input="上海2天",
            user_feedback=feedback,
        )
        await db.execute(
            update(PlanningJob)
            .where(PlanningJob.id == job.id)
            .values(status="completed", proposal_text="上海2天行程草案")
        )
        await db.commit()

        async with async_session_maker() as check_db:
            result = await check_db.execute(select(PlanningJob).where(PlanningJob.id == job.id))
            updated = result.scalar_one()
            assert updated.status == "completed"

        async with async_session_maker() as fresh_db:
            fresh_repo = PlanningJobRepository(fresh_db)
            latest = await fresh_repo.get_by_session("reconnect-1", limit=1)
            assert len(latest) == 1
            state = latest[0].user_feedback
            assert state is not None
            trip = state.get("profile", {}).get("trip", state.get("profile", {}))
            assert trip.get("destination") == "上海"
            assert trip.get("travel_days") == 2
            assert state.get("revision") == 1
            assert len(state.get("recent_messages", [])) >= 1

    @pytest.mark.asyncio
    async def test_reconnect_after_revision_bump(self, db: AsyncSession):
        """Reconnect after revision bump sees the new revision number."""
        _job1 = await PlanningJobRepository(db).create(
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

        async with async_session_maker() as fresh_db:
            fresh_repo = PlanningJobRepository(fresh_db)
            latest = await fresh_repo.get_by_session("reconnect-rev", limit=1)
            state = latest[0].user_feedback
            assert state["revision"] == 2
            assert state["profile"]["pace"] == "moderate"
            assert state["phase"] == "completed"
            assert len(state["recent_messages"]) == 4
