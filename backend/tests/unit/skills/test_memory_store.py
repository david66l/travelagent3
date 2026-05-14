"""Tests for MemoryStoreSkill."""

import pytest
from unittest.mock import AsyncMock
from skills.memory_store import MemoryStoreSkill


class TestMemoryStoreSkill:
    """Test database persistence operations."""

    @pytest.mark.asyncio
    async def test_save_conversation(self, db_session):
        await MemoryStoreSkill.save_conversation(
            db=db_session,
            session_id="sess-123",
            user_message="你好",
            assistant_response="你好！有什么可以帮您的？",
            intent="chitchat",
        )
        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_itinerary(self, db_session):
        itinerary_id = await MemoryStoreSkill.save_itinerary(
            db=db_session,
            session_id="sess-123",
            user_id="user-1",
            destination="北京",
            travel_days=3,
            daily_plans=[{"day_number": 1, "activities": []}],
            preference_snapshot={"destination": "北京"},
            budget_snapshot={"total": 5000},
            status="confirmed",
        )
        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()
        db_session.refresh.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_preference_change(self, db_session):
        await MemoryStoreSkill.save_preference_change(
            db=db_session,
            session_id="sess-123",
            user_id="user-1",
            field="food_preferences",
            old_value="[\"辣\"]",
            new_value="[\"辣\", \"甜品\"]",
            source_message="我还喜欢吃甜品",
        )
        db_session.add.assert_called_once()
        db_session.commit.assert_called_once()
