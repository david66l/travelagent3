"""Tests for MemoryManagementAgent."""

import pytest
from unittest.mock import AsyncMock, patch
from agents.memory_management import MemoryManagementAgent
from schemas import UserProfile, DayPlan


class TestMemoryManagementAgent:
    """Test memory management operations."""

    @pytest.mark.asyncio
    async def test_save_conversation_turn(self):
        db = AsyncMock()
        with patch("agents.memory_management.MemoryStoreSkill.save_conversation", AsyncMock()) as mock_save:
            await MemoryManagementAgent.save_conversation_turn(
                db=db,
                session_id="sess-1",
                user_msg="你好",
                assistant_msg="你好！",
                intent="chitchat",
            )
            mock_save.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_itinerary(self):
        db = AsyncMock()
        itinerary = [DayPlan(day_number=1, activities=[])]
        profile = UserProfile(destination="北京", travel_days=1)
        with patch("agents.memory_management.MemoryStoreSkill.save_itinerary", AsyncMock(return_value="it-1")) as mock_save:
            result = await MemoryManagementAgent.save_itinerary(
                db=db,
                session_id="sess-1",
                user_id="user-1",
                itinerary=itinerary,
                profile=profile,
                budget={"total": 1000},
                confirmed=True,
            )
            assert result == "it-1"
            mock_save.assert_called_once()
            assert mock_save.call_args.kwargs["status"] == "confirmed"

    @pytest.mark.asyncio
    async def test_save_itinerary_draft(self):
        db = AsyncMock()
        itinerary = [DayPlan(day_number=1, activities=[])]
        profile = UserProfile(destination="成都")
        with patch("agents.memory_management.MemoryStoreSkill.save_itinerary", AsyncMock(return_value="it-2")) as mock_save:
            await MemoryManagementAgent.save_itinerary(
                db=db,
                session_id="sess-1",
                user_id="user-1",
                itinerary=itinerary,
                profile=profile,
                budget={},
                confirmed=False,
            )
            assert mock_save.call_args.kwargs["status"] == "draft"

    @pytest.mark.asyncio
    async def test_get_user_memory(self):
        db = AsyncMock()
        with patch("agents.memory_management.MemoryRetrieveSkill.get_user_memory", AsyncMock()) as mock_get:
            await MemoryManagementAgent.get_user_memory(db, "user-1", "sess-1")
            mock_get.assert_called_once_with(db, "user-1", "sess-1")

    @pytest.mark.asyncio
    async def test_save_preference_change(self):
        db = AsyncMock()
        with patch("agents.memory_management.MemoryStoreSkill.save_preference_change", AsyncMock()) as mock_save:
            await MemoryManagementAgent.save_preference_change(
                db=db,
                session_id="sess-1",
                user_id="user-1",
                change={"field": "food_preferences", "old_value": "[]", "new_value": '["辣"]'},
                source_message="我喜欢吃辣",
            )
            mock_save.assert_called_once()
