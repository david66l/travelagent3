"""Tests for MemoryRetrieveSkill."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime
from skills.memory_retrieve import MemoryRetrieveSkill


class TestMemoryRetrieveSkill:
    """Test database query operations."""

    def _make_mock_result(self, items):
        result = MagicMock()
        result.scalars.return_value.all.return_value = items
        return result

    @pytest.mark.asyncio
    async def test_get_recent_conversations(self):
        mock_item = MagicMock()
        mock_item.user_message = "你好"
        mock_item.assistant_response = "你好！"
        mock_item.intent = "chitchat"
        mock_item.created_at = datetime(2026, 5, 1, 10, 0)

        db = AsyncMock()
        db.execute = AsyncMock(return_value=self._make_mock_result([mock_item]))

        result = await MemoryRetrieveSkill.get_recent_conversations(db, "sess-1")
        assert len(result) == 1
        assert result[0]["user_message"] == "你好"
        assert result[0]["intent"] == "chitchat"

    @pytest.mark.asyncio
    async def test_get_recent_conversations_empty(self):
        db = AsyncMock()
        db.execute = AsyncMock(return_value=self._make_mock_result([]))
        result = await MemoryRetrieveSkill.get_recent_conversations(db, "sess-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_recent_itineraries(self):
        mock_it = MagicMock()
        mock_it.id = 1
        mock_it.destination = "北京"
        mock_it.travel_days = 3
        mock_it.daily_plans = []
        mock_it.preference_snapshot = {}
        mock_it.created_at = datetime(2026, 5, 1)

        db = AsyncMock()
        db.execute = AsyncMock(return_value=self._make_mock_result([mock_it]))

        result = await MemoryRetrieveSkill.get_recent_itineraries(db, "user-1")
        assert len(result) == 1
        assert result[0]["destination"] == "北京"

    @pytest.mark.asyncio
    async def test_get_preference_changes(self):
        mock_change = MagicMock()
        mock_change.field = "food_preferences"
        mock_change.old_value = '["辣"]'
        mock_change.new_value = '["辣", "甜品"]'
        mock_change.created_at = datetime(2026, 5, 1)

        db = AsyncMock()
        db.execute = AsyncMock(return_value=self._make_mock_result([mock_change]))

        result = await MemoryRetrieveSkill.get_preference_changes(db, "sess-1")
        assert len(result) == 1
        assert result[0]["field"] == "food_preferences"

    @pytest.mark.asyncio
    async def test_get_user_memory(self):
        mock_it = MagicMock()
        mock_it.id = 1
        mock_it.destination = "北京"
        mock_it.travel_days = 3
        mock_it.daily_plans = []
        mock_it.preference_snapshot = {"pace": "moderate", "food_preferences": ["烤鸭"], "budget_range": 5000}
        mock_it.created_at = datetime(2026, 5, 1)

        mock_conv = MagicMock()
        mock_conv.user_message = "你好"
        mock_conv.assistant_response = "你好！"
        mock_conv.intent = "chitchat"
        mock_conv.created_at = datetime(2026, 5, 1)

        db = AsyncMock()
        db.execute = AsyncMock(side_effect=[
            self._make_mock_result([mock_it]),
            self._make_mock_result([mock_conv]),
        ])

        result = await MemoryRetrieveSkill.get_user_memory(db, "user-1", "sess-1")
        assert result.recent_itineraries[0]["destination"] == "北京"
        assert result.preference_patterns["preferred_pace"] == "moderate"
        assert "烤鸭" in result.preference_patterns["preferred_food"]
        assert result.preference_patterns["avg_budget"] == 5000.0
        assert result.preference_patterns["trip_count"] == 1

    def test_extract_patterns_empty(self):
        result = MemoryRetrieveSkill._extract_patterns([])
        assert result == {}

    def test_extract_patterns_multiple(self):
        itineraries = [
            {"destination": "北京", "preference_snapshot": {"pace": "moderate", "food_preferences": ["烤鸭"], "budget_range": 4000}},
            {"destination": "上海", "preference_snapshot": {"pace": "relaxed", "food_preferences": ["小笼包"], "budget_range": 6000}},
        ]
        result = MemoryRetrieveSkill._extract_patterns(itineraries)
        assert result["avg_budget"] == 5000.0
        assert result["trip_count"] == 2
        assert set(result["favorite_cities"]) == {"北京", "上海"}
