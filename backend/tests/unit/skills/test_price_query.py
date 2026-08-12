"""Tests for PriceQuerySkill."""

import pytest
from unittest.mock import AsyncMock, patch
from core.local_cache import tool_local_cache
from skills.price_query import PriceQuerySkill


class TestPriceQuerySkill:
    """Test price querying."""

    def setup_method(self):
        self.skill = PriceQuerySkill()

    @pytest.fixture(autouse=True)
    async def _clear_l1(self):
        await tool_local_cache.clear()
        yield
        await tool_local_cache.clear()

    @pytest.mark.asyncio
    async def test_query_price_ticket_fallback(self):
        result = await self.skill.query_price("故宫", "北京", "ticket")
        assert result.poi_name == "故宫"
        assert result.price_type == "ticket"
        assert result.price_range == "¥40-60"
        assert result.data_source == "built_in"
        assert result.is_fallback is True

    @pytest.mark.asyncio
    async def test_query_price_meal_fallback(self):
        result = await self.skill.query_price("全聚德", "北京", "meal")
        assert result.price_range is not None
        assert result.data_source == "built_in"
        assert result.is_fallback is True

    @pytest.mark.asyncio
    async def test_query_price_hotel_fallback(self):
        result = await self.skill.query_price("某酒店", "上海", "hotel")
        assert result.price_range is not None
        assert result.data_source == "built_in"

    @pytest.mark.asyncio
    async def test_query_price_unknown_type(self):
        result = await self.skill.query_price("某物", "成都", "unknown")
        assert result.price_range is None
        assert result.data_source == "fallback"

    @pytest.mark.asyncio
    async def test_query_price_api_path(self):
        from schemas import PriceInfo, ToolResult

        api_info = PriceInfo(
            poi_name="故宫",
            price_type="ticket",
            price_range="60元",
            source="api",
            data_source="api",
            confidence=0.8,
            is_fallback=False,
        )
        with patch.object(
            self.skill,
            "execute",
            AsyncMock(return_value=ToolResult(data=api_info, data_source="api")),
        ):
            result = await self.skill.query_price("故宫", "北京", "ticket")
            assert result.data_source == "api"
            assert result.is_fallback is False
