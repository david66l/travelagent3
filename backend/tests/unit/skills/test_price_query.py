"""Tests for PriceQuerySkill."""

import pytest
from unittest.mock import AsyncMock, patch
from skills.price_query import PriceQuerySkill
from skills.web_search import SearchResult


class TestPriceQuerySkill:
    """Test price querying."""

    def setup_method(self):
        self.skill = PriceQuerySkill()

    @pytest.mark.asyncio
    async def test_query_price_ticket(self):
        with patch.object(self.skill.search_skill, "search", AsyncMock(return_value=[
            SearchResult(title="故宫门票", url="http://example.com", snippet="60元"),
        ])):
            result = await self.skill.query_price("故宫", "北京", "ticket")
            assert result.poi_name == "故宫"
            assert result.price_type == "ticket"
            assert result.price_range == "50-200元"
            assert result.source == "http://example.com"

    @pytest.mark.asyncio
    async def test_query_price_meal(self):
        with patch.object(self.skill.search_skill, "search", AsyncMock(return_value=[])):
            result = await self.skill.query_price("全聚德", "北京", "meal")
            assert result.price_range == "80-200元/人"
            assert result.source == ""

    @pytest.mark.asyncio
    async def test_query_price_hotel(self):
        with patch.object(self.skill.search_skill, "search", AsyncMock(return_value=[])):
            result = await self.skill.query_price("某酒店", "上海", "hotel")
            assert result.price_range == "300-800元/晚"

    @pytest.mark.asyncio
    async def test_query_price_unknown_type(self):
        with patch.object(self.skill.search_skill, "search", AsyncMock(return_value=[])):
            result = await self.skill.query_price("某物", "成都", "unknown")
            assert result.price_range is None
