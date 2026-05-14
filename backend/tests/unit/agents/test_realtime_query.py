"""Tests for RealtimeQueryAgent."""

import pytest
from unittest.mock import AsyncMock, patch
from agents.realtime_query import RealtimeQueryAgent


class TestRealtimeQueryAgent:
    """Test parallel real-time queries."""

    def setup_method(self):
        self.agent = RealtimeQueryAgent()

    @pytest.mark.asyncio
    async def test_query_all(self):
        mock_poi = AsyncMock(return_value=[])
        mock_weather = AsyncMock(return_value=[])
        mock_price = AsyncMock(return_value={})

        with patch.object(self.agent.poi_skill, "search_pois", mock_poi):
            with patch.object(self.agent.weather_skill, "query", mock_weather):
                with patch.object(self.agent, "_query_prices", mock_price):
                    from schemas import UserProfile
                    result = await self.agent.query_all(
                        city="北京",
                        dates=["2026-05-01"],
                        keywords=["历史"],
                        profile=UserProfile(),
                    )
                    assert result == ([], [], {})
                    mock_poi.assert_called_once_with("北京", ["历史"])
                    mock_weather.assert_called_once_with("北京", "2026-05-01", "2026-05-01")

    @pytest.mark.asyncio
    async def test_query_pois(self):
        with patch.object(self.agent.poi_skill, "search_pois", AsyncMock(return_value=[])) as mock:
            result = await self.agent.query_pois("北京", ["历史"])
            assert result == []
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_weather(self):
        with patch.object(self.agent.weather_skill, "query", AsyncMock(return_value=[])) as mock:
            result = await self.agent.query_weather("北京", "2026-05-01", "2026-05-03")
            assert result == []
            mock.assert_called_once_with("北京", "2026-05-01", "2026-05-03")

    @pytest.mark.asyncio
    async def test_query_prices(self):
        from schemas import ScoredPOI
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9),
            ScoredPOI(name="全聚德", category="restaurant", score=0.8),
        ]
        with patch.object(self.agent.price_skill, "query_price", AsyncMock(return_value=AsyncMock(poi_name="故宫"))) as mock:
            result = await self.agent._query_prices("北京", pois)
            assert len(result) == 1  # only attractions
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_prices_empty(self):
        result = await self.agent._query_prices("北京", [])
        assert result == {}
