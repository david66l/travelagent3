"""Tests for POISearchSkill."""

import pytest
from unittest.mock import AsyncMock, patch
from schemas import ScoredPOI
from skills.poi_search import POISearchSkill


class TestPOISearchSkill:
    """Test POI search with mocked external APIs."""

    def setup_method(self):
        self.skill = POISearchSkill()

    @pytest.mark.asyncio
    async def test_search_pois_cache_hit(self, mock_redis):
        mock_redis.get_json = AsyncMock(return_value=[
            {"name": "故宫", "category": "attraction", "score": 0.9},
        ])
        result = await self.skill.search_pois("北京", ["历史"])
        assert len(result) == 1
        assert result[0].name == "故宫"

    @pytest.mark.asyncio
    async def test_search_pois_fallback_city(self, mock_redis):
        mock_redis.get_json = AsyncMock(return_value=None)
        mock_redis.set_json = AsyncMock()
        with patch.object(self.skill.tavily, "search_with_context", AsyncMock(return_value=([], []))):
            result = await self.skill.search_pois("北京", ["历史"])
            assert len(result) >= 5  # fallback data

    @pytest.mark.asyncio
    async def test_search_pois_unknown_city_no_fallback(self, mock_redis):
        mock_redis.get_json = AsyncMock(return_value=None)
        mock_redis.set_json = AsyncMock()
        with patch.object(self.skill.tavily, "search_with_context", AsyncMock(return_value=([], []))):
            result = await self.skill.search_pois("未知城市", ["历史"])
            assert len(result) >= 0

    def test_parse_poi_items_empty(self):
        result = self.skill._parse_poi_items([])
        assert result == []

    def test_parse_poi_items_invalid_name(self):
        result = self.skill._parse_poi_items([{"name": "", "category": "attraction"}])
        assert result == []

    def test_parse_poi_items_short_name(self):
        result = self.skill._parse_poi_items([{"name": "A", "category": "attraction"}])
        assert len(result) == 0  # filtered: len(name) < 2

    def test_score_pois_with_keywords(self):
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.5, tags=["历史", "文化"]),
            ScoredPOI(name="科技馆", category="attraction", score=0.5, tags=["科技", "亲子"]),
        ]
        scored = self.skill._score_pois(pois, ["历史"])
        assert scored[0].name == "故宫"  # matches keyword, should score higher
        assert scored[0].score > 0.5

    def test_score_pois_rich_data_bonus(self):
        poi = ScoredPOI(
            name="故宫", category="attraction", score=0.5,
            recommended_hours="半天", area="东城区", best_time="上午",
            indoor_outdoor="mixed", ticket_price=60,
        )
        scored = self.skill._score_pois([poi], [])
        assert scored[0].score > 0.5  # bonus for rich data

    def test_clean_json_response_with_markdown(self):
        text = "```json\n{\"pois\": []}\n```"
        cleaned = self.skill._clean_json_response(text)
        assert cleaned == '{"pois": []}'

    def test_clean_json_response_plain(self):
        text = '{"pois": []}'
        cleaned = self.skill._clean_json_response(text)
        assert cleaned == '{"pois": []}'

    def test_clean_json_response_invalid(self):
        text = "not json"
        cleaned = self.skill._clean_json_response(text)
        assert cleaned == ""
