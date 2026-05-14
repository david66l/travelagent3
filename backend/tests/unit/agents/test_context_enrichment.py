"""Tests for ContextEnrichmentAgent."""

import pytest
from unittest.mock import AsyncMock, patch
from agents.context_enrichment import ContextEnrichmentAgent
from schemas import TravelContext, UserProfile


class TestEnrich:
    """Test context enrichment with mocked search."""

    def setup_method(self):
        self.agent = ContextEnrichmentAgent()

    @pytest.mark.asyncio
    async def test_enrich_with_destination(self):
        with patch.object(self.agent.search_skill, "search", AsyncMock(return_value=[])):
            result = await self.agent.enrich("北京", 3, "2026-05-01")
            assert isinstance(result, TravelContext)

    @pytest.mark.asyncio
    async def test_enrich_no_destination(self):
        result = await self.agent.enrich("", 0, "")
        assert isinstance(result, TravelContext)


class TestMergeInto:
    """Test TravelContext merging."""

    def test_merge_basic(self):
        target = TravelContext(route_suggestions="路线A\n")
        source = TravelContext(accommodation_areas="区域B", transport_tips="地铁")
        ContextEnrichmentAgent._merge_into(target, source)
        assert "路线A" in target.route_suggestions
        assert "区域B" in target.accommodation_areas
        assert "地铁" in target.transport_tips

    def test_merge_overwrite_empty(self):
        target = TravelContext(route_suggestions="")
        source = TravelContext(route_suggestions="路线B")
        ContextEnrichmentAgent._merge_into(target, source)
        assert "路线B" in target.route_suggestions

    def test_merge_preserve_existing(self):
        target = TravelContext(route_suggestions="路线A\n")
        source = TravelContext(route_suggestions="路线B")
        ContextEnrichmentAgent._merge_into(target, source)
        # Appends if not already contained as substring
        assert "路线A" in target.route_suggestions
        assert "路线B" in target.route_suggestions

    def test_merge_list_append(self):
        target = TravelContext(pitfall_tips=["tip1"])
        source = TravelContext(pitfall_tips=["tip2", "tip1"])  # duplicate
        ContextEnrichmentAgent._merge_into(target, source)
        assert set(target.pitfall_tips) == {"tip1", "tip2"}


class TestExtractYearMonth:
    """Test date parsing helper."""

    def test_iso_date(self):
        result = ContextEnrichmentAgent._extract_year_month("2026-05-01")
        assert result == ("2026", "05")

    def test_month_day(self):
        result = ContextEnrichmentAgent._extract_year_month("5月1日")
        assert result[1] == "05"

    def test_range(self):
        result = ContextEnrichmentAgent._extract_year_month("2026-05-01 to 2026-05-05")
        assert result == ("2026", "05")

    def test_invalid(self):
        result = ContextEnrichmentAgent._extract_year_month("不知道")
        # Should handle gracefully
        assert isinstance(result, tuple)
