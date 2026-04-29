"""Tests for POI fuzzy matching logic."""

import pytest
from agents.itinerary_planner import ItineraryPlannerAgent
from schemas import ScoredPOI, Location


class TestPOIMatching:
    """Test POI fuzzy match with confidence thresholds."""

    def setup_method(self):
        self.agent = ItineraryPlannerAgent()
        self.original_pois = [
            ScoredPOI(name="故宫博物院", category="attraction", score=0.9, location=Location(lat=39.9, lng=116.4)),
            ScoredPOI(name="东方明珠广播电视塔", category="attraction", score=0.9, location=Location(lat=31.2, lng=121.5)),
            ScoredPOI(name="外滩", category="attraction", score=0.9, location=Location(lat=31.2, lng=121.5)),
            ScoredPOI(name="南翔馒头店", category="restaurant", score=0.7),
        ]

    def test_exact_match(self):
        result = self.agent._find_original_poi("故宫博物院", self.original_pois)
        assert result is not None
        assert result.name == "故宫博物院"

    def test_substring_match(self):
        result = self.agent._find_original_poi("故宫博", self.original_pois)
        assert result is not None
        assert result.name == "故宫博物院"

    def test_no_match(self):
        result = self.agent._find_original_poi("不存在景点", self.original_pois)
        assert result is None

    def test_similarity_match(self):
        result = self.agent._find_original_poi("东方明珠塔", self.original_pois)
        assert result is not None
        assert result.name == "东方明珠广播电视塔"

    def test_empty_input(self):
        result = self.agent._find_original_poi("", self.original_pois)
        assert result is None

    def test_short_substring_rejected(self):
        # "南" is too short, should not match "南翔馒头店"
        result = self.agent._find_original_poi("南", self.original_pois)
        assert result is None
