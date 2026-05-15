"""Unit tests for heuristic strategy engine."""

import pytest

from schemas import ScoredPOI, UserProfile
from planner.core.heuristic_strategy import build_strategy


class TestHeuristicStrategy:
    def test_empty_pois_returns_empty_strategy(self):
        profile = UserProfile(destination="北京", travel_days=3)
        strategy = build_strategy([], profile)
        assert strategy.day_themes == []
        assert strategy.must_see == []

    def test_groups_pois_by_area(self):
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9, area="东城区"),
            ScoredPOI(name="天坛", category="attraction", score=0.85, area="东城区"),
            ScoredPOI(name="颐和园", category="attraction", score=0.8, area="海淀区"),
        ]
        profile = UserProfile(destination="北京", travel_days=2)
        strategy = build_strategy(pois, profile)

        assert len(strategy.area_groups) == 2
        assert strategy.area_groups[0].area == "东城区"
        assert strategy.area_groups[0].poi_names == ["故宫", "天坛"]

    def test_assigns_day_themes_by_area(self):
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9, area="东城区"),
            ScoredPOI(name="天坛", category="attraction", score=0.85, area="东城区"),
            ScoredPOI(name="颐和园", category="attraction", score=0.8, area="海淀区"),
        ]
        profile = UserProfile(destination="北京", travel_days=2)
        strategy = build_strategy(pois, profile)

        assert len(strategy.day_themes) == 2
        assert strategy.day_themes[0].area_focus == "东城区"
        assert strategy.day_themes[1].area_focus == "海淀区"

    def test_detects_user_mentioned_must_see(self):
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9, area="东城区"),
        ]
        profile = UserProfile(
            destination="北京",
            travel_days=1,
            special_requests=["一定要去故宫"],
        )
        strategy = build_strategy(pois, profile)
        assert "故宫" in strategy.must_see

    def test_detects_landmark_must_see(self):
        pois = [
            ScoredPOI(
                name="博物馆",
                category="museum",
                score=0.9,
                area="东城区",
                tags=["博物馆"],
            ),
        ]
        profile = UserProfile(destination="北京", travel_days=1)
        strategy = build_strategy(pois, profile)
        assert "博物馆" in strategy.must_see

    def test_no_llm_dependency(self):
        """Strategy generation must not use LLM."""
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9, area="东城区"),
        ]
        profile = UserProfile(destination="北京", travel_days=1)
        strategy = build_strategy(pois, profile)
        # If this runs without any network/LLM mocks, it passes
        assert strategy.day_themes[0].theme == "东城区探索"
