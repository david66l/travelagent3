"""Unit tests for MemoryConflictResolver."""

import pytest

from agents.memory_conflict_resolver import MemoryConflictResolver


@pytest.fixture
def resolver():
    return MemoryConflictResolver()


class TestMemoryConflictResolver:
    def test_short_term_destination_wins(self, resolver):
        result = resolver.resolve(
            {"destination": "成都"},
            {"destination": "北京", "interests": ["历史"]},
        )
        assert result["destination"] == "成都"

    def test_long_term_used_when_short_term_missing(self, resolver):
        result = resolver.resolve(
            {"travel_days": None},
            {"travel_days": 5, "pace": "relaxed"},
        )
        assert result["travel_days"] == 5
        assert result["pace"] == "relaxed"

    def test_food_taboos_merge_and_short_term_priority(self, resolver):
        result = resolver.resolve(
            {"food_taboos": ["香菜"]},
            {"food_taboos": ["花生"], "food_preferences": ["辣"]},
        )
        assert "香菜" in result["food_taboos"]
        assert "花生" in result["food_taboos"]

    def test_conservative_walk_minutes(self, resolver):
        result = resolver.resolve(
            {"max_walk_minutes": 90},
            {"max_walk_minutes": 180},
        )
        assert result["max_walk_minutes"] == 90

    def test_conservative_special_flags(self, resolver):
        result = resolver.resolve(
            {"has_elderly": False},
            {"has_elderly": True},
        )
        assert result["has_elderly"] is True

    def test_short_term_special_flag_overrides(self, resolver):
        result = resolver.resolve(
            {"has_children": True},
            {"has_children": False},
        )
        assert result["has_children"] is True

    def test_short_term_origin_and_dates_win(self, resolver):
        result = resolver.resolve(
            {"origin": "济南", "travel_dates": "2026-06-21", "total_budget": 5000},
            {"origin": "北京", "travel_dates": "2026-05-01"},
        )
        assert result["origin"] == "济南"
        assert result["travel_dates"] == "2026-06-21"
        assert result["total_budget"] == 5000

    def test_default_values_when_both_empty(self, resolver):
        result = resolver.resolve({}, {})
        assert result["pace"] == "moderate"
        assert result["transport_preference"] is None
        assert result["max_walk_minutes"] == 180
        assert result["max_transit_minutes"] == 120
        assert result["travelers_count"] is None
        assert result["has_children"] is None

    def test_trip_poi_constraints_and_fatigue_survive_resolution(self, resolver):
        result = resolver.resolve(
            {
                "must_visit": ["故宫"],
                "must_not_visit": ["外滩"],
                "fatigue_preference": "low",
            },
            {},
        )

        assert result["must_visit"] == ["故宫"]
        assert result["must_not_visit"] == ["外滩"]
        assert result["fatigue_preference"] == "low"
