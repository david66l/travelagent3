"""Unit tests for ContentSafetyEngine."""

import pytest

from agents.content_safety import ContentSafetyEngine
from schemas import Activity, DayPlan


class TestContentSafetyEngine:
    def test_shopping_trip_blocked_by_user_input(self):
        state = {
            "user_input": "我要报一个低价购物团，零负团费那种",
            "itinerary": [],
        }
        result = ContentSafetyEngine.check(state)
        assert result.passed is False
        assert "shopping_trip" in result.critical_failures
        assert any("购物" in s for s in result.improvement_suggestions)

    def test_shopping_trip_blocked_by_high_shopping_ratio_and_low_budget(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="翡翠店", category="shopping", tags=["购物"]),
                    Activity(poi_name="乳胶枕店", category="shopping", tags=["购物"]),
                    Activity(poi_name="普通景点", category="attraction"),
                ],
            )
        ]
        state = {
            "user_input": "帮我看看这个行程",
            "itinerary": itinerary,
        }
        result = ContentSafetyEngine.check(state)
        assert result.passed is False
        assert "shopping_trip" in result.critical_failures

    def test_shopping_trip_allowed_with_normal_ratio(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="商场", category="shopping", tags=["购物"]),
                    Activity(poi_name="故宫", category="attraction"),
                    Activity(poi_name="烤鸭", category="restaurant"),
                    Activity(poi_name="长城", category="attraction"),
                ],
            )
        ]
        state = {
            "user_input": "北京三日游",
            "itinerary": itinerary,
        }
        result = ContentSafetyEngine.check(state)
        assert result.passed is True
        assert result.scores["shopping_trip"] > 0.0

    def test_illegal_route_blocked(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(
                        poi_name="未开放边境徒步",
                        category="attraction",
                        recommendation_reason="非法穿越路线",
                    ),
                ],
            )
        ]
        result = ContentSafetyEngine.check({"user_input": "x", "itinerary": itinerary})
        assert result.passed is False
        assert "illegal_route" in result.critical_failures

    def test_unsafe_activity_blocked(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(
                        poi_name="徒手攀岩",
                        category="attraction",
                    ),
                ],
            )
        ]
        result = ContentSafetyEngine.check({"user_input": "x", "itinerary": itinerary})
        assert result.passed is False
        assert "unsafe_activity" in result.critical_failures

    def test_clean_itinerary_passes(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="故宫", category="attraction", tags=["历史"]),
                    Activity(poi_name="全聚德", category="restaurant", tags=["烤鸭"]),
                ],
            )
        ]
        result = ContentSafetyEngine.check(
            {"user_input": "北京历史文化一日游", "itinerary": itinerary}
        )
        assert result.passed is True
        assert result.total_score == 1.0
        assert result.critical_failures == []

    def test_missing_state_passes(self):
        result = ContentSafetyEngine.check({})
        assert result.passed is True

    def test_missing_itinerary_with_user_input_passes(self):
        result = ContentSafetyEngine.check({"user_input": "你好"})
        assert result.passed is True

    def test_dict_itinerary_supported(self):
        state = {
            "user_input": "x",
            "itinerary": [
                {
                    "day_number": 1,
                    "activities": [
                        {"poi_name": "军事禁区", "category": "attraction"},
                    ],
                }
            ],
        }
        result = ContentSafetyEngine.check(state)
        assert result.passed is False
        assert "illegal_route" in result.critical_failures
