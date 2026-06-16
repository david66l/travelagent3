"""Tests for PRD cache key builders."""

from core.cache_keys import (
    CACHE_VERSION,
    itinerary_draft_key,
    poi_key,
    price_key,
    tool_cache_key,
    weather_key,
)
from schemas import UserProfile


def test_poi_key_format():
    key = poi_key("北京", "attraction", ["美食", "历史"])
    assert key.startswith("poi:北京:attraction:")
    assert key.endswith(f":{CACHE_VERSION}")


def test_weather_key_range():
    key = weather_key("上海", "2026-06-01", "2026-06-03")
    assert key == f"weather:上海:2026-06-01_2026-06-03:{CACHE_VERSION}"


def test_price_key_format():
    key = price_key("故宫", "2026-06-01", "ticket")
    assert key == f"price:故宫:2026-06-01:ticket:{CACHE_VERSION}"


def test_itinerary_draft_key_stable():
    profile = UserProfile(
        destination="成都",
        travel_days=3,
        budget_range=2000,
        interests=["美食"],
        pace="moderate",
    )
    key1 = itinerary_draft_key(profile)
    key2 = itinerary_draft_key(profile.model_dump())
    assert key1 == key2
    assert key1.startswith("itinerary:成都:")


def test_tool_cache_key_poi_search():
    key = tool_cache_key("poi_search", {"city": "杭州", "keywords": ["西湖"], "category": None})
    assert key.startswith("poi:杭州:")
