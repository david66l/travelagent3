"""Unit tests for RetrievalFallback."""

from unittest.mock import AsyncMock

import pytest

from data.retrieval_fallback import RetrievalFallback
from models.poi import POI
from models.travel_slots import TravelSlots


def _make_poi(spot_id: str, name: str = "", city: str = "") -> POI:
    return POI(
        spot_id=spot_id,
        spot_name=name or spot_id,
        spot_type="attraction",
        city=city,
        lat=0.0,
        lng=0.0,
    )


@pytest.mark.asyncio
async def test_fallback_returns_structured_when_enough():
    fake_pois = [_make_poi(f"p{i}", city="北京") for i in range(15)]
    repo = AsyncMock()
    repo.search_structured = AsyncMock(return_value=fake_pois)

    fallback = RetrievalFallback(repo=repo)
    slots = TravelSlots(destination="北京", travel_days=3)
    result = await fallback.fallback(slots, min_results=15)

    assert result["retrieval_empty"] is False
    assert len(result["poi_candidates"]) == 15
    assert "relaxed_structured_filters" in result["fallback_reason"]


@pytest.mark.asyncio
async def test_fallback_uses_popular_when_structured_insufficient():
    structured = [_make_poi("p1", city="上海")]
    popular = [_make_poi(f"pop{i}", city="上海") for i in range(15)]
    repo = AsyncMock()
    repo.search_structured = AsyncMock(return_value=structured)
    repo.get_popular_attractions = AsyncMock(return_value=popular)

    fallback = RetrievalFallback(repo=repo)
    slots = TravelSlots(destination="上海", travel_days=3)
    result = await fallback.fallback(slots, min_results=15)

    assert result["retrieval_empty"] is False
    assert len(result["poi_candidates"]) == 16
    assert "popular_city_attractions" in result["fallback_reason"]


@pytest.mark.asyncio
async def test_fallback_returns_safety_set_for_known_city():
    repo = AsyncMock()
    repo.search_structured = AsyncMock(return_value=[])
    repo.get_popular_attractions = AsyncMock(return_value=[])

    fallback = RetrievalFallback(repo=repo)
    slots = TravelSlots(destination="北京", travel_days=3)
    result = await fallback.fallback(slots, min_results=15)

    assert result["retrieval_empty"] is False
    assert len(result["poi_candidates"]) > 0
    assert result["poi_candidates"][0].spot_name == "故宫博物院"


@pytest.mark.asyncio
async def test_fallback_empty_when_no_destination():
    fallback = RetrievalFallback(repo=AsyncMock())
    slots = TravelSlots(travel_days=3)
    result = await fallback.fallback(slots)

    assert result["retrieval_empty"] is True
    assert result["poi_candidates"] == []
