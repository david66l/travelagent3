"""Unit tests for ProfileRecallAgent."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.profile_recall import ProfileRecallAgent
from data.profile_service import ProfileService
from models.travel_slots import TravelSlots


def _fake_async_session(rows: list | None = None):
    """Return an async context manager that yields a mocked db session."""
    db = MagicMock()
    db.fetchrow = AsyncMock(return_value=None)
    db.fetch = AsyncMock(return_value=rows or [])

    @asynccontextmanager
    async def _ctx(*args, **kwargs):
        yield db

    return _ctx, db


@pytest.fixture
def agent():
    return ProfileRecallAgent(profile_service_instance=ProfileService())


@pytest.mark.asyncio
async def test_anonymous_returns_empty(agent):
    slots = TravelSlots(destination="成都", travel_days=3)
    result = await agent.recall(None, slots)
    assert result["source"] == "anonymous"
    assert result["inferred_slots"] == {}
    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_recall_infers_budget_from_avg_daily(agent):
    slots = TravelSlots(destination="成都", travel_days=3)
    service = ProfileService()
    service.get_profile = AsyncMock(return_value={"avg_daily_budget": 600, "pace": "relaxed"})
    ctx, db = _fake_async_session()

    agent_with_mock = ProfileRecallAgent(profile_service_instance=service)
    with patch("agents.profile_recall.async_session_maker", ctx):
        result = await agent_with_mock.recall("user-1", slots)

    assert result["source"] == "long_term"
    assert result["inferred_slots"]["total_budget"] == 1800
    assert result["inferred_slots"]["pace"] == "relaxed"


@pytest.mark.asyncio
async def test_recall_does_not_override_existing_slots(agent):
    slots = TravelSlots(destination="成都", travel_days=3, total_budget=9000, pace="intensive")
    service = ProfileService()
    service.get_profile = AsyncMock(return_value={"avg_daily_budget": 600, "pace": "relaxed"})
    ctx, _ = _fake_async_session()
    agent_with_mock = ProfileRecallAgent(profile_service_instance=service)

    with patch("agents.profile_recall.async_session_maker", ctx):
        result = await agent_with_mock.recall("user-1", slots)

    assert "total_budget" not in result["inferred_slots"]
    assert "pace" not in result["inferred_slots"]


@pytest.mark.asyncio
async def test_short_term_state_used_for_inference(agent):
    slots = TravelSlots(destination="成都")
    state = {
        "profile": {
            "trip": {"travel_days": 4},
            "personal": {"interests": ["美食"], "pace": "relaxed"},
        }
    }
    service = ProfileService()
    service.get_profile = AsyncMock(return_value={})
    ctx, _ = _fake_async_session()
    agent_with_mock = ProfileRecallAgent(profile_service_instance=service)

    with patch("agents.profile_recall.async_session_maker", ctx):
        result = await agent_with_mock.recall("user-1", slots, short_term_state=state)

    assert result["source"] == "short_term"
    assert "美食" in result["inferred_slots"]["interests"]
    assert result["inferred_slots"]["pace"] == "relaxed"


@pytest.mark.asyncio
async def test_service_failure_is_graceful(agent):
    slots = TravelSlots(destination="成都", travel_days=3)
    service = ProfileService()
    service.get_profile = AsyncMock(side_effect=RuntimeError("db down"))
    ctx, _ = _fake_async_session()
    agent_with_mock = ProfileRecallAgent(profile_service_instance=service)

    with patch("agents.profile_recall.async_session_maker", ctx):
        result = await agent_with_mock.recall("user-1", slots)

    assert result["source"] == "long_term"
    assert result["inferred_slots"] == {}


@pytest.mark.asyncio
async def test_redis_short_term_memory_read(agent):
    slots = TravelSlots(destination="成都")
    redis_mock = MagicMock()
    redis_mock.get_json = AsyncMock(
        return_value={"profile": {"personal": {"pace": "relaxed"}}}
    )
    service = ProfileService()
    service.get_profile = AsyncMock(return_value={})
    ctx, _ = _fake_async_session()
    agent_with_mock = ProfileRecallAgent(
        profile_service_instance=service, redis_client_instance=redis_mock
    )

    with patch("agents.profile_recall.async_session_maker", ctx):
        result = await agent_with_mock.recall(
            "user-1", slots, session_id="session-1"
        )

    redis_mock.get_json.assert_awaited_once_with("session:session-1:state")
    assert result["source"] == "short_term"
    assert result["inferred_slots"]["pace"] == "relaxed"


@pytest.mark.asyncio
async def test_similar_users_boost_long_term(agent):
    slots = TravelSlots(destination="北京", travel_days=3)
    service = ProfileService()
    service.get_profile = AsyncMock(return_value={})
    similar_profile = {
        "profile_json": {"pace": "relaxed", "interests": ["历史"]},
        "visited_cities": ["北京"],
        "favorite_spots": [],
        "liked_foods": [],
        "avoided_foods": [],
        "avg_daily_budget": 700,
        "preferred_transport": "public",
        "preferred_accommodation": "hotel",
    }
    ctx, db = _fake_async_session(rows=[similar_profile])
    db.fetchrow = AsyncMock(return_value={"preference_embedding": [0.1] * 1024})
    agent_with_mock = ProfileRecallAgent(profile_service_instance=service)

    with patch("agents.profile_recall.async_session_maker", ctx):
        result = await agent_with_mock.recall("user-1", slots)

    assert result["source"] == "long_term"
    assert result["inferred_slots"]["pace"] == "relaxed"
    assert "历史" in result["inferred_slots"]["interests"]
