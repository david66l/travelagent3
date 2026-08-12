"""Unit tests for RetrievalRepository."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from data.retrieval_repository import RetrievalRepository


def _make_mock_session(rows):
    """Return async context manager yielding a mocked AsyncSession."""
    result = MagicMock()
    result.mappings.return_value.all.return_value = rows

    db = MagicMock()
    db.execute = AsyncMock(return_value=result)

    @asynccontextmanager
    async def _ctx(*args, **kwargs):
        yield db

    return _ctx, db


@pytest.mark.asyncio
async def test_search_structured_builds_filter_query():
    rows = [
        {
            "id": "uuid-1",
            "name": "故宫",
            "city": "北京",
            "lat": 39.9,
            "lng": 116.4,
            "duration_minutes": 180,
            "walk_intensity": 3,
            "tags": ["历史"],
            "suitable_for": ["family_elder"],
        }
    ]
    ctx, db = _make_mock_session(rows)
    repo = RetrievalRepository()

    with patch("data.retrieval_repository.async_session_maker", ctx):
        pois = await repo.search_structured(
            "北京",
            max_walk_intensity=3,
            wheelchair_only=True,
            avoid_tags=["爬山"],
            suitable_for=["family_elder"],
            limit=10,
        )

    assert len(pois) == 1
    assert pois[0].spot_name == "故宫"
    assert ":city" in db.execute.call_args[0][0].text
    assert "wheelchair_accessible = true" in db.execute.call_args[0][0].text


@pytest.mark.asyncio
async def test_search_vector_calls_embedding_and_db():
    rows = [
        {
            "id": "uuid-2",
            "name": "天坛",
            "city": "北京",
            "lat": 39.88,
            "lng": 116.41,
            "duration_minutes": 120,
            "walk_intensity": 2,
            "tags": ["历史"],
            "suitable_for": ["family_elder"],
        }
    ]
    ctx, db = _make_mock_session(rows)
    embedder = MagicMock()
    embedder.aencode_single = AsyncMock(return_value=[0.1] * 1024)

    repo = RetrievalRepository()
    with patch("data.retrieval_repository.async_session_maker", ctx):
        with patch("data.retrieval_repository.get_embedder", new=AsyncMock(return_value=embedder)):
            pois = await repo.search_vector("北京 历史", city="北京", limit=5)

    assert len(pois) == 1
    assert pois[0].spot_name == "天坛"
    assert pois[0].source == "vector"
    embedder.aencode_single.assert_awaited_once_with("北京 历史")


@pytest.mark.asyncio
async def test_search_bm25_uses_portable_chinese_lexical_match():
    rows = [
        {
            "id": "uuid-3",
            "name": "颐和园",
            "city": "北京",
            "lat": 40.0,
            "lng": 116.27,
            "duration_minutes": 240,
            "walk_intensity": 3,
            "tags": ["园林"],
            "suitable_for": ["family_elder"],
        }
    ]
    ctx, db = _make_mock_session(rows)
    repo = RetrievalRepository()

    with patch("data.retrieval_repository.async_session_maker", ctx):
        pois = await repo.search_bm25("颐和园", city="北京", limit=5)

    assert len(pois) == 1
    assert pois[0].spot_name == "颐和园"
    assert pois[0].source == "bm25"
    query = db.execute.call_args[0][0].text
    assert "regexp_split_to_array" in query
    assert "ILIKE" in query
    assert "plainto_tsquery" not in query
