"""Unit tests for embedding_sync script."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.embedding_sync import sync_attraction_embeddings


def _make_mock_session(rows):
    """Return async context manager yielding a mocked AsyncSession."""
    result = MagicMock()
    result.mappings.return_value.all.return_value = rows

    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()

    @asynccontextmanager
    async def _ctx(*args, **kwargs):
        yield db

    return _ctx, db


@pytest.mark.asyncio
async def test_sync_skips_empty_descriptions_and_updates_others():
    rows = [
        {"id": "uuid-1", "name": "故宫", "city": "北京", "description": "明清皇宫"},
        {"id": "uuid-2", "name": "空描述", "city": "北京", "description": ""},
    ]
    ctx, db = _make_mock_session(rows)
    embedder = MagicMock()
    embedder.encode_single = MagicMock(return_value=[0.1] * 1024)

    with patch("scripts.embedding_sync.async_session_maker", ctx):
        with patch("scripts.embedding_sync.get_embedder", new=AsyncMock(return_value=embedder)):
            stats = await sync_attraction_embeddings(batch_size=10)

    assert stats["updated"] == 1
    assert stats["skipped"] == 1
    assert db.execute.call_count == 2  # SELECT + UPDATE (single batch, then break)
