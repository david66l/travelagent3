"""Redlock quorum behavior (M3)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.redlock import Redlock


@pytest.mark.asyncio
async def test_acquire_returns_token_on_quorum():
    client = MagicMock()
    client.set = AsyncMock(return_value=True)
    client.close = AsyncMock()
    lock = Redlock(["redis://localhost:6379/2"])

    with patch("core.redlock.aioredis.from_url", new=AsyncMock(return_value=client)):
        token = await lock.acquire("resource:1", 10, blocking=False)
        assert token is not None
        client.set.assert_awaited()


@pytest.mark.asyncio
async def test_acquire_returns_none_when_not_blocking_and_busy():
    client = MagicMock()
    client.set = AsyncMock(return_value=False)
    client.close = AsyncMock()
    lock = Redlock(["redis://localhost:6379/2"])

    with patch("core.redlock.aioredis.from_url", new=AsyncMock(return_value=client)):
        token = await lock.acquire("resource:1", 10, blocking=False)
        assert token is None
