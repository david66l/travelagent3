"""Tests for RedisClient loop-binding helpers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.redis_client import RedisClient


@pytest.mark.asyncio
async def test_ensure_client_raises_when_not_connected():
    client = RedisClient()
    with pytest.raises(RuntimeError, match="not connected"):
        client._ensure_client()


@pytest.mark.asyncio
async def test_connect_binds_to_current_loop():
    client = RedisClient()
    mock_redis = MagicMock()
    mock_redis.close = AsyncMock()

    with patch("core.redis_client.aioredis.from_url", new=AsyncMock(return_value=mock_redis)):
        await client.connect()

    assert client._client is mock_redis
    assert client._loop is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_ensure_client_returns_client_when_loop_matches():
    client = RedisClient()
    mock_redis = MagicMock()
    mock_redis.close = AsyncMock()
    client._client = mock_redis
    client._loop = asyncio.get_running_loop()

    assert client._ensure_client() is mock_redis


@pytest.mark.asyncio
async def test_ensure_client_reconnects_on_loop_mismatch():
    client = RedisClient()
    mock_redis = MagicMock()
    mock_redis.close = AsyncMock()
    client._client = mock_redis
    # Pretend the client was created on a different loop
    client._loop = asyncio.new_event_loop()

    with pytest.raises(RuntimeError, match="different event loop"):
        client._ensure_client()

    assert client._client is None
    assert client._loop is None


@pytest.mark.asyncio
async def test_disconnect_closes_client():
    client = RedisClient()
    mock_redis = MagicMock()
    mock_redis.close = AsyncMock()
    client._client = mock_redis
    client._loop = asyncio.get_running_loop()

    await client.disconnect()

    mock_redis.close.assert_awaited_once()
    assert client._client is None
    assert client._loop is None
