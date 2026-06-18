"""Unit tests for cost circuit breaker."""

from unittest.mock import AsyncMock

import pytest

from core.cost_circuit_breaker import (
    is_cost_circuit_active,
    record_daily_tokens,
    record_external_api_cost,
)


@pytest.mark.asyncio
async def test_cost_circuit_not_active_by_default(mock_redis):
    store: dict[str, str] = {}

    async def _get(key):
        return store.get(key)

    async def _incrby(key, amount):
        value = int(store.get(key, "0")) + amount
        store[key] = str(value)
        return value

    mock_redis.get = AsyncMock(side_effect=_get)
    mock_redis.incrby = AsyncMock(side_effect=_incrby)
    mock_redis.expire = AsyncMock()

    assert await is_cost_circuit_active() is False


@pytest.mark.asyncio
async def test_cost_circuit_trips_on_daily_tokens(mock_redis, monkeypatch):
    from core.settings import settings

    monkeypatch.setattr(settings, "cost_circuit_breaker_daily_tokens", 100)
    store: dict[str, str] = {}

    async def _get(key):
        return store.get(key)

    async def _incrby(key, amount):
        value = int(store.get(key, "0")) + amount
        store[key] = str(value)
        return value

    mock_redis.get = AsyncMock(side_effect=_get)
    mock_redis.incrby = AsyncMock(side_effect=_incrby)
    mock_redis.expire = AsyncMock()

    await record_daily_tokens(150)
    assert await is_cost_circuit_active() is True


@pytest.mark.asyncio
async def test_cost_circuit_trips_on_api_cost(mock_redis, monkeypatch):
    from core.settings import settings

    monkeypatch.setattr(settings, "cost_circuit_breaker_daily_api_cost_cny", 1.0)
    store: dict[str, str] = {}

    async def _get(key):
        return store.get(key)

    async def _incrby(key, amount):
        value = int(store.get(key, "0")) + amount
        store[key] = str(value)
        return value

    mock_redis.get = AsyncMock(side_effect=_get)
    mock_redis.incrby = AsyncMock(side_effect=_incrby)
    mock_redis.expire = AsyncMock()

    await record_external_api_cost(2.0)
    assert await is_cost_circuit_active() is True
