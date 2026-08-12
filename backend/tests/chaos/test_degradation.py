"""Chaos / degradation tests (M6 §13.4)."""

import pytest
from unittest.mock import AsyncMock

from core.cost_circuit_breaker import is_cost_circuit_active, record_daily_tokens
from core.model_router import select_model
from core.settings import settings


@pytest.mark.chaos
@pytest.mark.asyncio
async def test_cost_circuit_degrades_to_small_model(mock_redis, monkeypatch):
    monkeypatch.setattr(settings, "cost_circuit_breaker_daily_tokens", 10)
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

    monkeypatch.setattr(settings, "local_llm_enabled", True, raising=False)
    await record_daily_tokens(100)
    active = await is_cost_circuit_active()
    model = select_model(role="premium", task_type="planning", cost_circuit_active=active)
    assert active is True
    # Cost circuit forces the free local model regardless of task.
    assert model == settings.local_llm_model


@pytest.mark.chaos
@pytest.mark.asyncio
async def test_model_router_non_intent_uses_cloud(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test", raising=False)
    model = select_model(role="guest", task_type="planning")
    # Non-intent tasks route to the DeepSeek cloud model.
    assert model == settings.llm_model
