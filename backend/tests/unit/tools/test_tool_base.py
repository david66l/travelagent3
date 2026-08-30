"""Tests for the unified Tool base class."""

import asyncio
import pytest
from unittest.mock import AsyncMock

from schemas import ToolResult
from core.local_cache import tool_local_cache
from tools.base import Tool


@pytest.fixture(autouse=True)
async def _clear_tool_l1_cache():
    await tool_local_cache.clear()
    yield
    await tool_local_cache.clear()


class DummyTool(Tool):
    name = "dummy"
    timeout = 1.0
    retries = 2
    cache_ttl = 60

    def __init__(self, execute_result=None, fallback_result=None):
        self._execute_result = execute_result
        self._fallback_result = fallback_result

    async def execute(self, params: dict):
        if isinstance(self._execute_result, Exception):
            raise self._execute_result
        return self._execute_result

    async def _fallback(self, params: dict, last_error):
        if self._fallback_result is not None:
            return self._fallback_result
        return await super()._fallback(params, last_error)


@pytest.mark.asyncio
async def test_tool_cache_hit(mock_redis):
    cached = {"data": {"value": 42}, "data_source": "api", "retries": 0}
    mock_redis.get_json = AsyncMock(return_value=cached)
    tool = DummyTool(execute_result={"value": 99})
    result = await tool.run({"q": "beijing"})
    assert result.data == {"value": 42}
    assert result.data_source == "api"
    mock_redis.set_json.assert_not_called()


@pytest.mark.asyncio
async def test_tool_execute_success(mock_redis):
    mock_redis.get_json = AsyncMock(return_value=None)
    mock_redis.set_json = AsyncMock()
    tool = DummyTool(execute_result={"value": 42})
    result = await tool.run({"q": "beijing"})
    assert result.data == {"value": 42}
    assert result.data_source == "api"
    assert result.retries == 0
    mock_redis.set_json.assert_called_once()


@pytest.mark.asyncio
async def test_tool_executes_when_cache_lock_backend_is_unavailable(mock_redis, monkeypatch):
    monkeypatch.setattr(
        "tools.base.redis_client.lock",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("redis unavailable")),
    )
    tool = DummyTool(execute_result={"value": 42})

    result = await tool.run({"q": "beijing"})

    assert result.data == {"value": 42}
    assert result.data_source == "api"


@pytest.mark.asyncio
async def test_tool_retry_then_success(mock_redis):
    mock_redis.get_json = AsyncMock(return_value=None)
    mock_redis.set_json = AsyncMock()
    tool = DummyTool()
    attempts = [RuntimeError("boom"), {"value": 1}]

    async def side_effect(params):
        item = attempts.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    tool.execute = side_effect
    result = await tool.run({"q": "beijing"})
    assert result.data == {"value": 1}
    assert result.retries == 1


@pytest.mark.asyncio
async def test_tool_retries_exhausted_fallback(mock_redis):
    mock_redis.get_json = AsyncMock(return_value=None)
    mock_redis.set_json = AsyncMock()
    fallback = ToolResult(data=[], data_source="fallback", is_fallback=True, fallback_reason="down")
    tool = DummyTool(execute_result=RuntimeError("boom"), fallback_result=fallback)
    result = await tool.run({"q": "beijing"})
    assert result.data_source == "fallback"
    assert result.is_fallback is True
    assert result.retries == 2


@pytest.mark.asyncio
async def test_tool_timeout_fallback(mock_redis):
    mock_redis.get_json = AsyncMock(return_value=None)
    mock_redis.set_json = AsyncMock()
    fallback = ToolResult(data=None, data_source="unavailable")
    tool = DummyTool(execute_result=asyncio.TimeoutError("timeout"), fallback_result=fallback)
    tool.timeout = 0.1
    result = await tool.run({"q": "beijing"})
    assert result.data_source == "unavailable"


@pytest.mark.asyncio
async def test_tool_execute_returns_tool_result(mock_redis):
    mock_redis.get_json = AsyncMock(return_value=None)
    mock_redis.set_json = AsyncMock()
    inner = ToolResult(data=[1, 2], data_source="built_in", is_fallback=True)
    tool = DummyTool(execute_result=inner)
    result = await tool.run({"q": "beijing"})
    assert result.data == [1, 2]
    assert result.data_source == "built_in"
    assert result.is_fallback is True
