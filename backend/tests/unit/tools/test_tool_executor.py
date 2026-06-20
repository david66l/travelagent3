"""Tests for ToolExecutor."""

from unittest.mock import AsyncMock, patch

import pytest

from schemas import ToolResult
from tools.tool_executor import ToolExecutor


@pytest.fixture
def executor():
    return ToolExecutor()


@pytest.mark.asyncio
async def test_available_tools_count(executor):
    assert len(executor.available_tools) == 11


@pytest.mark.asyncio
async def test_execute_unknown_tool(executor):
    results = await executor.execute(
        [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "unknown_tool", "arguments": "{}"},
            }
        ]
    )
    assert len(results) == 1
    assert results[0]["name"] == "unknown_tool"
    assert results[0]["result"]["is_fallback"] is True
    assert "unknown tool" in results[0]["result"]["fallback_reason"]


@pytest.mark.asyncio
async def test_execute_invalid_arguments(executor):
    results = await executor.execute(
        [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "not-json"},
            }
        ]
    )
    assert len(results) == 1
    assert results[0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_get_weather_handler(executor):
    with patch.object(
        executor._weather,
        "query",
        new=AsyncMock(return_value=[]),
    ):
        result = await executor._handle_get_weather({"city": "北京"})
    assert isinstance(result, ToolResult)


@pytest.mark.asyncio
async def test_check_reservation_handler(executor):
    result = await executor._handle_check_reservation({"poi_name": "故宫"})
    assert result.data["need_reserve"] is True


@pytest.mark.asyncio
async def test_get_route_handler(executor):
    result = await executor._handle_get_route(
        {"origin": "酒店", "destination": "故宫", "mode": "taxi"}
    )
    assert result.data["minutes"] > 0
    assert result.is_fallback is True


@pytest.mark.asyncio
async def test_get_poi_detail_fallback(executor):
    with patch.object(
        executor._poi,
        "search_pois",
        new=AsyncMock(return_value=[]),
    ):
        result = await executor._handle_get_poi_detail({"poi_name": "未知景点"})
    assert result.is_fallback is True


@pytest.mark.asyncio
async def test_update_user_profile_handler(executor):
    result = await executor._handle_update_user_profile(
        {"key": "budget_per_day", "value": 500}
    )
    assert result.data["updated"]["budget_per_day"] == 500


@pytest.mark.asyncio
async def test_handler_exception_is_isolated(executor):
    async def failing_handler(args):
        raise RuntimeError("boom")

    executor._handlers["get_weather"] = failing_handler
    results = await executor.execute(
        [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "北京"}'},
            }
        ]
    )
    assert results[0]["result"]["is_fallback"] is True
    assert "boom" in results[0]["result"]["fallback_reason"]
