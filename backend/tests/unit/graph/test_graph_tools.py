"""Tests for tool_call_node and output_node integration."""

from unittest.mock import AsyncMock, patch

import pytest

from graph.nodes import output_node, tool_call_node


@pytest.mark.asyncio
async def test_tool_call_node_executes_default_calls():
    state = {
        "profile": {"destination": "北京"},
        "itinerary": [
            {
                "day_number": 1,
                "activities": [{"poi_name": "故宫"}],
            }
        ],
    }
    result = await tool_call_node(state)
    assert "tool_results" in result
    assert len(result["tool_results"]) > 0
    assert result["stage"] == "tools_executed"


@pytest.mark.asyncio
async def test_tool_call_node_uses_pending_calls():
    state = {
        "pending_tool_calls": [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "北京"}'},
            }
        ]
    }
    result = await tool_call_node(state)
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["name"] == "get_weather"


@pytest.mark.asyncio
async def test_tool_call_node_empty():
    result = await tool_call_node({})
    assert result["tool_results"] == []


@pytest.mark.asyncio
async def test_output_node_populates_urls():
    state = {
        "itinerary": [
            {
                "day_number": 1,
                "activities": [{"poi_name": "故宫", "start_time": "09:00"}],
            }
        ],
        "profile": {"destination": "北京", "travel_days": 1},
        "messages": [],
        "session_id": "s1",
    }
    with patch("graph.node_impl._output_async", new=AsyncMock(return_value={
        "messages": [{"role": "assistant", "content": "# 北京", "type": "itinerary"}],
        "itinerary": state["itinerary"],
        "stage": "awaiting_booking",
    })):
        result = await output_node(state)

    assert "output_markdown" in result
    assert result["messages"][-1]["content"] == "# 北京"
