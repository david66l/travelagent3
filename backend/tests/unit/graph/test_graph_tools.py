"""Tests for tool_call_node and output_node integration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from graph.nodes import _agent_clarification_message, output_node, tool_call_node


def test_event_tradeoff_hides_internal_verifier_language():
    ledger = SimpleNamespace(
        failures=[
            SimpleNamespace(
                message="EVENT_FIELDS_INCOMPLETE:start_time, EVENT_VENUE_UNGROUNDED:lat,lng"
            )
        ],
        goal=SimpleNamespace(
            hard_constraints={
                "event_query": "某演唱会",
                "start_date": "2026-10-01",
            }
        ),
    )
    artifact = SimpleNamespace(
        payload={
            "reason": "finalize_research verifier failed",
            "options": ["retry verifier"],
        }
    )

    question, options = _agent_clarification_message(ledger, artifact)

    assert "2026-10-01" in question
    assert "官方活动页" in question
    assert "verifier" not in question
    assert len(options) == 3


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
    with patch(
        "graph.node_impl._output_async",
        new=AsyncMock(
            return_value={
                "messages": [{"role": "assistant", "content": "# 北京", "type": "itinerary"}],
                "itinerary": state["itinerary"],
                "stage": "awaiting_booking",
            }
        ),
    ):
        with patch(
            "agents.output_format.output_format_agent.stream_existing_markdown",
            new=AsyncMock(return_value="# 北京"),
        ) as stream_existing:
            with patch(
                "agents.output_format.output_format_agent.build_artifacts",
                new=AsyncMock(return_value={"pdf": None, "excel": None, "map": None}),
            ):
                result = await output_node(state)

    assert "output_markdown" in result
    assert result["messages"][-1]["content"] == "# 北京"
    stream_existing.assert_awaited_once()
