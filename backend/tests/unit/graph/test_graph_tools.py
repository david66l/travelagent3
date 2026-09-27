"""Tests for tool_call_node and output_node integration."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from graph.nodes import _agent_clarification_message, output_node


def test_clarification_renderer_preserves_model_question_despite_prior_failure():
    ledger = SimpleNamespace(failures=[SimpleNamespace(message="EVENT_FIELDS_INCOMPLETE")])
    artifact = SimpleNamespace(
        payload={"question": "请提供演出官方链接。", "options": ["稍后提供"]}
    )
    assert _agent_clarification_message(ledger, artifact) == ("请提供演出官方链接。", ["稍后提供"])


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
