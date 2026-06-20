"""Tests for the gathering LangGraph subgraph."""

from unittest.mock import AsyncMock, patch

import pytest

from graph.gathering import build_gathering_subgraph, gathering_turn_impl
from schemas import IntentResult


@pytest.mark.asyncio
async def test_gathering_turn_impl_clarify_when_profile_incomplete():
    intent = IntentResult(
        intent="generate_itinerary",
        confidence=0.9,
        sentiment="neutral",
        user_entities={},
        slots={},
        missing_required=["destination"],
        clarification_questions=["您需要把这个信息告诉我：目的地。"],
    )
    conv_after = {
        "profile": {"trip": {"origin": "济南"}},
        "turn": 1,
        "recent_messages": [{"role": "user", "content": "5000", "ts": 1}],
    }

    async def _turn(state, content):
        state.update(conv_after)
        return intent

    with patch("graph.gathering.process_user_turn", new=AsyncMock(side_effect=_turn)):
        with patch("graph.gathering.is_profile_ready", return_value=False):
            result = await gathering_turn_impl(
                {"user_input": "5000", "_conversation_state": {"profile": {}}}
            )

    assert result["next_action"] == "clarify"
    assert result["stage"] == "gathering"
    assert result["conversation_sync"]["phase"] == "gathering"
    assert result["clarification_questions"] == ["您需要把这个信息告诉我：目的地。"]


@pytest.mark.asyncio
async def test_gathering_turn_impl_plan_when_profile_ready():
    intent = IntentResult(
        intent="generate_itinerary",
        confidence=0.9,
        sentiment="neutral",
        user_entities={"destination": "成都"},
        slots={"destination": "成都"},
        missing_required=[],
        clarification_questions=[],
    )

    with patch("graph.gathering.process_user_turn", new=AsyncMock(return_value=intent)):
        with patch("graph.gathering.is_profile_ready", return_value=True):
            result = await gathering_turn_impl(
                {
                    "user_input": "去成都4天",
                    "profile": {},
                    "_conversation_state": {"phase": "gathering"},
                }
            )

    assert result["next_action"] == "plan"
    assert result["stage"] == "demand_parsed"
    assert result["conversation_sync"]["phase"] == "planning"
    assert result["intent_ready_message"] == "意图识别已完成，接下来将进行大致的规划。"


@pytest.mark.asyncio
async def test_gathering_turn_impl_skips_notice_when_already_planning():
    intent = IntentResult(
        intent="modify_itinerary",
        confidence=0.9,
        sentiment="neutral",
        user_entities={"destination": "成都"},
        slots={"destination": "成都"},
        missing_required=[],
        clarification_questions=[],
    )

    with patch("graph.gathering.process_user_turn", new=AsyncMock(return_value=intent)):
        with patch("graph.gathering.is_profile_ready", return_value=True):
            result = await gathering_turn_impl(
                {
                    "user_input": "第三天换个景点",
                    "_conversation_state": {"phase": "planning"},
                }
            )

    assert result["next_action"] == "plan"
    assert "intent_ready_message" not in result


def test_gathering_subgraph_compiles():
    graph = build_gathering_subgraph()
    assert graph is not None
