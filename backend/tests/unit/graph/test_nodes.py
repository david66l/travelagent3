"""Tests for LangGraph node functions."""

import pytest
from unittest.mock import AsyncMock, patch
from graph.nodes import (
    intent_node,
    confirm_node,
    ask_modification_node,
    format_output_node,
    qa_node,
)
from core.state import ItineraryState


class TestIntentNode:
    """Test intent recognition node."""

    @pytest.mark.asyncio
    async def test_intent_node(self, mock_llm):
        from schemas import IntentResult
        mock_llm.structured_call = AsyncMock(
            return_value=IntentResult(
                intent="generate_itinerary",
                confidence=0.95,
                user_entities={"destination": "北京", "travel_days": 3},
                missing_required=[],
            )
        )
        state = ItineraryState(
            session_id="test",
            user_id="user",
            user_input="我想去北京玩3天",
            messages=[],
        )
        result = await intent_node(state)
        assert result["intent"] == "generate_itinerary"
        assert result["needs_clarification"] is False

    @pytest.mark.asyncio
    async def test_intent_node_needs_clarification(self, mock_llm):
        from schemas import IntentResult
        mock_llm.structured_call = AsyncMock(
            return_value=IntentResult(
                intent="generate_itinerary",
                confidence=0.95,
                user_entities={},
                missing_required=["destination"],
            )
        )
        state = ItineraryState(
            session_id="test",
            user_id="user",
            user_input="我想去旅游",
            messages=[],
        )
        result = await intent_node(state)
        assert result["needs_clarification"] is True


class TestConfirmNode:
    """Test confirm node."""

    @pytest.mark.asyncio
    async def test_confirm_node(self):
        state = ItineraryState(
            session_id="test",
            user_id="user",
            user_input="确认行程",
            messages=[],
        )
        result = await confirm_node(state)
        assert result["itinerary_status"] == "confirmed"
        assert result["waiting_for_confirmation"] is False


class TestAskModificationNode:
    """Test ask modification node."""

    @pytest.mark.asyncio
    async def test_with_itinerary(self):
        state = ItineraryState(
            session_id="test",
            user_id="user",
            user_input="继续修改",
            messages=[],
            current_itinerary=[{"day_number": 1, "activities": []}],
        )
        result = await ask_modification_node(state)
        assert "调整" in result["assistant_response"]
        assert result["waiting_for_confirmation"] is False

    @pytest.mark.asyncio
    async def test_without_itinerary(self):
        state = ItineraryState(
            session_id="test",
            user_id="user",
            user_input="修改",
            messages=[],
            current_itinerary=[],
        )
        result = await ask_modification_node(state)
        assert "需求" in result["assistant_response"]


class TestFormatOutputNode:
    """Test format output node."""

    @pytest.mark.asyncio
    async def test_format_with_response(self):
        state = ItineraryState(
            session_id="test",
            user_id="user",
            user_input="你好",
            messages=[],
            assistant_response="你好！",
        )
        result = await format_output_node(state)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_format_empty_response(self):
        state = ItineraryState(
            session_id="test",
            user_id="user",
            user_input="???",
            messages=[],
        )
        result = await format_output_node(state)
        assert len(result["messages"]) == 1
        assert "抱歉" in result["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_format_truncate_messages(self):
        state = ItineraryState(
            session_id="test",
            user_id="user",
            user_input="test",
            messages=[{"role": "user", "content": str(i)} for i in range(25)],
            assistant_response="reply",
        )
        result = await format_output_node(state)
        assert len(result["messages"]) <= 21  # truncated to 20 + 1 assistant


class TestQANode:
    """Test Q&A node."""

    @pytest.mark.asyncio
    async def test_qa_node(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="北京烤鸭很好吃")
        state = ItineraryState(
            session_id="test",
            user_id="user",
            user_input="北京有什么好吃的？",
            messages=[],
            user_profile={"destination": "北京"},
        )
        with patch("agents.qa_agent.WebSearchSkill.search", AsyncMock(return_value=[])):
            result = await qa_node(state)
            assert "北京烤鸭" in result["assistant_response"]
