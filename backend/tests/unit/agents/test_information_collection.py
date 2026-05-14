"""Tests for InformationCollectionAgent."""

import pytest
from unittest.mock import AsyncMock
from agents.information_collection import InformationCollectionAgent


class TestInformationCollectionAgent:
    """Test clarifying question generation."""

    def setup_method(self):
        self.agent = InformationCollectionAgent()

    @pytest.mark.asyncio
    async def test_generate_questions_missing_destination(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="想去哪里玩呢？")
        questions = await self.agent.generate_questions(
            missing_required=["destination"],
            missing_recommended=[],
            current_info={},
        )
        assert len(questions) > 0
        assert any("哪里" in q or "目的地" in q for q in questions)

    @pytest.mark.asyncio
    async def test_generate_questions_missing_days(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="计划玩几天？")
        questions = await self.agent.generate_questions(
            missing_required=["travel_days"],
            missing_recommended=[],
            current_info={},
        )
        assert len(questions) > 0
        assert any("几天" in q for q in questions)

    @pytest.mark.asyncio
    async def test_generate_questions_missing_dates(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="什么时候出发？")
        questions = await self.agent.generate_questions(
            missing_required=["travel_dates"],
            missing_recommended=[],
            current_info={},
        )
        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_generate_questions_max_two(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="问题1\n问题2\n问题3")
        questions = await self.agent.generate_questions(
            missing_required=["destination", "travel_days", "travel_dates"],
            missing_recommended=[],
            current_info={},
        )
        assert len(questions) <= 2

    @pytest.mark.asyncio
    async def test_generate_questions_recommended_when_no_required(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="预算大概多少？")
        questions = await self.agent.generate_questions(
            missing_required=[],
            missing_recommended=["budget_range"],
            current_info={"destination": "北京", "travel_days": 3},
        )
        assert len(questions) > 0

    @pytest.mark.asyncio
    async def test_generate_response(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="想了解一下您的出行偏好~")
        response = await self.agent.generate_response(
            missing_required=["destination"],
            missing_recommended=[],
            current_info={},
        )
        assert len(response) > 0
        assert "想" in response or "了解" in response or "目的地" in response

    @pytest.mark.asyncio
    async def test_generate_response_empty_questions(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="")
        response = await self.agent.generate_response(
            missing_required=[],
            missing_recommended=[],
            current_info={"destination": "北京"},
        )
        # When no questions, should still return something informative
        assert isinstance(response, str)
