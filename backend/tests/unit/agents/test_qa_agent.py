"""Tests for QAAgent."""

import pytest
from unittest.mock import AsyncMock, patch
from agents.qa_agent import QAAgent


class TestQAAgent:
    """Test Q&A Agent."""

    def setup_method(self):
        self.agent = QAAgent()

    @pytest.mark.asyncio
    async def test_answer_with_city(self, mock_llm):
        mock_search = AsyncMock(return_value=[])
        with patch.object(self.agent.search_skill, "search", mock_search):
            mock_llm.chat = AsyncMock(return_value="成都有很多好吃的，比如火锅、串串、担担面...")
            answer = await self.agent.answer("成都什么好吃？", city="成都")
            assert len(answer) > 0
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_answer_without_city(self, mock_llm):
        mock_search = AsyncMock(return_value=[])
        with patch.object(self.agent.search_skill, "search", mock_search):
            mock_llm.chat = AsyncMock(return_value="中国有很多美食...")
            answer = await self.agent.answer("中国有哪些美食？")
            assert len(answer) > 0
            mock_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_answer_uses_search_results(self, mock_llm):
        from skills.web_search import SearchResult
        mock_search = AsyncMock(return_value=[
            SearchResult(title="成都美食", snippet="火锅串串", url="http://example.com"),
        ])
        with patch.object(self.agent.search_skill, "search", mock_search):
            mock_llm.chat = AsyncMock(return_value="成都火锅很有名")
            answer = await self.agent.answer("成都美食", city="成都")
            assert "火锅" in answer or "成都" in answer
