"""Tests for ProposalGenerationAgent."""

import pytest
from unittest.mock import AsyncMock
from agents.proposal_generation import ProposalGenerationAgent


class TestProposalGenerationAgent:
    """Test proposal text generation."""

    def setup_method(self):
        self.agent = ProposalGenerationAgent()

    @pytest.mark.asyncio
    async def test_generate_proposal(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="# 北京3日游方案\n\n第一天...")
        planning_json = {
            "trip_profile": {"destination": "北京", "travel_days": 3},
            "days": [
                {"day_number": 1, "theme": "历史文化", "activities": []},
            ],
            "budget_panel": {"total_budget": 5000, "spent": 4500},
        }
        result = await self.agent.generate(planning_json)
        assert len(result) > 0
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_proposal_includes_days(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="包含每天的行程安排")
        planning_json = {
            "trip_profile": {"destination": "成都", "travel_days": 2},
            "days": [
                {"day_number": 1, "theme": "美食探索", "activities": []},
                {"day_number": 2, "theme": "文化体验", "activities": []},
            ],
        }
        result = await self.agent.generate(planning_json)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_generate_proposal_empty_days(self, mock_llm):
        mock_llm.chat = AsyncMock(return_value="暂未安排具体行程")
        planning_json = {
            "trip_profile": {"destination": "上海"},
            "days": [],
        }
        result = await self.agent.generate(planning_json)
        assert len(result) > 0
