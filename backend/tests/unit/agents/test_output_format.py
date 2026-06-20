"""Tests for OutputFormatAgent."""

from unittest.mock import AsyncMock, patch

import pytest

from agents.output_format import OutputFormatAgent


@pytest.fixture
def agent(tmp_path):
    return OutputFormatAgent(base_url="http://test")


@pytest.mark.asyncio
async def test_format_returns_markdown(agent):
    result = await agent.format("# 北京3天\n\n第一天：故宫", [], "北京", "sess1")
    assert "markdown" in result
    assert "output_markdown" in result
    assert result["markdown"]


@pytest.mark.asyncio
async def test_format_with_llm_polish(agent):
    with patch("core.llm_client.llm.chat", new=AsyncMock(return_value="polished markdown")):
        result = await agent.format("original", [], "北京", "sess1")
    assert result["markdown"] == "polished markdown"


@pytest.mark.asyncio
async def test_generate_excel(agent, tmp_path):
    itinerary = [
        {
            "day_number": 1,
            "activities": [
                {
                    "name": "故宫",
                    "start_time": "09:00",
                    "category": "attraction",
                    "duration_minutes": 180,
                    "note": "需预约",
                }
            ],
        }
    ]
    _, url = await agent.generate_excel("# 行程", itinerary, "sess1")
    assert url is not None
    assert "download/excel/" in url


@pytest.mark.asyncio
async def test_generate_map_url_without_amap_key(agent):
    with patch("agents.output_format.settings.amap_key", ""):
        url = agent.generate_map_url([{"activities": [{"name": "故宫"}]}], "北京")
    assert url is None


@pytest.mark.asyncio
async def test_generate_map_url_with_amap_key(agent):
    with patch("agents.output_format.settings.amap_key", "test-key"):
        url = agent.generate_map_url(
            [{"activities": [{"name": "故宫"}, {"name": "长城"}]}], "北京"
        )
    assert url is not None
    assert "restapi.amap.com" in url


def test_safe_filename():
    from agents.output_format import _safe_filename

    name = _safe_filename("pref", "pdf")
    assert name.startswith("pref_")
    assert name.endswith(".pdf")
