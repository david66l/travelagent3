"""Tests for OutputFormatAgent."""

from unittest.mock import AsyncMock, patch

import pytest

from agents.output_format import OutputFormatAgent


@pytest.fixture
def agent(tmp_path):
    return OutputFormatAgent(base_url="http://test")


@pytest.mark.asyncio
async def test_format_returns_markdown(agent):
    with patch("agents.output_format.settings.output_polish_enabled", False):
        result = await agent.format("# 北京3天\n\n第一天：故宫", [], "北京", "sess1")
    assert "markdown" in result
    assert "output_markdown" in result
    assert result["markdown"]


@pytest.mark.asyncio
async def test_format_with_llm_polish(agent):
    with (
        patch("agents.output_format.settings.output_polish_enabled", True),
        patch("core.llm_client.llm.chat", new=AsyncMock(return_value="polished markdown")),
    ):
        result = await agent.format("original", [], "北京", "sess1")
    assert result["markdown"] == "polished markdown"


@pytest.mark.asyncio
async def test_format_skips_polish_when_disabled(agent):
    """With polish disabled, the writer's prose is returned unchanged (no LLM)."""
    with (
        patch("agents.output_format.settings.output_polish_enabled", False),
        patch("core.llm_client.llm.chat", new=AsyncMock()) as mock_chat,
        patch("core.llm_client.llm.stream_chat", new=AsyncMock()) as mock_stream,
    ):
        result = await agent.format("# 北京3天\n\n第一天：故宫", [], "北京", "sess1")
    assert result["markdown"] == "# 北京3天\n\n第一天：故宫"
    mock_chat.assert_not_called()
    mock_stream.assert_not_called()


@pytest.mark.asyncio
async def test_polish_disabled_streams_existing_prose(agent):
    """Disabled polish still streams the precomputed prose via on_token."""
    chunks: list[str] = []

    async def collect(chunk: str) -> None:
        chunks.append(chunk)

    text = "# 标题\n第一行\n第二行\n"
    with (
        patch("agents.output_format.settings.output_polish_enabled", False),
        patch("core.llm_client.llm.stream_chat", new=AsyncMock()) as mock_stream,
    ):
        returned = await agent.polish_markdown(text, on_token=collect)
    assert returned == text
    assert "".join(chunks) == text
    mock_stream.assert_not_called()


@pytest.mark.asyncio
async def test_polish_enabled_streams_real_llm_tokens(agent):
    """With polish enabled, real model tokens stream through on_token in order."""

    async def fake_stream(messages, **kwargs):
        for tok in ["# 北京", "3日", "游\n", "第一天"]:
            yield tok

    chunks: list[str] = []

    async def collect(chunk: str) -> None:
        chunks.append(chunk)

    with (
        patch("agents.output_format.settings.output_polish_enabled", True),
        patch("core.llm_client.llm.stream_chat", new=fake_stream),
    ):
        returned = await agent.polish_markdown("原始行程", on_token=collect)

    assert chunks == ["# 北京", "3日", "游\n", "第一天"]
    assert returned == "# 北京3日游\n第一天"


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
        url = agent.generate_map_url([{"activities": [{"name": "故宫"}, {"name": "长城"}]}], "北京")
    assert url is None


def test_safe_filename():
    from agents.output_format import _safe_filename

    name = _safe_filename("pref", "pdf")
    assert name.startswith("pref_")
    assert name.endswith(".pdf")
