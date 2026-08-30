"""Tests for keyless public web-search fallbacks."""

from unittest.mock import AsyncMock, patch

import pytest

from skills.web_search import SearchProvidersUnavailable, WebSearchSkill


def test_parse_so_html_extracts_source_backed_results():
    html = """
    <ul>
      <li class="res-list">
        <h3><a href="https://www.so.com/link?m=one">拙政园开放时间</a></h3>
        <p class="res-desc">开放时间 07:30 至 17:30，以官方公告为准。</p>
      </li>
      <li class="res-list">
        <h3><a href="https://www.so.com/link?m=two">苏州园林官网</a></h3>
      </li>
    </ul>
    """

    results = WebSearchSkill._parse_so_html(html, 5)

    assert [item.title for item in results] == ["拙政园开放时间", "苏州园林官网"]
    assert results[0].url.startswith("https://www.so.com/link")
    assert "07:30" in results[0].snippet


@pytest.mark.asyncio
async def test_chinese_search_uses_mainland_provider_without_ddg_timeouts():
    skill = WebSearchSkill()
    fallback = WebSearchSkill._parse_so_html(
        '<li class="res-list"><h3><a href="https://source.example">结果</a></h3></li>',
        1,
    )
    with (
        patch.object(skill, "_instant_answer", new=AsyncMock(return_value=[])) as instant,
        patch.object(skill, "_html_search", new=AsyncMock(return_value=[])) as ddg_html,
        patch.object(skill, "_so_search", new=AsyncMock(return_value=fallback)) as so_search,
    ):
        results = await skill.search("测试", top_n=1)

    assert results == fallback
    so_search.assert_awaited_once_with("测试", 1)
    instant.assert_not_awaited()
    ddg_html.assert_not_awaited()


@pytest.mark.asyncio
async def test_all_provider_failures_are_not_misreported_as_zero_results():
    skill = WebSearchSkill()
    failure = TimeoutError("provider timeout")
    with (
        patch.object(skill, "_so_search", new=AsyncMock(side_effect=failure)),
        patch.object(skill, "_instant_answer", new=AsyncMock(side_effect=failure)),
        patch.object(skill, "_html_search", new=AsyncMock(side_effect=failure)),
    ):
        with pytest.raises(SearchProvidersUnavailable):
            await skill.search("上海演唱会", top_n=3)


@pytest.mark.asyncio
async def test_successful_empty_provider_response_is_valid_zero_results():
    skill = WebSearchSkill()
    with (
        patch.object(skill, "_so_search", new=AsyncMock(return_value=[])),
        patch.object(skill, "_instant_answer", new=AsyncMock(return_value=[])),
        patch.object(skill, "_html_search", new=AsyncMock(return_value=[])),
    ):
        assert await skill.search("上海演唱会", top_n=3) == []
