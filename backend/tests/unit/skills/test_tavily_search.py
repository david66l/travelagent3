"""Tests for TavilySearchSkill and UnifiedSearchSkill."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from skills.tavily_search import TavilySearchSkill, UnifiedSearchSkill


class TestTavilySearchSkill:
    """Test Tavily API search."""

    def setup_method(self):
        self.skill = TavilySearchSkill(api_key="test-key")

    def _make_resp(self, data):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=data)
        return resp

    def _make_client(self, resp):
        client = MagicMock()
        client.post = AsyncMock(return_value=resp)
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        return client

    @pytest.mark.asyncio
    async def test_search_success(self):
        resp = self._make_resp({
            "results": [
                {"title": "北京旅游", "url": "http://example.com", "content": "北京很好玩", "score": 0.9},
            ]
        })
        client = self._make_client(resp)

        with patch("httpx.AsyncClient", return_value=client):
            results = await self.skill.search("北京旅游")
            assert len(results) == 1
            assert results[0].title == "北京旅游"

    @pytest.mark.asyncio
    async def test_search_no_api_key(self):
        with patch("skills.tavily_search.settings.tavily_api_key", ""):
            skill = TavilySearchSkill(api_key="")
            results = await skill.search("test")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_error(self):
        client = MagicMock()
        client.post = MagicMock(side_effect=Exception("timeout"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client):
            results = await self.skill.search("test")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_multiple(self):
        resp = self._make_resp({"results": []})
        client = self._make_client(resp)

        with patch("httpx.AsyncClient", return_value=client):
            results = await self.skill.search_multiple(["北京", "上海"])
            assert len(results) == 2
            assert "北京" in results
            assert "上海" in results

    @pytest.mark.asyncio
    async def test_search_with_context(self):
        resp = self._make_resp({
            "results": [{"title": "T", "url": "U", "content": "C", "score": 0.8}],
            "answer": "AI generated answer",
        })
        client = self._make_client(resp)

        with patch("httpx.AsyncClient", return_value=client):
            results, answer = await self.skill.search_with_context("test")
            assert len(results) == 1
            assert answer == "AI generated answer"

    @pytest.mark.asyncio
    async def test_search_with_context_no_key(self):
        with patch("skills.tavily_search.settings.tavily_api_key", ""):
            skill = TavilySearchSkill(api_key="")
            results, answer = await skill.search_with_context("test")
            assert results == []
            assert answer == ""

    @pytest.mark.asyncio
    async def test_long_content_truncated(self):
        resp = self._make_resp({
            "results": [{"title": "T", "url": "U", "content": "x" * 1000, "score": 0.5}],
        })
        client = self._make_client(resp)

        with patch("httpx.AsyncClient", return_value=client):
            results = await self.skill.search("test")
            assert len(results) == 1
            assert len(results[0].snippet) <= 800


class TestUnifiedSearchSkill:
    """Test unified search with fallback."""

    def setup_method(self):
        self.skill = UnifiedSearchSkill()

    @pytest.mark.asyncio
    async def test_tavily_returns_results(self):
        with patch.object(self.skill.tavily, "search", AsyncMock(return_value=["result"])):
            with patch.object(self.skill.duckduckgo, "search", AsyncMock()):
                results = await self.skill.search("test")
                assert results == ["result"]

    @pytest.mark.asyncio
    async def test_fallback_to_duckduckgo(self):
        with patch.object(self.skill.tavily, "search", AsyncMock(return_value=[])):
            with patch.object(self.skill.duckduckgo, "search", AsyncMock(return_value=["fallback"])):
                results = await self.skill.search("test")
                assert results == ["fallback"]

    @pytest.mark.asyncio
    async def test_search_with_context_fallback(self):
        with patch.object(self.skill.tavily, "search_with_context", AsyncMock(return_value=([], ""))):
            with patch.object(self.skill.duckduckgo, "search", AsyncMock(return_value=["fallback"])):
                results, answer = await self.skill.search_with_context("test")
                assert results == ["fallback"]
                assert answer == ""
