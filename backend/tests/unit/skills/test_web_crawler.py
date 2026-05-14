"""Tests for WebCrawlerSkill."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from skills.web_crawler import WebCrawlerSkill, CrawledPage


class TestWebCrawlerSkill:
    """Test web crawler parsing and rate limiting."""

    def setup_method(self):
        self.crawler = WebCrawlerSkill()

    def _make_response(self, status=200, text=""):
        resp = MagicMock()
        resp.status = status
        resp.text = AsyncMock(return_value=text)
        resp.__aenter__ = AsyncMock(return_value=resp)
        resp.__aexit__ = AsyncMock(return_value=False)
        return resp

    def _make_session(self, response):
        session = MagicMock()
        session.get = MagicMock(return_value=response)
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        return session

    @pytest.mark.asyncio
    async def test_crawl_success(self):
        html = "<html><head><title>Test Page</title></head><body><p>Hello World</p><a href='http://example.com/2'>Link</a></body></html>"
        resp = self._make_response(200, html)
        session = self._make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            with patch.object(self.crawler, "_rate_limit", AsyncMock()):
                result = await self.crawler.crawl("http://example.com")
                assert result.title == "Test Page"
                assert "Hello World" in result.text
                assert "http://example.com/2" in result.links

    @pytest.mark.asyncio
    async def test_crawl_non_200(self):
        resp = self._make_response(404, "")
        session = self._make_session(resp)

        with patch("aiohttp.ClientSession", return_value=session):
            with patch.object(self.crawler, "_rate_limit", AsyncMock()):
                result = await self.crawler.crawl("http://example.com")
                assert result.title == ""
                assert result.text == ""

    @pytest.mark.asyncio
    async def test_crawl_exception(self):
        session = MagicMock()
        session.get = MagicMock(side_effect=Exception("timeout"))
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)

        with patch("aiohttp.ClientSession", return_value=session):
            with patch.object(self.crawler, "_rate_limit", AsyncMock()):
                result = await self.crawler.crawl("http://example.com")
                assert result.title == ""

    def test_parse_basic(self):
        html = "<html><head><title>My Page</title></head><body><script>alert(1)</script><p>Content</p><a href='http://x.com'>X</a></body></html>"
        result = self.crawler._parse(html, "http://example.com")
        assert result.title == "My Page"
        assert "Content" in result.text
        assert "alert(1)" not in result.text
        assert "http://x.com" in result.links

    def test_parse_no_title(self):
        html = "<html><body><p>Text</p></body></html>"
        result = self.crawler._parse(html, "http://example.com")
        assert result.title == ""
        assert result.text == "Text"

    def test_parse_truncate(self):
        html = f"<html><title>T</title><body><p>{'x' * 10000}</p></body></html>"
        result = self.crawler._parse(html, "http://example.com")
        assert len(result.text) <= 5000

    def test_parse_only_external_links(self):
        html = "<html><body><a href='/local'>L</a><a href='http://ext.com'>E</a></body></html>"
        result = self.crawler._parse(html, "http://example.com")
        assert "/local" not in result.links
        assert "http://ext.com" in result.links
