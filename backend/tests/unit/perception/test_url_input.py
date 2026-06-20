"""Tests for URL input parser."""

from unittest.mock import AsyncMock

import httpx
import pytest

from perception.url_input import UrlInputParser


def _make_response(text: str, content_type: str = "text/html; charset=utf-8") -> httpx.Response:
    return httpx.Response(
        200,
        text=text,
        headers={"content-type": content_type},
    )


@pytest.mark.asyncio
async def test_parse_url_extracts_text_and_title():
    html = """
    <html>
      <head><title>Example Page</title></head>
      <body>
        <script>alert('ignore')</script>
        <p>Hello world</p>
        <p>Second paragraph</p>
      </body>
    </html>
    """
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: _make_response(html)
    ))
    parser = UrlInputParser(client=client)
    result = await parser.parse("https://example.com/page")

    assert result["type"] == "url"
    assert result["extracted_text"] == "Hello world Second paragraph"
    assert result["metadata"]["title"] == "Example Page"
    assert result["metadata"]["status_code"] == 200


@pytest.mark.asyncio
async def test_parse_url_rejects_non_html():
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: _make_response("{}", "application/json")
    ))
    parser = UrlInputParser(client=client)
    result = await parser.parse("https://example.com/api")

    assert result["extracted_text"] is None
    assert "Unsupported content type" in result["metadata"]["error"]


@pytest.mark.asyncio
async def test_parse_url_rejects_invalid_scheme():
    parser = UrlInputParser()
    result = await parser.parse("ftp://example.com/page")
    assert result["extracted_text"] is None
    assert "Invalid URL scheme" in result["metadata"]["error"]


@pytest.mark.asyncio
async def test_parse_url_graceful_on_http_error():
    client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(500)
    ))
    parser = UrlInputParser(client=client)
    result = await parser.parse("https://example.com/page")

    assert result["extracted_text"] is None
    assert "error" in result["metadata"]
