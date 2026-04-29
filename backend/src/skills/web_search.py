"""Web Search Skill - DuckDuckGo search (no API key required)."""

import asyncio
from dataclasses import dataclass
from typing import Optional

import httpx


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class WebSearchSkill:
    """DuckDuckGo text search."""

    DUCKDUCKGO_URL = "https://html.duckduckgo.com/html/"
    MAX_RESULTS = 10

    async def search(self, query: str, top_n: int = 10) -> list[SearchResult]:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            try:
                resp = await client.post(
                    self.DUCKDUCKGO_URL,
                    data={"q": query, "kl": "zh-cn"},
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        )
                    },
                )
                resp.raise_for_status()
                return self._parse_results(resp.text, min(top_n, self.MAX_RESULTS))
            except Exception:
                return []

    def _parse_results(self, html: str, top_n: int) -> list[SearchResult]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        results = []
        for item in soup.select(".result")[:top_n]:
            title_tag = item.select_one(".result__title a")
            snippet_tag = item.select_one(".result__snippet")
            if title_tag:
                results.append(
                    SearchResult(
                        title=title_tag.get_text(strip=True),
                        url=title_tag.get("href", ""),
                        snippet=snippet_tag.get_text(strip=True) if snippet_tag else "",
                    )
                )
        return results
