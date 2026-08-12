"""Web Search Skill — DDG Instant Answer API + HTML fallback.

Primary:   DuckDuckGo Instant Answer API (api.duckduckgo.com)
           Free, no API key, returns structured JSON with Abstract/RelatedTopics.
Secondary: DuckDuckGo HTML search (html.duckduckgo.com)
           Parses the SERP HTML when Instant Answer returns no web results.

DDG Instant Answer docs: https://duckduckgo.com/api
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0


class WebSearchSkill:
    """Free web search via DuckDuckGo, no API key needed.

    Strategy:
    1. Instant Answer API (structured JSON, zero parsing)
    2. HTML SERP fallback (when Instant Answer has no web results)
    """

    INSTANT_ANSWER_URL = "https://api.duckduckgo.com/"
    HTML_SEARCH_URL = "https://html.duckduckgo.com/html/"
    MAX_RESULTS = 10

    def __init__(self, user_agent: Optional[str] = None):
        self._ua = user_agent or (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def search(self, query: str, top_n: int = 10) -> list[SearchResult]:
        """Search via DDG Instant Answer → HTML fallback."""
        top_n = min(top_n, self.MAX_RESULTS)

        # 1. Instant Answer API
        try:
            results = await self._instant_answer(query, top_n)
            if results:
                logger.debug("DDG Instant Answer: %d results for '%s'", len(results), query[:60])
                return results
        except Exception:
            logger.debug("DDG Instant Answer failed, trying HTML fallback", exc_info=True)

        # 2. HTML SERP fallback
        try:
            results = await self._html_search(query, top_n)
            if results:
                logger.debug("DDG HTML: %d results for '%s'", len(results), query[:60])
                return results
        except Exception:
            logger.debug("DDG HTML search failed", exc_info=True)

        return []

    # ------------------------------------------------------------------ #
    # Instant Answer API
    # ------------------------------------------------------------------ #

    async def _instant_answer(self, query: str, top_n: int) -> list[SearchResult]:
        """Fetch structured results from DDG Instant Answer API.

        Returns: AbstractText + RelatedTopics as SearchResult list.
        """
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "no_redirect": "1",
            "skip_disambig": "1",
            "kl": "cn-zh",  # Chinese locale
        }
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            resp = await client.get(self.INSTANT_ANSWER_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        results: list[SearchResult] = []

        # Abstract (Wikipedia-style summary)
        abstract = data.get("AbstractText", "").strip()
        abstract_url = data.get("AbstractURL", "")
        if abstract:
            results.append(
                SearchResult(
                    title=data.get("Heading", query),
                    url=abstract_url,
                    snippet=abstract[:500],
                    score=0.9,
                )
            )

        # Direct answer (calculator, facts, etc.)
        answer = data.get("Answer", "").strip()
        if answer and not abstract:
            results.append(
                SearchResult(
                    title="即时回答",
                    url="",
                    snippet=answer[:500],
                    score=0.85,
                )
            )

        # Related topics
        for topic in data.get("RelatedTopics", []):
            if isinstance(topic, dict) and topic.get("Text"):
                url = topic.get("FirstURL", "")
                snippet = topic.get("Text", "")
                if snippet:
                    results.append(
                        SearchResult(
                            title=snippet[:80],
                            url=url,
                            snippet=snippet[:500],
                            score=0.7,
                        )
                    )

        return results[:top_n]

    # ------------------------------------------------------------------ #
    # HTML SERP fallback
    # ------------------------------------------------------------------ #

    async def _html_search(self, query: str, top_n: int) -> list[SearchResult]:
        """Fallback: scrape DDG HTML search results page."""
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.post(
                self.HTML_SEARCH_URL,
                data={"q": query, "kl": "cn-zh"},
                headers={"User-Agent": self._ua},
            )
            resp.raise_for_status()
            return self._parse_html(resp.text, top_n)

    @staticmethod
    def _parse_html(html: str, top_n: int) -> list[SearchResult]:
        """Parse DDG HTML SERP into SearchResult objects."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return []

        soup = BeautifulSoup(html, "html.parser")
        results: list[SearchResult] = []
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
