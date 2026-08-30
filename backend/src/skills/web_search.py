"""Keyless web-search skill with multiple public fallbacks.

Chinese queries: 360 Search HTML first, avoiding blocked-provider timeouts.
Other queries:   DuckDuckGo Instant Answer, then DuckDuckGo HTML, then 360.

DDG Instant Answer docs: https://duckduckgo.com/api
"""

from __future__ import annotations

import logging
import re
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


class SearchProvidersUnavailable(ConnectionError):
    """Every configured web-search provider failed before returning a response."""


class WebSearchSkill:
    """Free web search via DuckDuckGo, no API key needed.

    Strategy:
    1. Instant Answer API (structured JSON, zero parsing)
    2. HTML SERP fallback (when Instant Answer has no web results)
    """

    INSTANT_ANSWER_URL = "https://api.duckduckgo.com/"
    HTML_SEARCH_URL = "https://html.duckduckgo.com/html/"
    SO_SEARCH_URL = "https://www.so.com/s"
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
        """Use a locale-aware provider order with bounded fallbacks."""
        top_n = min(top_n, self.MAX_RESULTS)
        provider_responded = False
        failures: list[BaseException] = []

        # Chinese travel queries are both better served and much faster through
        # a mainland endpoint.  In deployments where DDG is blocked, trying it
        # first adds two network timeouts to every Agent tool call.
        chinese_query = bool(re.search(r"[\u4e00-\u9fff]", query))
        if chinese_query:
            try:
                results = await self._so_search(query, top_n)
                provider_responded = True
                if results:
                    logger.debug("360 Search HTML: %d results for '%s'", len(results), query[:60])
                    return results
            except Exception as exc:
                failures.append(exc)
                logger.debug("360 Search HTML failed", exc_info=True)

        # 1. Instant Answer API
        try:
            results = await self._instant_answer(query, top_n)
            provider_responded = True
            if results:
                logger.debug("DDG Instant Answer: %d results for '%s'", len(results), query[:60])
                return results
        except Exception as exc:
            failures.append(exc)
            logger.debug("DDG Instant Answer failed, trying HTML fallback", exc_info=True)

        # 2. HTML SERP fallback
        try:
            results = await self._html_search(query, top_n)
            provider_responded = True
            if results:
                logger.debug("DDG HTML: %d results for '%s'", len(results), query[:60])
                return results
        except Exception as exc:
            failures.append(exc)
            logger.debug("DDG HTML search failed", exc_info=True)

        # 3. 360 Search is reachable from common mainland deployment regions
        # where DuckDuckGo may be blocked.  Keeping it behind DDG avoids tying
        # non-Chinese traffic to one region-specific provider.
        if not chinese_query:
            try:
                results = await self._so_search(query, top_n)
                provider_responded = True
                if results:
                    logger.debug("360 Search HTML: %d results for '%s'", len(results), query[:60])
                    return results
            except Exception as exc:
                failures.append(exc)
                logger.debug("360 Search HTML failed", exc_info=True)

        if not provider_responded and failures:
            raise SearchProvidersUnavailable(
                f"all web-search providers failed ({len(failures)} attempts)"
            ) from failures[-1]
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
        async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
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
        async with httpx.AsyncClient(timeout=7.0, follow_redirects=True) as client:
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

    async def _so_search(self, query: str, top_n: int) -> list[SearchResult]:
        """Fallback for mainland deployments using 360 Search's public SERP."""
        async with httpx.AsyncClient(timeout=7.0, follow_redirects=True) as client:
            resp = await client.get(
                self.SO_SEARCH_URL,
                params={"q": query},
                headers={"User-Agent": self._ua},
            )
            resp.raise_for_status()
            return self._parse_so_html(resp.text, top_n)

    @staticmethod
    def _parse_so_html(html: str, top_n: int) -> list[SearchResult]:
        """Parse organic 360 results and discard empty/challenge entries."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return []

        soup = BeautifulSoup(html, "html.parser")
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for item in soup.select("li.res-list"):
            title_tag = item.select_one("h3 a")
            if title_tag is None:
                continue
            url = str(title_tag.get("href") or "").strip()
            title = title_tag.get_text(" ", strip=True)
            if not url or not title or url in seen_urls:
                continue
            snippet_tag = (
                item.select_one(".res-desc")
                or item.select_one(".summary")
                or item.select_one(".res-rich")
                or item.select_one("p")
            )
            snippet = snippet_tag.get_text(" ", strip=True) if snippet_tag else ""
            snippet = re.sub(r"\s+", " ", snippet).strip()
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet[:800],
                    score=max(0.5, 0.8 - len(results) * 0.04),
                )
            )
            seen_urls.add(url)
            if len(results) >= top_n:
                break
        return results
