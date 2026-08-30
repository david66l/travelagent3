"""Tavily Search Skill - AI-native search engine for high-quality results."""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from core.external_api_tracker import record_external_api_usage
from core.settings import settings
from core.thought_logger import get_current_user_id, get_current_user_role

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0


class TavilySearchSkill:
    """Tavily API search - optimized for AI applications.

    Sign up: https://tavily.com (free 1000 calls/month)
    """

    API_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.tavily_api_key

    async def _track_success(self) -> None:
        try:
            user_id = get_current_user_id()
            role = get_current_user_role() or "user"
            await record_external_api_usage(
                "tavily",
                status="success",
                data_source="api",
                user_id=user_id,
                role=role,
            )
        except Exception:
            pass

    async def search(
        self, query: str, top_n: int = 5, search_depth: str = "advanced"
    ) -> list[SearchResult]:
        """Search using Tavily API.

        Args:
            query: Search query (supports Chinese)
            top_n: Number of results to return
            search_depth: "basic" (fast) or "advanced" (comprehensive)
        """
        if not self.api_key:
            return []

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                self.API_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": min(top_n, 20),
                    "include_answer": False,
                    "include_images": False,
                    "include_raw_content": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for r in data.get("results", []):
                content = r.get("content", "")
                snippet = content[:800] if len(content) > 800 else content
                results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=snippet,
                        score=r.get("score", 0.0),
                    )
                )
            await self._track_success()
            return results

    async def search_multiple(
        self, queries: list[str], top_n: int = 5
    ) -> dict[str, list[SearchResult]]:
        """Run multiple searches in parallel."""
        tasks = [self.search(q, top_n) for q in queries]
        results = await asyncio.gather(*tasks)
        return {q: r for q, r in zip(queries, results)}

    async def search_with_context(
        self, query: str, top_n: int = 5, search_depth: str = "advanced"
    ) -> tuple[list[SearchResult], str]:
        """Search and return results + Tavily's AI-generated answer."""
        if not self.api_key:
            return [], ""

        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.post(
                self.API_URL,
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": min(top_n, 20),
                    "include_answer": True,
                    "include_images": False,
                    "include_raw_content": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for r in data.get("results", []):
                content = r.get("content", "")
                snippet = content[:800] if len(content) > 800 else content
                results.append(
                    SearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        snippet=snippet,
                        score=r.get("score", 0.0),
                    )
                )

            answer = data.get("answer", "") or ""

            # Log search query and results to thought logger
            try:
                from core.thought_logger import thought_logger, get_current_step_name

                if get_current_step_name():
                    thought_logger.log_search_result(
                        query=query,
                        results=[r.title for r in results],
                    )
            except Exception:
                pass

            await self._track_success()
            return results, answer


# Backward-compatible import for skills that expect WebSearchSkill interface
# If Tavily key is not set, fallback to DuckDuckGo
from skills.web_search import SearchProvidersUnavailable, WebSearchSkill  # noqa: E402


class UnifiedSearchSkill:
    """Unified search that prefers Tavily, falls back to DuckDuckGo."""

    def __init__(self):
        self.tavily = TavilySearchSkill()
        self.duckduckgo = WebSearchSkill()
        self.prefer_tavily = settings.search_engine == "tavily" and bool(settings.tavily_api_key)

    async def search(self, query: str, top_n: int = 5) -> list[SearchResult]:
        """Search using preferred engine, fallback to DuckDuckGo."""
        results: list[SearchResult] = []
        preferred_responded = False
        if self.prefer_tavily:
            try:
                results = await self.tavily.search(query, top_n)
                preferred_responded = True
            except Exception:
                logger.warning("Tavily search failed; trying public providers", exc_info=True)
        if not results:
            try:
                results = await self.duckduckgo.search(query, top_n)
            except SearchProvidersUnavailable:
                if not preferred_responded:
                    raise

        # Log search query and results to thought logger
        try:
            from core.thought_logger import thought_logger, get_current_step_name

            if get_current_step_name():
                thought_logger.log_search_result(
                    query=query,
                    results=[r.title for r in results],
                )
        except Exception:
            pass

        return results

    async def search_with_context(
        self, query: str, top_n: int = 5
    ) -> tuple[list[SearchResult], str]:
        """Search using preferred engine, return results + answer/context."""
        results: list[SearchResult] = []
        answer = ""
        preferred_responded = False
        if self.prefer_tavily:
            try:
                results, answer = await self.tavily.search_with_context(query, top_n)
                preferred_responded = True
            except Exception:
                logger.warning(
                    "Tavily context search failed; trying public providers", exc_info=True
                )
        if not results:
            try:
                results = await self.duckduckgo.search(query, top_n)
            except SearchProvidersUnavailable:
                if not preferred_responded:
                    raise

        # Log search query and results to thought logger
        try:
            from core.thought_logger import thought_logger, get_current_step_name

            if get_current_step_name():
                thought_logger.log_search_result(
                    query=query,
                    results=[r.title for r in results],
                )
        except Exception:
            pass

        return results, answer
