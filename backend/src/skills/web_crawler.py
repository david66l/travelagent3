"""Web Crawler Skill - aiohttp async crawling with rate limiting."""

import asyncio
import random
from dataclasses import dataclass
from typing import Optional

import aiohttp
from bs4 import BeautifulSoup

from core.settings import settings


@dataclass
class CrawledPage:
    url: str
    title: str
    text: str
    links: list[str]


class WebCrawlerSkill:
    """Async web crawler with rate limiting."""

    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    def __init__(self):
        self._last_request_time: float = 0.0
        self._semaphore = asyncio.Semaphore(5)

    async def crawl(self, url: str, extract_fields: Optional[list[str]] = None) -> CrawledPage:
        await self._rate_limit()

        async with self._semaphore:
            headers = {"User-Agent": random.choice(self.USER_AGENTS)}
            async with aiohttp.ClientSession(headers=headers) as session:
                try:
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=settings.crawl_timeout)
                    ) as resp:
                        if resp.status != 200:
                            return CrawledPage(url=url, title="", text="", links=[])
                        html = await resp.text()
                        return self._parse(html, url)
                except Exception:
                    return CrawledPage(url=url, title="", text="", links=[])

    async def crawl_multiple(self, urls: list[str]) -> list[CrawledPage]:
        tasks = [self.crawl(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def _rate_limit(self):
        now = asyncio.get_event_loop().time()
        elapsed = now - self._last_request_time
        if elapsed < settings.crawl_rate_limit:
            await asyncio.sleep(settings.crawl_rate_limit - elapsed)
        self._last_request_time = asyncio.get_event_loop().time()

    def _parse(self, html: str, url: str) -> CrawledPage:
        soup = BeautifulSoup(html, "html.parser")

        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        title = soup.title.get_text(strip=True) if soup.title else ""
        text = soup.get_text(separator="\n", strip=True)
        links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].startswith("http")]

        return CrawledPage(url=url, title=title, text=text[:5000], links=links[:20])
