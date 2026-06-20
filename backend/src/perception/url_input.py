"""URL input parser — fetch and extract readable text from web pages."""

from __future__ import annotations

import html
import logging
import re
from html.parser import HTMLParser
from typing import Any

import httpx

from perception.types import AttachmentMeta

logger = logging.getLogger(__name__)

# Safety limits
_MAX_FETCH_SIZE = 2 * 1024 * 1024  # 2 MiB
_FETCH_TIMEOUT = 15.0


class _TextExtractor(HTMLParser):
    """Extract visible text from HTML, dropping script/style tags."""

    def __init__(self) -> None:
        super().__init__()
        self._texts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("head", "script", "style", "nav", "footer", "header", "aside"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("head", "script", "style", "nav", "footer", "header", "aside") and self._skip > 0:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if self._skip == 0:
            self._texts.append(data)

    def get_text(self) -> str:
        raw = " ".join(self._texts)
        # Decode HTML entities and collapse whitespace
        decoded = html.unescape(raw)
        return re.sub(r"\s+", " ", decoded).strip()


class UrlInputParser:
    """Fetch a URL and return extracted text + metadata."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def parse(self, source: str, filename: str | None = None) -> AttachmentMeta:
        """Fetch URL content and extract readable text."""
        if not source.startswith(("http://", "https://")):
            return AttachmentMeta(
                type="url",
                mime_type="text/uri-list",
                source=source,
                filename=filename,
                extracted_text=None,
                metadata={"error": "Invalid URL scheme"},
            )

        client = self._client or httpx.AsyncClient(timeout=_FETCH_TIMEOUT, follow_redirects=True)
        try:
            response = await client.get(source)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                return AttachmentMeta(
                    type="url",
                    mime_type=content_type or "text/uri-list",
                    source=source,
                    filename=filename,
                    extracted_text=None,
                    metadata={"error": f"Unsupported content type: {content_type}"},
                )

            if len(response.content) > _MAX_FETCH_SIZE:
                return AttachmentMeta(
                    type="url",
                    mime_type=content_type,
                    source=source,
                    filename=filename,
                    extracted_text=None,
                    metadata={"error": "Page too large"},
                )

            extractor = _TextExtractor()
            extractor.feed(response.text)
            text = extractor.get_text()

            return AttachmentMeta(
                type="url",
                mime_type=content_type,
                source=source,
                filename=filename,
                extracted_text=text or None,
                metadata={
                    "title": _extract_title(response.text),
                    "status_code": response.status_code,
                    "content_length": len(response.content),
                },
            )
        except Exception as exc:
            logger.warning("URL fetch failed for %s: %s", source, exc)
            return AttachmentMeta(
                type="url",
                mime_type="text/uri-list",
                source=source,
                filename=filename,
                extracted_text=None,
                metadata={"error": str(exc)},
            )
        finally:
            if self._client is None and isinstance(client, httpx.AsyncClient):
                await client.aclose()


def _extract_title(html_text: str) -> str | None:
    """Best-effort <title> extraction."""
    match = re.search(r"<title[^>]*>(.*?)</title>", html_text, re.IGNORECASE | re.DOTALL)
    if match:
        return html.unescape(re.sub(r"\s+", " ", match.group(1)).strip()) or None
    return None
