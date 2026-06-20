"""Unified attachment parser dispatcher."""

from __future__ import annotations

import logging
from typing import Any

from perception.audio_input import AudioInputParser
from perception.image_input import ImageInputParser
from perception.pdf_input import PdfInputParser
from perception.types import AttachmentMeta
from perception.url_input import UrlInputParser

logger = logging.getLogger(__name__)


class AttachmentParser:
    """Parse image / PDF / audio / URL attachments into AttachmentMeta."""

    def __init__(self) -> None:
        self._image_parser = ImageInputParser()
        self._pdf_parser = PdfInputParser()
        self._audio_parser = AudioInputParser()
        self._url_parser = UrlInputParser()

    async def parse(self, attachment: dict[str, Any]) -> AttachmentMeta:
        """Dispatch to the correct parser based on attachment type."""
        att_type = attachment.get("type", "file")
        source = attachment.get("source", "")
        filename = attachment.get("filename")

        if att_type == "image":
            return await self._image_parser.parse(source, filename=filename)
        if att_type == "pdf":
            return await self._pdf_parser.parse(source, filename=filename)
        if att_type == "audio":
            return await self._parse_audio(source, filename)
        if att_type == "url":
            return await self._parse_url(source, filename)

        # Generic file: keep metadata but do not attempt extraction.
        return AttachmentMeta(
            type="file",
            mime_type=attachment.get("mime_type", "application/octet-stream"),
            source=source,
            filename=filename,
            extracted_text=None,
            metadata={},
        )

    async def parse_many(
        self, attachments: list[dict[str, Any]]
    ) -> list[AttachmentMeta]:
        """Parse a list of attachments concurrently."""
        import asyncio

        if not attachments:
            return []
        return await asyncio.gather(*[self.parse(att) for att in attachments])

    async def _parse_audio(self, source: str, filename: str | None) -> AttachmentMeta:
        """Transcribe audio to text via OpenAI-compatible Whisper API."""
        return await self._audio_parser.parse(source, filename=filename)

    async def _parse_url(self, source: str, filename: str | None) -> AttachmentMeta:
        """Fetch URL and extract readable text."""
        return await self._url_parser.parse(source, filename=filename)
