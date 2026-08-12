"""Audio input parser — fetch and transcribe speech to text."""

from __future__ import annotations

import base64
import io
import logging
import mimetypes

import httpx

from core.settings import settings
from perception.types import AttachmentMeta

logger = logging.getLogger(__name__)

# Safety limits
_MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25 MiB (Whisper limit)
_FETCH_TIMEOUT = 30.0
_TRANSCRIBE_TIMEOUT = 60.0


class AudioInputParser:
    """Transcribe audio attachments using an OpenAI-compatible Whisper API."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._client = client

    async def parse(self, source: str, filename: str | None = None) -> AttachmentMeta:
        """Download audio (if URL) and transcribe to text."""
        try:
            audio_bytes, mime_type = await self._load_audio(source)
        except Exception as exc:
            logger.warning("Audio load failed: %s", exc)
            return self._error_meta(source, filename, str(exc), mime_type="audio/*")

        if len(audio_bytes) > _MAX_AUDIO_SIZE:
            return self._error_meta(
                source, filename, "Audio file too large", mime_type=mime_type or "audio/*"
            )

        if not settings.openai_api_key:
            return self._error_meta(
                source,
                filename,
                "OPENAI_API_KEY not configured; transcription skipped",
                mime_type=mime_type or "audio/*",
            )

        try:
            text = await self._transcribe(audio_bytes, filename, mime_type)
            return AttachmentMeta(
                type="audio",
                mime_type=mime_type or "audio/*",
                source=source,
                filename=filename,
                extracted_text=text or None,
                metadata={"transcribed": True, "duration_seconds": None},
            )
        except Exception as exc:
            logger.warning("Audio transcription failed: %s", exc)
            return self._error_meta(source, filename, str(exc), mime_type=mime_type or "audio/*")

    async def _load_audio(self, source: str) -> tuple[bytes, str | None]:
        """Resolve audio bytes and MIME type from a data URI or URL."""
        if source.startswith("data:"):
            return self._decode_data_uri(source)

        if source.startswith(("http://", "https://")):
            client = self._client or httpx.AsyncClient(
                timeout=_FETCH_TIMEOUT, follow_redirects=True
            )
            try:
                response = await client.get(source)
                response.raise_for_status()
                content_type = response.headers.get("content-type")
                return response.content, content_type
            finally:
                if self._client is None and isinstance(client, httpx.AsyncClient):
                    await client.aclose()

        # Treat as raw base64
        return base64.b64decode(source), None

    @staticmethod
    def _decode_data_uri(source: str) -> tuple[bytes, str | None]:
        """Decode a base64 data URI into bytes + MIME type."""
        header, _, payload = source.partition(",")
        mime_type = None
        if header.startswith("data:"):
            mime_part = header[5:].split(";")[0]
            if mime_part:
                mime_type = mime_part
        raw = base64.b64decode(payload) if payload else base64.b64decode(source)
        return raw, mime_type

    async def _transcribe(
        self, audio_bytes: bytes, filename: str | None, mime_type: str | None
    ) -> str:
        """Call OpenAI Whisper-compatible /audio/transcriptions endpoint."""
        client = self._client or httpx.AsyncClient(timeout=_TRANSCRIBE_TIMEOUT)
        try:
            ext = _guess_extension(filename, mime_type)
            files = {
                "file": (
                    filename or f"audio.{ext}",
                    io.BytesIO(audio_bytes),
                    mime_type or f"audio/{ext}",
                ),
                "model": (None, "whisper-1"),
            }
            response = await client.post(
                "https://api.openai.com/v1/audio/transcriptions",
                files=files,
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            return str(data.get("text", "")).strip()
        finally:
            if self._client is None and isinstance(client, httpx.AsyncClient):
                await client.aclose()

    def _error_meta(
        self, source: str, filename: str | None, error: str, mime_type: str
    ) -> AttachmentMeta:
        return AttachmentMeta(
            type="audio",
            mime_type=mime_type,
            source=source,
            filename=filename,
            extracted_text=None,
            metadata={"error": error},
        )


def _guess_extension(filename: str | None, mime_type: str | None) -> str:
    """Return a file extension suitable for Whisper upload."""
    if filename:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in ("mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"):
            return ext
    if mime_type:
        guess = mimetypes.guess_extension(mime_type)
        if guess:
            return guess.lstrip(".")
    return "mp3"
