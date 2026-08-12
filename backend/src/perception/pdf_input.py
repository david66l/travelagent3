"""PDF input parser using PyMuPDF."""

from __future__ import annotations

import base64
import io
import logging

import fitz  # PyMuPDF

from perception.types import AttachmentMeta

logger = logging.getLogger(__name__)

MAX_PAGES = 20


class PdfInputParser:
    """Extract text from PDF attachments."""

    async def parse(self, source: str, filename: str | None = None) -> AttachmentMeta:
        """Extract text from the first MAX_PAGES pages of a PDF."""
        try:
            payload = source
            if "," in source:
                payload = source.split(",", 1)[1]
            raw = base64.b64decode(payload)
            doc = fitz.open(stream=io.BytesIO(raw), filetype="pdf")
        except Exception as exc:
            logger.warning("Failed to open PDF: %s", exc)
            return self._error_meta(source, filename, exc)

        try:
            page_count = doc.page_count
            texts: list[str] = []
            processed = 0
            for page in doc:
                if processed >= MAX_PAGES:
                    break
                text = page.get_text()
                if text:
                    texts.append(text)
                processed += 1
            doc.close()

            extracted = "\n\n".join(texts).strip()
            return AttachmentMeta(
                type="pdf",
                mime_type="application/pdf",
                source=source,
                filename=filename,
                extracted_text=extracted or None,
                metadata={
                    "page_count": page_count,
                    "processed_pages": processed,
                },
            )
        except Exception as exc:
            logger.warning("PDF text extraction failed: %s", exc)
            return self._error_meta(source, filename, exc)

    def _error_meta(self, source: str, filename: str | None, exc: Exception) -> AttachmentMeta:
        return AttachmentMeta(
            type="pdf",
            mime_type="application/pdf",
            source=source,
            filename=filename,
            extracted_text=None,
            metadata={"error": str(exc)},
        )
