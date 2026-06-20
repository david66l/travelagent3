"""Image input parser using PaddleOCR."""

from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING

from perception.types import AttachmentMeta

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


class ImageInputParser:
    """Extract text from images with PaddleOCR (lazy-loaded)."""

    def __init__(self) -> None:
        self._ocr: object | None = None

    def _load_ocr(self) -> object:
        """Lazy-load PaddleOCR to avoid import/model-download overhead on startup."""
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore[import-untyped]

                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang="ch",
                    show_log=False,
                )
            except Exception as exc:  # pragma: no cover - environment dependent
                logger.warning("Failed to load PaddleOCR: %s", exc)
                raise
        return self._ocr

    def _decode(self, source: str) -> "ndarray":
        """Decode a base64 data URI or raw base64 string into a numpy array."""
        from numpy import asarray
        from PIL import Image

        payload = source
        if "," in source:
            payload = source.split(",", 1)[1]
        raw = base64.b64decode(payload)
        image = Image.open(io.BytesIO(raw))
        return asarray(image)

    async def parse(self, source: str, filename: str | None = None) -> AttachmentMeta:
        """Run OCR on the image and return structured metadata."""
        try:
            ocr = self._load_ocr()
            image = self._decode(source)
            result = ocr.ocr(image, cls=True)
        except Exception as exc:
            logger.warning("Image OCR failed: %s", exc)
            return AttachmentMeta(
                type="image",
                mime_type=self._mime_type(source, filename),
                source=source,
                filename=filename,
                extracted_text=None,
                metadata={"error": str(exc)},
            )

        texts: list[str] = []
        if result:
            for line in result:
                if line:
                    for item in line:
                        if item and len(item) >= 2:
                            texts.append(str(item[1][0]))

        extracted = "\n".join(texts)
        return AttachmentMeta(
            type="image",
            mime_type=self._mime_type(source, filename),
            source=source,
            filename=filename,
            extracted_text=extracted or None,
            metadata={"ocr_blocks": len(texts)},
        )

    @staticmethod
    def _mime_type(source: str, filename: str | None) -> str:
        if source.startswith("data:"):
            return source.split(";")[0].replace("data:", "") or "image/*"
        if filename:
            ext = filename.split(".")[-1].lower()
            mapping = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "webp": "image/webp",
                "bmp": "image/bmp",
            }
            return mapping.get(ext, "image/*")
        return "image/*"
