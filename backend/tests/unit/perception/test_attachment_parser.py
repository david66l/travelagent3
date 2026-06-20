"""Tests for attachment parser dispatcher."""

import base64
from unittest.mock import AsyncMock, patch

import pytest

from perception import AttachmentParser


@pytest.fixture
def parser():
    return AttachmentParser()


@pytest.mark.asyncio
async def test_parse_image_delegates_to_ocr(parser):
    with patch.object(
        parser._image_parser,
        "parse",
        new=AsyncMock(return_value={"type": "image", "extracted_text": "OCR text"}),
    ) as mock_parse:
        result = await parser.parse(
            {"type": "image", "source": "data:image/png;base64,abc", "filename": "a.png"}
        )
        mock_parse.assert_awaited_once_with("data:image/png;base64,abc", filename="a.png")
        assert result["type"] == "image"
        assert result["extracted_text"] == "OCR text"


@pytest.mark.asyncio
async def test_parse_pdf_delegates_to_pdf_parser(parser):
    with patch.object(
        parser._pdf_parser,
        "parse",
        new=AsyncMock(return_value={"type": "pdf", "extracted_text": "PDF text"}),
    ) as mock_parse:
        result = await parser.parse(
            {"type": "pdf", "source": "data:application/pdf;base64,abc", "filename": "a.pdf"}
        )
        mock_parse.assert_awaited_once_with("data:application/pdf;base64,abc", filename="a.pdf")
        assert result["type"] == "pdf"
        assert result["extracted_text"] == "PDF text"


@pytest.mark.asyncio
async def test_parse_audio_delegates_to_audio_parser(parser):
    with patch.object(
        parser._audio_parser,
        "parse",
        new=AsyncMock(return_value={"type": "audio", "extracted_text": "hello"}),
    ) as mock_parse:
        result = await parser.parse({"type": "audio", "source": "data:audio/wav;base64,abc"})
        mock_parse.assert_awaited_once()
        assert result["type"] == "audio"
        assert result["extracted_text"] == "hello"


@pytest.mark.asyncio
async def test_parse_url_delegates_to_url_parser(parser):
    with patch.object(
        parser._url_parser,
        "parse",
        new=AsyncMock(return_value={"type": "url", "extracted_text": "page text"}),
    ) as mock_parse:
        result = await parser.parse({"type": "url", "source": "https://example.com/page"})
        mock_parse.assert_awaited_once()
        assert result["type"] == "url"
        assert result["extracted_text"] == "page text"


@pytest.mark.asyncio
async def test_parse_many_concurrently(parser):
    with patch.object(
        parser._image_parser,
        "parse",
        new=AsyncMock(return_value={"type": "image", "extracted_text": "img"}),
    ):
        results = await parser.parse_many(
            [
                {"type": "image", "source": "data:image/png;base64,abc"},
                {"type": "image", "source": "data:image/png;base64,def"},
            ]
        )
        assert len(results) == 2
        assert results[0]["extracted_text"] == "img"


def test_pdf_parser_extracts_text_from_blank_pdf():
    """A blank PDF should be parsed without crashing."""
    from perception.pdf_input import PdfInputParser

    # Minimal valid PDF in base64 (one empty page)
    pdf_bytes = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 100 100]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n156\n%%EOF\n"
    source = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode()

    import asyncio

    parser = PdfInputParser()
    result = asyncio.run(parser.parse(source, filename="blank.pdf"))
    assert result["type"] == "pdf"
    assert result["metadata"]["page_count"] == 1
