"""Tests for audio input parser."""

import base64
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from perception.audio_input import AudioInputParser


@pytest.fixture
def parser():
    return AudioInputParser()


@pytest.mark.asyncio
async def test_parse_audio_data_uri_success(parser):
    audio_b64 = base64.b64encode(b"fake audio bytes").decode()
    source = f"data:audio/wav;base64,{audio_b64}"

    with patch.object(parser, "_transcribe", new=AsyncMock(return_value="hello world")):
        result = await parser.parse(source, filename="test.wav")

    assert result["type"] == "audio"
    assert result["mime_type"] == "audio/wav"
    assert result["extracted_text"] == "hello world"
    assert result["metadata"]["transcribed"] is True


@pytest.mark.asyncio
async def test_parse_audio_missing_key_returns_error(parser):
    audio_b64 = base64.b64encode(b"fake audio bytes").decode()
    source = f"data:audio/mp3;base64,{audio_b64}"

    with patch("perception.audio_input.settings.openai_api_key", ""):
        result = await parser.parse(source, filename="test.mp3")

    assert result["extracted_text"] is None
    assert "OPENAI_API_KEY" in result["metadata"]["error"]


@pytest.mark.asyncio
async def test_parse_audio_transcribe_failure_returns_error(parser):
    audio_b64 = base64.b64encode(b"fake audio bytes").decode()
    source = f"data:audio/mp3;base64,{audio_b64}"

    with patch("perception.audio_input.settings.openai_api_key", "sk-test"):
        with patch.object(
            parser, "_transcribe", new=AsyncMock(side_effect=RuntimeError("api error"))
        ):
            result = await parser.parse(source, filename="test.mp3")

    assert result["extracted_text"] is None
    assert "api error" in result["metadata"]["error"]


@pytest.mark.asyncio
async def test_parse_audio_url_downloads_and_transcribes(parser):
    audio_bytes = b"fake audio bytes"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=audio_bytes, headers={"content-type": "audio/mpeg"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    parser_with_client = AudioInputParser(client=client)

    with patch.object(
        parser_with_client, "_transcribe", new=AsyncMock(return_value="transcribed from url")
    ):
        result = await parser_with_client.parse("https://example.com/audio.mp3")

    assert result["extracted_text"] == "transcribed from url"
    assert result["mime_type"] == "audio/mpeg"
