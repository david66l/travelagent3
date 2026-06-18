"""Unit tests for LLM client vLLM routing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIStatusError

from core.llm_client import LLMClient
from core.settings import settings


@pytest.mark.asyncio
async def test_vllm_enabled_uses_vllm_base_url():
    with patch.object(settings, "vllm_enabled", True):
        with patch.object(settings, "vllm_base_url", "http://vllm:8000/v1"):
            with patch.object(settings, "vllm_api_key", "test-key"):
                client = LLMClient()
                assert client._using_vllm is True
                assert client.client.base_url == "http://vllm:8000/v1/"


@pytest.mark.asyncio
async def test_create_completion_retries_on_vllm_503():
    client = LLMClient()
    client._using_vllm = True

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
    mock_response.usage = None

    err = APIStatusError(
        message="unavailable",
        response=MagicMock(status_code=503),
        body=None,
    )
    client.client = MagicMock()
    client.client.chat.completions.create = AsyncMock(side_effect=[err, mock_response])

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await client._create_completion(model="m", messages=[])

    assert result is mock_response
    assert client.client.chat.completions.create.await_count == 2
