"""Unit tests for LLM client vLLM routing."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import APIStatusError
from pydantic import BaseModel

from core.llm_client import LLMClient
from core.settings import settings


class _IntentOutput(BaseModel):
    intent: str


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


@pytest.mark.asyncio
async def test_intent_structured_call_disables_deepseek_thinking_only():
    client = LLMClient()
    client._prepare_request = AsyncMock(
        return_value=(
            "deepseek-v4-flash",
            [{"role": "user", "content": "parse this"}],
            "free",
        )
    )

    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content='{"intent":"generate_itinerary"}'))]
    response.usage = None
    client._create_completion = AsyncMock(return_value=response)

    result = await client.structured_call(
        [{"role": "user", "content": "parse this"}],
        _IntentOutput,
        task_type="intent",
    )

    assert result.intent == "generate_itinerary"
    request = client._create_completion.await_args.kwargs
    assert request["extra_body"] == {"thinking": {"type": "disabled"}}


def test_output_format_disables_deepseek_thinking_without_affecting_other_tasks():
    client = LLMClient()

    assert client._thinking_extra_body("deepseek-v4-flash", "output_format") == {
        "thinking": {"type": "disabled"}
    }
    assert client._thinking_extra_body("deepseek-v4-flash", "planning") is None
    assert client._thinking_extra_body("qwen2.5-7b-instruct", "output_format") is None
