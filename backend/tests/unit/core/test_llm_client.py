"""Unit tests for LLM client vLLM routing."""

import asyncio
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


def test_explicit_vllm_endpoint_supports_split_student_teacher_services():
    client = LLMClient(
        base_url="http://vllm-teacher:8002/v1",
        api_key="test-key",
        using_vllm=True,
    )

    assert client._using_vllm is True
    assert client.client.base_url == "http://vllm-teacher:8002/v1/"


@pytest.mark.asyncio
async def test_agent_policy_state_bypasses_generic_middle_truncation():
    client = LLMClient()
    state_json = '{"artifact":"' + "x" * 3000 + '","tail":"must-remain"}'
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": state_json},
    ]

    with patch("core.llm_client.is_cost_circuit_active", new=AsyncMock(return_value=False)):
        with patch("core.llm_client.select_model", return_value="policy-model"):
            _, prepared, _ = await client._prepare_request(messages, "agent_policy")

    assert prepared == messages
    assert prepared[1]["content"].endswith('"tail":"must-remain"}')


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


@pytest.mark.asyncio
async def test_structured_call_exposes_actual_total_token_usage():
    client = LLMClient()
    client._prepare_request = AsyncMock(
        return_value=("model", [{"role": "user", "content": "parse"}], "free")
    )
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content='{"intent":"chat"}'))]
    response.usage.total_tokens = 37
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 17
    client._create_completion = AsyncMock(return_value=response)

    await client.structured_call(
        [{"role": "user", "content": "parse"}], _IntentOutput, task_type="agent_policy"
    )

    assert client.last_token_usage == 37


@pytest.mark.asyncio
async def test_concurrent_structured_calls_keep_token_usage_task_local():
    client = LLMClient()
    client._prepare_request = AsyncMock(
        return_value=("model", [{"role": "user", "content": "parse"}], "free")
    )
    barrier = asyncio.Event()
    calls = 0

    async def completion(**kwargs):
        nonlocal calls
        calls += 1
        call_number = calls
        if call_number == 1:
            await barrier.wait()
        else:
            barrier.set()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content='{"intent":"chat"}'))]
        response.usage.total_tokens = 10 * call_number
        response.usage.prompt_tokens = 5 * call_number
        response.usage.completion_tokens = 5 * call_number
        return response

    client._create_completion = AsyncMock(side_effect=completion)

    async def invoke() -> int:
        await client.structured_call(
            [{"role": "user", "content": "parse"}],
            _IntentOutput,
            task_type="agent_policy",
        )
        return client.last_token_usage

    usages = await asyncio.gather(invoke(), invoke())

    assert sorted(usages) == [10, 20]


def test_output_format_disables_deepseek_thinking_without_affecting_other_tasks():
    client = LLMClient()

    assert client._thinking_extra_body("deepseek-v4-flash", "output_format") == {
        "thinking": {"type": "disabled"}
    }
    assert client._thinking_extra_body("deepseek-v4-flash", "agent_policy") == {
        "thinking": {"type": "disabled"}
    }
    assert client._thinking_extra_body("deepseek-v4-flash", "planning") is None
    assert client._thinking_extra_body("qwen2.5-7b-instruct", "output_format") is None


@pytest.mark.asyncio
async def test_native_tool_call_returns_one_parsed_function_and_usage():
    client = LLMClient()
    client._prepare_request = AsyncMock(
        return_value=("policy-model", [{"role": "user", "content": "state"}], "free")
    )
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = [MagicMock()]
    response.choices[0].message.tool_calls[0].function.name = "get_weather"
    response.choices[0].message.tool_calls[0].function.arguments = '{"date":"2026-08-12"}'
    response.usage.total_tokens = 29
    response.usage.prompt_tokens = 20
    response.usage.completion_tokens = 9
    client._create_completion = AsyncMock(return_value=response)

    result = await client.tool_call(
        [{"role": "user", "content": "state"}],
        [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        model_override="local-agent-policy",
        seed=73421,
    )

    assert result == {"action": "get_weather", "arguments": {"date": "2026-08-12"}}
    assert client.last_token_usage == 29
    assert client.last_request_metrics is not None
    assert client.last_request_metrics.model == "local-agent-policy"
    assert client.last_request_metrics.prompt_tokens == 20
    assert client.last_request_metrics.completion_tokens == 9
    assert client.last_request_metrics.backend == "cloud-openai-compatible"
    request = client._create_completion.await_args.kwargs
    assert request["model"] == "local-agent-policy"
    assert request["tool_choice"] == "required"
    assert request["parallel_tool_calls"] is False
    assert request["seed"] == 73421


@pytest.mark.asyncio
async def test_vllm_tool_call_disables_qwen_thinking_template():
    client = LLMClient()
    client._using_vllm = True
    client._prepare_request = AsyncMock(
        return_value=("qwen3-policy", [{"role": "user", "content": "state"}], "free")
    )
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.tool_calls = [MagicMock()]
    response.choices[0].message.tool_calls[0].function.name = "ask_user"
    response.choices[0].message.tool_calls[0].function.arguments = "{}"
    response.usage.total_tokens = 5
    client._create_completion = AsyncMock(return_value=response)

    await client.tool_call(
        [{"role": "user", "content": "state"}],
        [{"type": "function", "function": {"name": "ask_user", "parameters": {}}}],
    )

    request = client._create_completion.await_args.kwargs
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert client.last_request_metrics is not None
    assert client.last_request_metrics.backend == "vllm"
    assert client.last_request_metrics.thinking_mode == "disabled"
