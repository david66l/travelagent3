import json
import asyncio

import httpx
import pytest

from core.glm_tool_client import GLMProviderError, GLMToolClient, SINGLE_CALL_REMINDER


def reply(arguments="{}", *, count=1, finish="tool_calls", usage=True):
    data = {"id": "test-response", "model": "glm-5.3-flash", "choices": [{
        "finish_reason": finish, "message": {"reasoning_content": "private audit text",
        "tool_calls": [{"function": {"name": "get_weather", "arguments": arguments}}] * count}}]}
    if usage:
        data["usage"] = {"prompt_tokens": 120, "completion_tokens": 30,
                         "completion_tokens_details": {"reasoning_tokens": 24}}
    return data


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", ["{}", {}])
async def test_native_request_preserves_tools_and_counts_completion_only(arguments):
    requests = []
    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=reply(arguments))
    client = GLMToolClient("test-secret", transport=httpx.MockTransport(handler))
    messages = [{"role": "user", "content": "current state"}]
    tools = [{"type": "function", "function": {"name": "get_weather"}}]
    try:
        result = await client.tool_call(messages, tools)
        assert result == {"action": "get_weather", "arguments": {}}
        request = requests[0]
        assert request["messages"][:-1] == messages and request["tools"] == tools
        assert request["messages"][-1] == {"role": "user", "content": SINGLE_CALL_REMINDER}
        assert messages == [{"role": "user", "content": "current state"}]
        assert request["tool_choice"] == "auto"
        assert request["thinking"] == {"type": "enabled", "clear_thinking": True}
        assert request["reasoning_effort"] == "max"
        assert "seed" not in request and "parallel_tool_calls" not in request
        assert client.last_token_usage == 30
        assert client.last_request_metrics.total_tokens == 150
        assert client.last_generation_audit["reasoning_tokens"] == 24
        assert client.last_generation_audit["response"]["id"] == "test-response"
        assert "test-secret" not in json.dumps(client.last_generation_audit)
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [reply(count=0), reply(count=2), reply("[1]"), reply("{"), reply(finish="length")])
async def test_invalid_generation_cannot_invent_an_action(payload):
    client = GLMToolClient("test-secret", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))
    try:
        with pytest.raises(ValueError):
            await client.tool_call([], [])
        assert client.last_token_usage == 30
        assert client.last_generation_audit["response"] == payload
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_provider_failure_no_retry_no_secret_and_resets_previous_usage():
    responses = [httpx.Response(200, json=reply()), httpx.Response(401, json={"error": {"code": "1000", "message": "test-secret"}})]
    client = GLMToolClient("test-secret", transport=httpx.MockTransport(lambda r: responses.pop(0)))
    try:
        await client.tool_call([], [])
        with pytest.raises(GLMProviderError, match="HTTP 401") as error:
            await client.tool_call([], [])
        assert "test-secret" not in str(error.value)
        assert not responses
        assert client.last_request_metrics is None and client.last_token_usage == 0
        assert "test-secret" not in json.dumps(client.last_generation_audit)
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_missing_usage_is_infrastructure_failure_not_zero_cost():
    client = GLMToolClient("test-secret", transport=httpx.MockTransport(lambda r: httpx.Response(200, json=reply(usage=False))))
    try:
        with pytest.raises(GLMProviderError, match="missing token usage"):
            await client.tool_call([], [])
        assert client.last_generation_audit["usage_known"] is False
        assert client.last_request_metrics is None
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_episode_cancellation_is_not_provider_outage():
    async def handler(request):
        raise asyncio.CancelledError()
    client = GLMToolClient("test-secret", transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(asyncio.CancelledError):
            await client.tool_call([], [])
        assert client.last_generation_audit["request_cancelled"] is True
        assert "provider_error" not in client.last_generation_audit
        assert client.last_generation_audit["usage_known"] is False
    finally:
        await client.aclose()
