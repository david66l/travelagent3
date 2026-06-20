"""Unit tests for the composite rate-limit and cost controller."""

from unittest.mock import AsyncMock

import pytest

from core.exceptions import RateLimitException
from monitoring.rate_limit_controller import RateLimitCostController


@pytest.fixture
def controller(mock_redis):
    return RateLimitCostController(redis_client=mock_redis)


@pytest.mark.asyncio
async def test_check_ip_rate_allows_then_blocks(controller, monkeypatch):
    monkeypatch.setattr(controller.settings, "rate_limit_ip_per_minute", 2)
    controller.redis.incr = AsyncMock(side_effect=[1, 2, 3])

    await controller.check_ip_rate("1.1.1.1")
    await controller.check_ip_rate("1.1.1.1")

    with pytest.raises(RateLimitException) as exc_info:
        await controller.check_ip_rate("1.1.1.1")

    assert "IP rate limit exceeded" in str(exc_info.value)
    assert controller.redis.incr.call_count == 3


@pytest.mark.asyncio
async def test_check_user_rate_allows_then_blocks(controller, monkeypatch):
    monkeypatch.setattr(controller.settings, "rate_limit_user_per_minute", 2)
    controller.redis.incr = AsyncMock(side_effect=[1, 2, 3])

    await controller.check_user_rate("user-1", "free")
    await controller.check_user_rate("user-1", "free")

    with pytest.raises(RateLimitException) as exc_info:
        await controller.check_user_rate("user-1", "free")

    assert "User rate limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_check_user_rate_uses_guest_limit(controller, monkeypatch):
    monkeypatch.setattr(controller.settings, "rate_limit_guest_per_minute", 1)
    controller.redis.incr = AsyncMock(side_effect=[1, 2])

    await controller.check_user_rate("guest-1", "guest")

    with pytest.raises(RateLimitException) as exc_info:
        await controller.check_user_rate("guest-1", "guest")

    assert "User rate limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_check_concurrent_sse_allows_then_blocks(controller, monkeypatch):
    monkeypatch.setattr(controller.settings, "rate_limit_max_concurrent_sse", 2)
    controller.redis.incr = AsyncMock(side_effect=[1, 2, 3])

    await controller.check_concurrent_sse("user-1")
    await controller.check_concurrent_sse("user-1")

    with pytest.raises(RateLimitException) as exc_info:
        await controller.check_concurrent_sse("user-1")

    assert "Concurrent SSE limit exceeded" in str(exc_info.value)


@pytest.mark.asyncio
async def test_check_token_quota_delegates_and_blocks(controller, monkeypatch):
    calls = []

    async def _fake_check(user_id, role, tokens):
        calls.append((user_id, role, tokens))
        if len(calls) > 1:
            raise RateLimitException("token blocked")

    monkeypatch.setattr(
        "monitoring.rate_limit_controller.check_and_record_tokens",
        AsyncMock(side_effect=_fake_check),
    )

    await controller.check_token_quota("user-1", "free", 10)
    assert calls == [("user-1", "free", 10)]

    with pytest.raises(RateLimitException) as exc_info:
        await controller.check_token_quota("user-1", "free", 10)
    assert "token blocked" in str(exc_info.value)


@pytest.mark.asyncio
async def test_check_external_api_quota_allows_then_blocks(controller, monkeypatch):
    monkeypatch.setattr(controller.settings, "external_api_quota_free_daily", 5)
    controller.redis.incrby = AsyncMock(side_effect=[3, 6])
    controller.redis.decrby = AsyncMock()

    await controller.check_external_api_quota("user-1", "free", calls=2)

    with pytest.raises(RateLimitException) as exc_info:
        await controller.check_external_api_quota("user-1", "free", calls=2)

    assert "Daily external API quota exceeded" in str(exc_info.value)
    controller.redis.decrby.assert_awaited_once()


@pytest.mark.asyncio
async def test_check_cost_circuit_blocks_when_active(controller, monkeypatch):
    monkeypatch.setattr(
        "monitoring.rate_limit_controller.is_cost_circuit_active",
        AsyncMock(return_value=True),
    )

    with pytest.raises(RateLimitException) as exc_info:
        await controller.check_cost_circuit()

    assert "Global cost circuit active" in str(exc_info.value)


@pytest.mark.asyncio
async def test_check_cost_circuit_allows_when_inactive(controller, monkeypatch):
    monkeypatch.setattr(
        "monitoring.rate_limit_controller.is_cost_circuit_active",
        AsyncMock(return_value=False),
    )

    await controller.check_cost_circuit()


@pytest.mark.parametrize(
    "blocking_method, kwargs",
    [
        (
            "check_ip_rate",
            {
                "ip": "1.1.1.1",
                "concurrent_sse": True,
                "estimated_tokens": 1,
                "external_api_cost": 1,
            },
        ),
        (
            "check_user_rate",
            {
                "ip": None,
                "concurrent_sse": True,
                "estimated_tokens": 1,
                "external_api_cost": 1,
            },
        ),
        (
            "check_concurrent_sse",
            {
                "ip": None,
                "concurrent_sse": True,
                "estimated_tokens": 1,
                "external_api_cost": 1,
            },
        ),
        (
            "check_token_quota",
            {
                "ip": None,
                "concurrent_sse": False,
                "estimated_tokens": 1,
                "external_api_cost": 1,
            },
        ),
        (
            "check_external_api_quota",
            {
                "ip": None,
                "concurrent_sse": False,
                "estimated_tokens": 0,
                "external_api_cost": 1,
            },
        ),
        (
            "check_cost_circuit",
            {
                "ip": None,
                "concurrent_sse": False,
                "estimated_tokens": 0,
                "external_api_cost": 0,
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_check_request_allowed_order(controller, monkeypatch, blocking_method, kwargs):
    """Each layer must raise before any later layer is evaluated."""

    def _raise(message):
        async def _inner(*args, **kwargs):
            raise RateLimitException(message)

        return _inner

    # Default all checks to no-ops.
    for method in (
        "check_ip_rate",
        "check_user_rate",
        "check_concurrent_sse",
        "check_token_quota",
        "check_external_api_quota",
        "check_cost_circuit",
    ):
        monkeypatch.setattr(controller, method, AsyncMock())

    monkeypatch.setattr(
        controller,
        blocking_method,
        _raise(f"{blocking_method} blocked"),
    )

    with pytest.raises(RateLimitException) as exc_info:
        await controller.check_request_allowed("user-1", "free", **kwargs)

    assert f"{blocking_method} blocked" in str(exc_info.value)


@pytest.mark.asyncio
async def test_record_request_records_tokens_and_cost(controller, monkeypatch):
    record_tokens = AsyncMock()
    record_cost = AsyncMock()
    monkeypatch.setattr("monitoring.rate_limit_controller.record_daily_tokens", record_tokens)
    monkeypatch.setattr("monitoring.rate_limit_controller.record_external_api_cost", record_cost)

    await controller.record_request("user-1", "free", tokens=10, api_cost=1.5)

    record_tokens.assert_awaited_once_with(10)
    record_cost.assert_awaited_once_with(1.5)


@pytest.mark.asyncio
async def test_record_request_ignores_zeros(controller, monkeypatch):
    record_tokens = AsyncMock()
    record_cost = AsyncMock()
    monkeypatch.setattr("monitoring.rate_limit_controller.record_daily_tokens", record_tokens)
    monkeypatch.setattr("monitoring.rate_limit_controller.record_external_api_cost", record_cost)

    await controller.record_request("user-1", "free")

    record_tokens.assert_not_awaited()
    record_cost.assert_not_awaited()


@pytest.mark.asyncio
async def test_redis_failure_fails_open(controller):
    controller.redis.incr = AsyncMock(side_effect=RuntimeError("redis down"))

    await controller.check_ip_rate("1.1.1.1")
    await controller.check_user_rate("user-1", "free")
    await controller.check_concurrent_sse("user-1")
