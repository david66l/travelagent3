"""Tests for Redis-persisted task retry helpers."""

from unittest.mock import AsyncMock

import pytest

from core.task_retry import compute_retry_delay, is_retryable


def test_is_retryable_validation_errors():
    assert is_retryable(ValueError("bad")) is False
    assert is_retryable(PermissionError()) is False


def test_is_retryable_transient_errors():
    assert is_retryable(TimeoutError()) is True
    assert is_retryable(ConnectionError()) is True
    assert is_retryable(RuntimeError("api down")) is True


def test_compute_retry_delay_caps_and_positive():
    d1 = compute_retry_delay(1, initial=2.0, max_delay=30.0, jitter=False)
    d2 = compute_retry_delay(3, initial=2.0, max_delay=30.0, jitter=False)
    assert d1 == 2.0
    assert d2 == 8.0
    d_cap = compute_retry_delay(10, initial=2.0, max_delay=30.0, jitter=False)
    assert d_cap == 30.0


@pytest.mark.asyncio
async def test_increment_and_reset_retry(mock_redis):
    from core.task_retry import get_retry_count, increment_retry, reset_retry

    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.incr = AsyncMock(side_effect=[1, 2])
    mock_redis.expire = AsyncMock()

    count = await increment_retry("task", "job-1")
    assert count == 1
    count2 = await increment_retry("task", "job-1")
    assert count2 == 2

    mock_redis.get = AsyncMock(return_value="2")
    assert await get_retry_count("task", "job-1") == 2

    await reset_retry("task", "job-1")
    mock_redis.delete.assert_called()
