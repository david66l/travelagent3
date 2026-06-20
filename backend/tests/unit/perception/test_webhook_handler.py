"""Tests for webhook event handler."""

import json
from unittest.mock import AsyncMock

import pytest

from core.redis_client import redis_client
from perception.webhook_handler import WebhookHandler


@pytest.mark.asyncio
async def test_validate_missing_type():
    ok, error = WebhookHandler.validate({"payload": {}})
    assert not ok
    assert "type" in error


@pytest.mark.asyncio
async def test_validate_unsupported_type():
    ok, error = WebhookHandler.validate({"type": "unknown", "payload": {}})
    assert not ok
    assert "unsupported" in error


@pytest.mark.asyncio
async def test_validate_missing_payload():
    ok, error = WebhookHandler.validate({"type": "weather_alert"})
    assert not ok
    assert "payload" in error


@pytest.mark.asyncio
async def test_enqueue_calls_redis_lpush(mock_redis):
    mock_redis.lpush.reset_mock()
    event = {"type": "weather_alert", "payload": {"city": "杭州", "alert": "暴雨"}}
    await WebhookHandler.enqueue(event)
    mock_redis.lpush.assert_awaited_once_with("replan_queue", json.dumps(event))


@pytest.mark.asyncio
async def test_list_pending_returns_events():
    event = {"type": "weather_alert", "payload": {"city": "杭州"}}
    redis_client.lrange = AsyncMock(return_value=[json.dumps(event)])
    pending = await WebhookHandler.list_pending()
    assert len(pending) == 1
    assert pending[0]["type"] == "weather_alert"


@pytest.mark.asyncio
async def test_clear_calls_redis_delete(mock_redis):
    mock_redis.delete.reset_mock()
    await WebhookHandler.clear()
    mock_redis.delete.assert_awaited_once_with("replan_queue")
