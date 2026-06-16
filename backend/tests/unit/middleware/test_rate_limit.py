"""Tests for rate limiting middleware and core rate limiter."""

from unittest.mock import AsyncMock

import pytest
from starlette.requests import Request

from core.exceptions import RateLimitException
from core.rate_limit import RateLimiter, check_rate_limit, get_client_ip


class FakeRedis:
    """In-memory fake for sliding-window tests."""

    def __init__(self):
        self.windows = {}

    async def sliding_window_add(
        self, key: str, score: float, member: str, window: int, limit: int
    ) -> bool:
        now = score
        min_score = now - window
        entries = self.windows.get(key, [])
        entries = [e for e in entries if e > min_score]
        allowed = len(entries) < limit
        if allowed:
            entries.append(now)
        self.windows[key] = entries
        return allowed


class TestRateLimiter:
    async def test_allows_under_limit(self):
        fake = FakeRedis()
        limiter = RateLimiter(fake)
        for _ in range(5):
            assert await limiter.is_allowed("key", limit=5, window_seconds=60) is True

    async def test_blocks_over_limit(self):
        fake = FakeRedis()
        limiter = RateLimiter(fake)
        for _ in range(3):
            assert await limiter.is_allowed("key", limit=3, window_seconds=60) is True
        assert await limiter.is_allowed("key", limit=3, window_seconds=60) is False

    async def test_check_rate_limit_raises(self):
        fake = FakeRedis()
        with pytest.raises(RateLimitException):
            await check_rate_limit(
                fake,
                key="key",
                limit=0,
                window_seconds=60,
            )


class TestClientIp:
    def test_forwarded_for(self):
        scope = {
            "type": "http",
            "headers": [(b"x-forwarded-for", b"1.2.3.4, 5.6.7.8")],
            "client": ("127.0.0.1", 12345),
        }
        assert get_client_ip(Request(scope)) == "1.2.3.4"

    def test_fallback_to_client(self):
        scope = {
            "type": "http",
            "headers": [],
            "client": ("127.0.0.1", 12345),
        }
        assert get_client_ip(Request(scope)) == "127.0.0.1"

    def test_unknown_when_no_client(self):
        scope = {"type": "http", "headers": []}
        assert get_client_ip(Request(scope)) == "unknown"


class TestRateLimitMiddleware:
    def test_blocks_when_limit_exceeded(self, client, monkeypatch):
        import core.redis_client

        original = core.redis_client.redis_client
        core.redis_client.redis_client.sliding_window_add = AsyncMock(return_value=False)

        try:
            response = client.get("/api/v1/users/me")
            assert response.status_code == 429
            assert response.json()["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        finally:
            core.redis_client.redis_client = original
