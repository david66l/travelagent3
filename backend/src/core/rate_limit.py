"""Rate limiting utilities using Redis sliding windows."""

import time
from uuid import uuid4

from core.exceptions import RateLimitException
from core.redis_client import RedisClient


class RateLimiter:
    """Redis-backed sliding-window rate limiter."""

    def __init__(self, redis: RedisClient):
        self._redis = redis

    async def is_allowed(self, key: str, limit: int, window_seconds: int) -> bool:
        """Return True if the request is allowed under the given limit.

        Args:
            key: Unique Redis key for this limit bucket.
            limit: Maximum number of requests in the window.
            window_seconds: Sliding window size in seconds.
        """
        score = time.time()
        member = f"{score}:{uuid4()}"
        return await self._redis.sliding_window_add(key, score, member, window_seconds, limit)


def get_client_ip(request) -> str:
    """Extract the client IP from a Starlette request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def rate_limit_key(
    *,
    kind: str,
    identifier: str,
    path: str | None = None,
) -> str:
    """Build a Redis key for a rate limit bucket."""
    if path:
        return f"rate_limit:{kind}:{identifier}:{path}"
    return f"rate_limit:{kind}:{identifier}"


async def check_rate_limit(
    redis: RedisClient,
    *,
    key: str,
    limit: int,
    window_seconds: int,
) -> None:
    """Raise RateLimitException if the bucket is over the limit."""
    limiter = RateLimiter(redis)
    allowed = await limiter.is_allowed(key, limit, window_seconds)
    if not allowed:
        raise RateLimitException(
            message="Rate limit exceeded. Please try again later.",
            retry_after=window_seconds,
        )
