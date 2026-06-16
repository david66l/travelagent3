"""Async Redis client with JSON helpers and TTL support."""

import asyncio
import json
from typing import Optional

import redis.asyncio as aioredis

from core.settings import settings


class RedisClient:
    """Async Redis wrapper with JSON serialization and TTL support."""

    def __init__(self):
        self._client: Optional[aioredis.Redis] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def connect(self):
        """Initialize Redis connection bound to the current event loop."""
        self._client = await aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )
        self._loop = asyncio.get_running_loop()

    async def disconnect(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
            self._loop = None

    def _ensure_client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        current_loop = asyncio.get_running_loop()
        if self._loop is not current_loop:
            # Stale client bound to a different loop; schedule its cleanup on
            # the current loop and force a reconnect.
            stale = self._client
            self._client = None
            self._loop = None
            try:
                current_loop.create_task(stale.close())
            except Exception:
                pass
            raise RuntimeError("Redis client bound to a different event loop. Reconnecting.")
        return self._client

    async def get(self, key: str) -> Optional[str]:
        """Get raw string value."""
        client = self._ensure_client()
        return await client.get(key)

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set string value with optional TTL (seconds)."""
        client = self._ensure_client()
        await client.set(key, value, ex=ttl)

    async def set_nx(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set string value only if key does not exist, with optional TTL."""
        client = self._ensure_client()
        return bool(await client.set(key, value, nx=True, ex=ttl))

    async def get_json(self, key: str) -> Optional[dict | list]:
        """Get and deserialize JSON value."""
        raw = await self.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    async def set_json(self, key: str, value: dict | list, ttl: Optional[int] = None) -> None:
        """Serialize and store JSON value with optional TTL (seconds)."""
        await self.set(key, json.dumps(value, ensure_ascii=False), ttl=ttl)

    async def delete(self, key: str) -> None:
        """Delete a key."""
        client = self._ensure_client()
        await client.delete(key)

    async def zadd(self, key: str, mapping: dict) -> int:
        """Add members with scores to a sorted set."""
        client = self._ensure_client()
        return await client.zadd(key, mapping)

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        """Remove members from a sorted set by score range."""
        client = self._ensure_client()
        return await client.zremrangebyscore(key, min_score, max_score)

    async def zcard(self, key: str) -> int:
        """Return the cardinality of a sorted set."""
        client = self._ensure_client()
        return await client.zcard(key)

    async def zrangebyscore(self, key: str, min_score: float, max_score: float) -> list:
        """Return members of a sorted set within a score range."""
        client = self._ensure_client()
        return await client.zrangebyscore(key, min_score, max_score)

    async def expire(self, key: str, ttl: int) -> None:
        """Set expiration on a key."""
        client = self._ensure_client()
        await client.expire(key, ttl)

    async def incr(self, key: str) -> int:
        """Increment integer value at key."""
        client = self._ensure_client()
        return int(await client.incr(key))

    async def decr(self, key: str) -> int:
        """Decrement integer value at key."""
        client = self._ensure_client()
        return int(await client.decr(key))

    async def lpush(self, key: str, value: str) -> int:
        """Push value onto the head of a list."""
        client = self._ensure_client()
        return int(await client.lpush(key, value))

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        """Return a slice of a list."""
        client = self._ensure_client()
        return list(await client.lrange(key, start, end))

    async def lrem(self, key: str, count: int, value: str) -> int:
        """Remove list elements equal to value."""
        client = self._ensure_client()
        return int(await client.lrem(key, count, value))

    async def setbit(self, key: str, offset: int, value: int) -> int:
        """Set a single bit in a string value."""
        client = self._ensure_client()
        return int(await client.setbit(key, offset, value))

    async def getbit(self, key: str, offset: int) -> int:
        """Return the bit value at offset."""
        client = self._ensure_client()
        return int(await client.getbit(key, offset))

    async def scan(
        self, cursor: int = 0, match: str = "*", count: int = 100
    ) -> tuple[int, list[str]]:
        """Iterate Redis keys matching a pattern."""
        client = self._ensure_client()
        return await client.scan(cursor=cursor, match=match, count=count)

    def lock(self, key: str, ttl: int, blocking_timeout: float = 0.1):
        """Return an async Redis lock context manager."""
        client = self._ensure_client()
        return client.lock(key, timeout=ttl, blocking_timeout=blocking_timeout)

    async def sliding_window_add(
        self, key: str, score: float, member: str, window: int, limit: int
    ) -> bool:
        """Atomically add a request timestamp to a sliding window.

        Returns True if the request is within the limit, False if it should
        be throttled.
        """
        client = self._ensure_client()
        lua = """
        local key = KEYS[1]
        local window = tonumber(ARGV[1])
        local limit = tonumber(ARGV[2])
        local score = tonumber(ARGV[3])
        local member = ARGV[4]
        local min_score = score - window
        redis.call("ZREMRANGEBYSCORE", key, 0, min_score)
        local count = redis.call("ZCARD", key)
        if count < limit then
            redis.call("ZADD", key, score, member)
            redis.call("EXPIRE", key, window)
            return 1
        else
            return 0
        end
        """
        result = await client.eval(lua, 1, key, str(window), str(limit), str(score), member)
        return bool(result)


# Global singleton
redis_client = RedisClient()
