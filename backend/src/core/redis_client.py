"""Async Redis client with JSON helpers and TTL support."""

import json
from typing import Optional

import redis.asyncio as aioredis

from core.settings import settings


class RedisClient:
    """Async Redis wrapper with JSON serialization and TTL support."""

    def __init__(self):
        self._client: Optional[aioredis.Redis] = None

    async def connect(self):
        """Initialize Redis connection."""
        self._client = await aioredis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

    async def disconnect(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None

    def _ensure_client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("Redis client not connected. Call connect() first.")
        return self._client

    async def get(self, key: str) -> Optional[str]:
        """Get raw string value."""
        client = self._ensure_client()
        return await client.get(key)

    async def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set string value with optional TTL (seconds)."""
        client = self._ensure_client()
        await client.set(key, value, ex=ttl)

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


# Global singleton
redis_client = RedisClient()
