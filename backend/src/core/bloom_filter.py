"""Redis-backed bloom filter for cache penetration guard (PRD §4.8.4)."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.redis_client import RedisClient


class RedisBloomFilter:
    """Approximate set using Redis bitmap; stdlib-only hashing."""

    def __init__(
        self,
        redis: RedisClient,
        key: str,
        size: int = 1_000_000,
        num_hashes: int = 7,
    ) -> None:
        self._redis = redis
        self._key = key
        self._size = size
        self._num_hashes = num_hashes

    def _offsets(self, item: str) -> list[int]:
        digest = hashlib.sha256(item.encode("utf-8")).digest()
        offsets: list[int] = []
        for i in range(self._num_hashes):
            chunk = digest[i * 4:(i + 1) * 4]
            if len(chunk) < 4:
                chunk = (chunk + digest)[:4]
            value = int.from_bytes(chunk, "big")
            offsets.append(value % self._size)
        return offsets

    async def add(self, item: str) -> None:
        for offset in self._offsets(item):
            await self._redis.setbit(self._key, offset, 1)

    async def might_contain(self, item: str) -> bool:
        for offset in self._offsets(item):
            if not await self._redis.getbit(self._key, offset):
                return False
        return True


def get_tool_cache_bloom() -> RedisBloomFilter:
    from core.redis_client import redis_client

    return RedisBloomFilter(redis_client, key="bloom:tool_cache")
