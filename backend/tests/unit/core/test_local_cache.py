"""Tests for in-process L1 TTL cache."""

import pytest

from core.local_cache import LocalTTLCache


@pytest.mark.asyncio
async def test_local_ttl_cache_get_set():
    cache = LocalTTLCache[str](ttl_seconds=60, maxsize=10)
    await cache.set("k1", "value")
    assert await cache.get("k1") == "value"
    await cache.delete("k1")
    assert await cache.get("k1") is None
