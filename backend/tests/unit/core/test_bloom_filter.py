"""Tests for Redis bloom filter."""

from unittest.mock import AsyncMock

import pytest

from core.bloom_filter import RedisBloomFilter


@pytest.mark.asyncio
async def test_bloom_filter_add_and_check(mock_redis):
    bits: dict[int, int] = {}

    async def setbit(key, offset, value):
        bits[offset] = value
        return 0

    async def getbit(key, offset):
        return bits.get(offset, 0)

    mock_redis.setbit = AsyncMock(side_effect=setbit)
    mock_redis.getbit = AsyncMock(side_effect=getbit)

    bloom = RedisBloomFilter(mock_redis, key="bloom:test", size=1000, num_hashes=5)
    assert await bloom.might_contain("beijing-poi") is False
    await bloom.add("beijing-poi")
    assert await bloom.might_contain("beijing-poi") is True
