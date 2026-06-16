"""Tests for cache TTL jitter."""

from core.cache_policy import jitter_ttl


def test_jitter_ttl_within_band():
    base = 3600
    samples = {jitter_ttl(base, ratio=0.1) for _ in range(50)}
    assert all(3240 <= v <= 3960 for v in samples)
    assert len(samples) > 1
