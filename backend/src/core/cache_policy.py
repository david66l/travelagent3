"""Cache TTL helpers (jitter to reduce avalanche risk)."""

from __future__ import annotations

import random


def jitter_ttl(base: int, ratio: float = 0.1) -> int:
    """Return base TTL with small random jitter (±ratio)."""
    if base <= 0:
        return base
    delta = max(1, int(base * ratio))
    return max(1, base + random.randint(-delta, delta))
