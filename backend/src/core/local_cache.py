"""In-process L1 TTL cache (PRD §4.8 — hot POI / route keys, 5min)."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Generic, Optional, TypeVar

from core.settings import settings

T = TypeVar("T")


class LocalTTLCache(Generic[T]):
    """Simple async-safe TTL cache for a single process."""

    def __init__(self, ttl_seconds: int = 300, maxsize: int = 2048) -> None:
        self._ttl = ttl_seconds
        self._maxsize = maxsize
        self._store: dict[str, tuple[T, float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[T]:
        now = time.monotonic()
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at <= now:
                del self._store[key]
                return None
            return value

    async def set(self, key: str, value: T) -> None:
        expires_at = time.monotonic() + self._ttl
        async with self._lock:
            if len(self._store) >= self._maxsize and key not in self._store:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
            self._store[key] = (value, expires_at)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()


# Shared L1 for tool results (PRD: 5min for hot POI / routing)
tool_local_cache: LocalTTLCache[dict] = LocalTTLCache(
    ttl_seconds=settings.local_cache_ttl_seconds,
    maxsize=2048,
)
