"""Tool base class: retry + timeout + cache + fallback."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from core.bloom_filter import get_tool_cache_bloom
from core.cache_keys import tool_cache_key
from core.cache_policy import jitter_ttl
from core.circuit_breaker import get_circuit_breaker
from core.local_cache import tool_local_cache
from core.redis_client import redis_client
from core.settings import settings
from schemas import ToolResult

logger = logging.getLogger(__name__)


class Tool(ABC):
    """External tool abstraction with unified retry/cache/fallback semantics.

    Subclasses implement ``execute(params)`` and optionally ``_fallback(params)``.
    ``run(params)`` orchestrates cache lookup, execution with retries, and
    fallback handling.
    """

    name: str = ""
    timeout: float = settings.tool_timeout_seconds
    retries: int = settings.tool_max_retries
    cache_ttl: int = 3600
    backoff_base: float = 0.5

    @abstractmethod
    async def execute(self, params: dict) -> Any | ToolResult:
        """Primary tool implementation. May raise on failure."""

    async def _fallback(self, params: dict, last_error: Optional[Exception]) -> ToolResult:
        """Default fallback: mark data unavailable."""
        reason = f"all {self.retries} retries exhausted"
        if last_error:
            reason += f": {type(last_error).__name__}"
        return ToolResult(
            data=None,
            data_source="unavailable",
            is_fallback=True,
            fallback_reason=reason,
        )

    def _cache_key(self, params: dict) -> str:
        return tool_cache_key(self.name, params)

    def _negative_cache_key(self, cache_key: str) -> str:
        return f"neg:{cache_key}"

    async def _read_negative_cache(self, cache_key: str) -> Optional[ToolResult]:
        try:
            cached = await redis_client.get_json(self._negative_cache_key(cache_key))
            if cached:
                return ToolResult(**cached)
        except Exception:
            pass
        return None

    async def _set_negative_cache(self, cache_key: str, result: ToolResult) -> None:
        try:
            await redis_client.set_json(
                self._negative_cache_key(cache_key),
                result.model_dump(),
                ttl=60,
            )
            bloom = get_tool_cache_bloom()
            await bloom.add(cache_key)
        except Exception:
            pass

    async def _read_l1_cache(self, cache_key: str) -> Optional[ToolResult]:
        try:
            cached = await tool_local_cache.get(cache_key)
            if cached:
                return ToolResult(**cached)
        except Exception:
            pass
        return None

    async def _set_l1_cache(self, cache_key: str, result: ToolResult) -> None:
        try:
            await tool_local_cache.set(cache_key, result.model_dump())
        except Exception:
            pass

    def _effective_ttl(self) -> int:
        return jitter_ttl(self.cache_ttl, settings.cache_ttl_jitter_ratio)

    async def _mark_bloom(self, cache_key: str) -> None:
        try:
            bloom = get_tool_cache_bloom()
            await bloom.add(cache_key)
        except Exception:
            pass

    async def _read_cache(self, cache_key: str) -> Optional[ToolResult]:
        try:
            cached = await redis_client.get_json(cache_key)
            if cached:
                return ToolResult(**cached)
        except Exception:
            pass
        return None

    async def _set_cache(self, cache_key: str, result: ToolResult) -> None:
        if result.data_source == "unavailable":
            await self._set_negative_cache(cache_key, result)
            return
        try:
            await redis_client.set_json(
                cache_key,
                result.model_dump(),
                ttl=self._effective_ttl(),
            )
            await self._set_l1_cache(cache_key, result)
            await self._mark_bloom(cache_key)
        except Exception:
            pass

    async def _acquire_cache_lock(self, cache_key: str):
        """Return lock instance if acquired, else None."""
        lock = redis_client.lock(f"cache_lock:{cache_key}", ttl=30, blocking_timeout=5)
        try:
            if await lock.acquire():
                return lock
        except Exception:
            pass
        return None

    async def _release_cache_lock(self, lock) -> None:
        if lock is None:
            return
        try:
            await lock.release()
        except Exception:
            pass

    async def run(self, params: dict) -> ToolResult:
        """Execute the tool with caching, retries and fallback."""
        breaker = get_circuit_breaker(self.name) if self.name else None
        if breaker and breaker.is_open():
            fb = await self._fallback(params, None)
            if fb.fallback_reason:
                fb.fallback_reason = f"{fb.fallback_reason}; circuit open"
            else:
                fb.fallback_reason = "circuit open"
            return fb

        cache_key = self._cache_key(params)

        result = await self._read_l1_cache(cache_key)
        if result:
            return result

        result = await self._read_cache(cache_key)
        if result:
            await self._set_l1_cache(cache_key, result)
            return result

        neg = await self._read_negative_cache(cache_key)
        if neg:
            return neg

        try:
            bloom = get_tool_cache_bloom()
            if not await bloom.might_contain(cache_key):
                pass  # first-time key; proceed to fetch
        except Exception:
            pass

        cache_lock = await self._acquire_cache_lock(cache_key)
        if cache_lock:
            result = await self._read_l1_cache(cache_key)
            if result:
                await self._release_cache_lock(cache_lock)
                return result
            result = await self._read_cache(cache_key)
            if result:
                await self._set_l1_cache(cache_key, result)
                await self._release_cache_lock(cache_lock)
                return result
            neg = await self._read_negative_cache(cache_key)
            if neg:
                await self._release_cache_lock(cache_lock)
                return neg

        start = time.monotonic()
        last_exc: Optional[Exception] = None

        try:
            for attempt in range(self.retries + 1):
                try:
                    raw = await asyncio.wait_for(self.execute(params), timeout=self.timeout)
                    latency = int((time.monotonic() - start) * 1000)

                    if isinstance(raw, ToolResult):
                        result = raw
                        if result.latency_ms == 0:
                            result.latency_ms = latency
                        if result.retries == 0:
                            result.retries = attempt
                    else:
                        result = ToolResult(
                            data=raw,
                            data_source="api",
                            confidence=1.0,
                            latency_ms=latency,
                            retries=attempt,
                        )

                    await self._set_cache(cache_key, result)
                    if breaker:
                        breaker.record_success()
                    return result
                except asyncio.TimeoutError as exc:
                    last_exc = exc
                    logger.warning("%s timeout (attempt %d)", self.name, attempt)
                except Exception as exc:
                    last_exc = exc
                    logger.warning("%s error (attempt %d): %s", self.name, attempt, exc)

                if attempt < self.retries:
                    await asyncio.sleep(self.backoff_base * (2**attempt))

            fallback = await self._fallback(params, last_exc)
            fallback.latency_ms = int((time.monotonic() - start) * 1000)
            fallback.retries = self.retries
            await self._set_cache(cache_key, fallback)
            if breaker:
                breaker.record_failure()
            return fallback
        finally:
            await self._release_cache_lock(cache_lock)
