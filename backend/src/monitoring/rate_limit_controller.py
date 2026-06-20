"""Composite rate-limit and cost controller.

Aggregates per-IP, per-user, concurrent SSE, token quota, external API quota
and global cost-circuit checks into one interface used by middleware and
request handlers.

Redis failures are handled defensively: every check fails open (allows the
request) when Redis is unavailable, so a transient cache outage does not
hard-down the application.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from core.cost_circuit_breaker import (
    is_cost_circuit_active,
    record_daily_tokens,
    record_external_api_cost,
)
from core.exceptions import RateLimitException
from core.redis_client import redis_client as _redis_client
from core.settings import settings as _settings
from core.token_quota import check_and_record_tokens
from core.user_tier import tier_limits


class RateLimitCostController:
    """Rate-limit + cost guard with injectable Redis and settings."""

    def __init__(
        self,
        redis_client: Optional[Any] = None,
        settings: Optional[Any] = None,
    ) -> None:
        self.redis = redis_client or _redis_client
        self.settings = settings or _settings

    @staticmethod
    def _minute_bucket() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d%H%M")

    @staticmethod
    def _day_bucket() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d")

    async def check_ip_rate(self, ip: str) -> None:
        """Sliding-window per-minute IP rate limit.

        Uses a per-minute bucket counter. If Redis is unavailable the check
        is skipped (fail-open).
        """
        try:
            limit = self.settings.rate_limit_ip_per_minute
            key = f"rate:ip:{ip}:{self._minute_bucket()}"
            current = await self.redis.incr(key)
            if current == 1:
                await self.redis.expire(key, 120)
            if current > limit:
                raise RateLimitException(
                    "IP rate limit exceeded",
                    retry_after=60,
                )
        except RateLimitException:
            raise
        except Exception:
            # Redis unavailable → fail open.
            return

    async def check_user_rate(self, user_id: str, role: str) -> None:
        """Sliding-window per-minute user rate limit.

        Guests use ``rate_limit_guest_per_minute``. Authenticated users fall
        back to ``tier_limits(role).requests_per_minute`` when present, then
        ``rate_limit_user_per_minute``.
        """
        try:
            if role == "guest":
                limit = self.settings.rate_limit_guest_per_minute
            else:
                limits = tier_limits(role)
                limit = getattr(limits, "requests_per_minute", None)
                if limit is None:
                    limit = self.settings.rate_limit_user_per_minute

            key = f"rate:user:{user_id}:{self._minute_bucket()}"
            current = await self.redis.incr(key)
            if current == 1:
                await self.redis.expire(key, 120)
            if current > limit:
                raise RateLimitException(
                    "User rate limit exceeded",
                    retry_after=60,
                )
        except RateLimitException:
            raise
        except Exception:
            return

    async def check_concurrent_sse(self, user_id: str) -> None:
        """Limit concurrent Server-Sent Events connections per user."""
        try:
            limit = self.settings.rate_limit_max_concurrent_sse
            key = f"sse:user:{user_id}"
            current = await self.redis.incr(key)
            if current == 1:
                await self.redis.expire(key, 3600)
            if current > limit:
                raise RateLimitException(
                    "Concurrent SSE limit exceeded",
                    retry_after=60,
                )
        except RateLimitException:
            raise
        except Exception:
            return

    async def check_token_quota(self, user_id: str, role: str, estimated_tokens: int) -> None:
        """Delegate to the per-user daily LLM token quota."""
        if estimated_tokens <= 0:
            return
        try:
            await check_and_record_tokens(user_id, role, estimated_tokens)
        except RateLimitException:
            raise
        except Exception:
            return

    async def check_external_api_quota(self, user_id: str, role: str, calls: int = 1) -> None:
        """Per-user daily external API call quota.

        The limit is taken from ``tier_limits(role)`` first, then from the
        role-specific ``external_api_quota_{role}_daily`` setting.
        """
        if calls <= 0:
            return
        try:
            limits = tier_limits(role)
            limit = getattr(limits, "external_api_calls_daily", None)
            if limit is None:
                limit = getattr(limits, "daily_external_api_calls", None)
            if limit is None:
                fallback = getattr(self.settings, f"external_api_quota_{role}_daily", None)
                limit = fallback or self.settings.external_api_quota_free_daily

            key = f"ext_api:{user_id}:{self._day_bucket()}"
            total = await self.redis.incrby(key, calls)
            if total == calls:
                await self.redis.expire(key, 86400 + 3600)
            if total > limit:
                await self.redis.decrby(key, calls)
                raise RateLimitException(
                    "Daily external API quota exceeded",
                    retry_after=3600,
                )
        except RateLimitException:
            raise
        except Exception:
            return

    async def check_cost_circuit(self) -> None:
        """Raise when the global cost circuit breaker is active."""
        try:
            active = await is_cost_circuit_active()
        except RateLimitException:
            raise
        except Exception:
            return
        if active:
            raise RateLimitException("Global cost circuit active")

    async def check_request_allowed(
        self,
        user_id: str,
        role: str,
        *,
        ip: Optional[str] = None,
        concurrent_sse: bool = False,
        estimated_tokens: int = 0,
        external_api_cost: int = 0,
    ) -> None:
        """Run all guard layers in order and raise on the first violation.

        Order: IP → user → concurrent SSE → token quota → external API quota
        → global cost circuit.
        """
        if ip is not None:
            await self.check_ip_rate(ip)
        await self.check_user_rate(user_id, role)
        if concurrent_sse:
            await self.check_concurrent_sse(user_id)
        if estimated_tokens > 0:
            await self.check_token_quota(user_id, role, estimated_tokens)
        if external_api_cost > 0:
            await self.check_external_api_quota(user_id, role, external_api_cost)
        await self.check_cost_circuit()

    async def record_request(
        self,
        user_id: str,
        role: str,
        tokens: int = 0,
        api_cost: float = 0.0,
    ) -> None:
        """Record request cost for global cost tracking.

        Positive ``tokens`` update the daily token counter; positive
        ``api_cost`` (CNY) updates the daily external API cost counter.
        """
        if tokens > 0:
            await record_daily_tokens(tokens)
        if api_cost > 0.0:
            await record_external_api_cost(api_cost)
