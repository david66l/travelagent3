"""Per-user daily LLM token quota (PRD §4.3 / §4.10.7)."""

from __future__ import annotations

from datetime import datetime, timezone

from core.exceptions import RateLimitException
from core.redis_client import redis_client
from core.user_tier import tier_limits


def _quota_key(user_id: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"llm_tokens:{user_id}:{day}"


async def check_and_record_tokens(user_id: str, role: str, tokens: int) -> None:
    """Raise RateLimitException when daily quota would be exceeded."""
    if tokens <= 0:
        return
    limit = tier_limits(role).daily_tokens
    key = _quota_key(user_id)
    new_total = await redis_client.incrby(key, tokens)
    if new_total == tokens:
        await redis_client.expire(key, 86400 + 3600)
    if new_total > limit:
        await redis_client.decrby(key, tokens)
        raise RateLimitException(
            f"Daily LLM token quota exceeded ({limit})",
            retry_after=3600,
        )


async def get_daily_usage(user_id: str) -> int:
    raw = await redis_client.get(_quota_key(user_id))
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0
