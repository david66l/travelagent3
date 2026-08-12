"""External API usage and cost tracking (PRD §4.10.3 / §4.10.4)."""

from __future__ import annotations

from datetime import datetime, timezone

from core.cost_circuit_breaker import record_external_api_cost
from core.metrics import record_external_api_call, record_external_api_cost_cny
from core.redis_client import redis_client
from core.settings import settings
from core.user_tier import tier_limits


# Estimated unit cost in CNY per successful call (tunable via settings later)
_API_UNIT_COST = {
    "tavily": 0.02,
    "amap": 0.01,
    "weather": 0.005,
    "search": 0.02,
}


def _day_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _user_api_key(user_id: str) -> str:
    return f"external_api:user:{user_id}:{_day_suffix()}"


async def check_external_api_quota(user_id: str, role: str) -> bool:
    """Return True if the user may make another external API call."""
    limits = tier_limits(role)
    if limits.daily_external_api_calls >= 10_000_000:
        return True
    raw = await redis_client.get(_user_api_key(user_id))
    try:
        used = int(raw or 0)
    except (TypeError, ValueError):
        used = 0
    return used < limits.daily_external_api_calls


async def record_external_api_usage(
    api_name: str,
    *,
    status: str = "success",
    data_source: str = "api",
    user_id: str | None = None,
    role: str | None = None,
) -> None:
    record_external_api_call(api_name, status=status, data_source=data_source)

    if status != "success":
        return

    unit_cost = _API_UNIT_COST.get(api_name, settings.external_api_default_cost_cny)
    await record_external_api_cost(unit_cost)
    record_external_api_cost_cny(api_name, unit_cost)

    if user_id and role:
        key = _user_api_key(user_id)
        count = await redis_client.incr(key)
        if count == 1:
            await redis_client.expire(key, 86400 + 3600)
