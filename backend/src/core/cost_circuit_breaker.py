"""Global cost circuit breaker (PRD §4.10.6)."""

from __future__ import annotations

from datetime import datetime, timezone

from core.redis_client import redis_client
from core.settings import settings
from core.metrics import set_daily_cost_gauges

_DAY_KEY = "cost:day"
_HOUR_GPU_KEY = "cost:gpu:hour"


def _day_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _hour_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H")


async def record_daily_tokens(tokens: int) -> int:
    if tokens <= 0:
        return await get_daily_tokens()
    key = f"{_DAY_KEY}:tokens:{_day_suffix()}"
    total = await redis_client.incrby(key, tokens)
    if total == tokens:
        await redis_client.expire(key, 86400 + 3600)
    api_cost = await get_daily_external_api_cost()
    set_daily_cost_gauges(int(total), api_cost)
    return int(total)


async def get_daily_tokens() -> int:
    raw = await redis_client.get(f"{_DAY_KEY}:tokens:{_day_suffix()}")
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


async def record_external_api_cost(cny: float) -> float:
    if cny <= 0:
        return await get_daily_external_api_cost()
    key = f"{_DAY_KEY}:api_cost:{_day_suffix()}"
    # Store as milli-CNY to avoid float in Redis
    milli = int(round(cny * 1000))
    total_milli = await redis_client.incrby(key, milli)
    if total_milli == milli:
        await redis_client.expire(key, 86400 + 3600)
    tokens = await get_daily_tokens()
    set_daily_cost_gauges(tokens, total_milli / 1000.0)
    return total_milli / 1000.0


async def get_daily_external_api_cost() -> float:
    raw = await redis_client.get(f"{_DAY_KEY}:api_cost:{_day_suffix()}")
    try:
        return int(raw or 0) / 1000.0
    except (TypeError, ValueError):
        return 0.0


async def record_hourly_gpu_cost(cny: float) -> float:
    if cny <= 0:
        return await get_hourly_gpu_cost()
    key = f"{_HOUR_GPU_KEY}:{_hour_suffix()}"
    milli = int(round(cny * 1000))
    total_milli = await redis_client.incrby(key, milli)
    if total_milli == milli:
        await redis_client.expire(key, 7200)
    return total_milli / 1000.0


async def get_hourly_gpu_cost() -> float:
    raw = await redis_client.get(f"{_HOUR_GPU_KEY}:{_hour_suffix()}")
    try:
        return int(raw or 0) / 1000.0
    except (TypeError, ValueError):
        return 0.0


async def is_cost_circuit_active() -> bool:
    """True when any global cost threshold is exceeded."""
    if not settings.cost_circuit_breaker_enabled:
        return False

    manual = await redis_client.get("cost:circuit:manual")
    if manual == "1":
        return True

    tokens = await get_daily_tokens()
    if tokens > settings.cost_circuit_breaker_daily_tokens:
        return True

    api_cost = await get_daily_external_api_cost()
    if api_cost > settings.cost_circuit_breaker_daily_api_cost_cny:
        return True

    gpu_cost = await get_hourly_gpu_cost()
    if gpu_cost > settings.cost_circuit_breaker_hourly_gpu_cost_cny:
        return True

    return False


async def clear_manual_circuit() -> None:
    await redis_client.delete("cost:circuit:manual")
