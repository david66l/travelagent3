"""Redis-persisted Celery task retry counters (PRD §4.7.3)."""

from __future__ import annotations

import random
from typing import Type

from core.redis_client import redis_client
from core.settings import settings

_RETRYABLE_EXCEPTIONS: tuple[Type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
)

_RETRY_KEY_PREFIX = "task_retry:"


def _retry_key(task_name: str, job_id: str) -> str:
    return f"{_RETRY_KEY_PREFIX}{task_name}:{job_id}"


async def get_retry_count(task_name: str, job_id: str) -> int:
    raw = await redis_client.get(_retry_key(task_name, job_id))
    if raw is None:
        return 0
    try:
        return int(raw)
    except ValueError:
        return 0


async def increment_retry(task_name: str, job_id: str) -> int:
    key = _retry_key(task_name, job_id)
    count = await redis_client.incr(key)
    await redis_client.expire(key, settings.task_retry_counter_ttl_seconds)
    return count


async def reset_retry(task_name: str, job_id: str) -> None:
    await redis_client.delete(_retry_key(task_name, job_id))


def is_retryable(exc: BaseException) -> bool:
    """Non-retryable: validation, permission, user cancel semantics."""
    if isinstance(exc, (ValueError, PermissionError, KeyError, TypeError)):
        return False
    if isinstance(exc, _RETRYABLE_EXCEPTIONS):
        return True
    name = type(exc).__name__
    if name in ("TimeoutError", "ConnectError", "ReadTimeout", "ConnectTimeout"):
        return True
    # LLM / external API transient failures often surface as RuntimeError
    return True


def compute_retry_delay(
    attempt: int,
    *,
    initial: float | None = None,
    max_delay: float | None = None,
    jitter: bool = True,
) -> float:
    """Exponential backoff with optional jitter (PRD: 2s initial, 30s cap)."""
    base = initial if initial is not None else settings.planning_task_retry_initial_seconds
    cap = max_delay if max_delay is not None else settings.planning_task_retry_max_seconds
    delay = min(base * (2 ** max(attempt - 1, 0)), cap)
    if jitter:
        delay = delay * (0.5 + random.random())
    return max(delay, 0.1)
