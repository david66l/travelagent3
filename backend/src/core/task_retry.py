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

_RETRYABLE_EXCEPTION_NAMES = {
    "ConnectError",
    "ConnectTimeout",
    "ConnectionError",
    "ConnectionResetError",
    "DisconnectionError",
    "InterfaceError",
    "OperationalError",
    "ReadTimeout",
    "RedisError",
    "TimeoutError",
}


class RetryableTaskError(RuntimeError):
    """An explicitly classified transient failure safe to execute again."""


class NonRetryableTaskError(RuntimeError):
    """A deterministic task failure that must not be replayed."""


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
    """Return true only for explicit or recognisable transient failures.

    Unknown programming errors default to non-retryable. Retrying every
    ``RuntimeError`` can repeat costly model/tool side effects and hide defects.
    Wrapped timeout/connection/database errors are still detected through the
    exception cause chain.
    """
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, NonRetryableTaskError):
            return False
        if isinstance(current, RetryableTaskError):
            return True
        if isinstance(current, (ValueError, PermissionError, KeyError, TypeError)):
            return False
        if isinstance(current, _RETRYABLE_EXCEPTIONS):
            return True
        if type(current).__name__ in _RETRYABLE_EXCEPTION_NAMES:
            return True
        current = current.__cause__ or current.__context__
    return False


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
