"""In-memory circuit breaker for tool degradation.

Can be swapped for a Redis-backed implementation later for multi-instance
consistency. Tracks failures in a sliding window and opens after the
configured failure threshold.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from core.settings import settings


@dataclass
class CircuitBreaker:
    """Simple sliding-window circuit breaker."""

    name: str
    failure_threshold: float = settings.circuit_breaker_failure_threshold
    min_failures: int = settings.circuit_breaker_min_failures
    recovery_seconds: int = settings.circuit_breaker_recovery_seconds
    window_seconds: int = settings.circuit_breaker_window_seconds
    _successes: list[float] = field(default_factory=list, repr=False)
    _failures: list[float] = field(default_factory=list, repr=False)
    _opened_at: Optional[float] = field(default=None, repr=False)

    def is_open(self) -> bool:
        """Return True if the breaker is currently open."""
        now = time.monotonic()
        self._trim(now)

        if self._opened_at is not None:
            if now - self._opened_at < self.recovery_seconds:
                return True
            # Transition to half-open: clear opened_at and let one probe through
            self._opened_at = None

        return False

    def record_success(self) -> None:
        now = time.monotonic()
        self._trim(now)
        if self.is_open():
            return
        self._successes.append(now)

    def record_failure(self) -> None:
        now = time.monotonic()
        self._trim(now)
        if self.is_open():
            return
        self._failures.append(now)
        if self._should_open():
            self._opened_at = now

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        self._successes = [t for t in self._successes if t > cutoff]
        self._failures = [t for t in self._failures if t > cutoff]

    def _should_open(self) -> bool:
        failures = len(self._failures)
        successes = len(self._successes)
        total = failures + successes
        if failures < self.min_failures:
            return False
        if total == 0:
            return False
        return failures / total >= self.failure_threshold

    def state(self) -> str:
        if self.is_open():
            return "OPEN"
        return "CLOSED"


_breakers: dict[str, CircuitBreaker] = {}


def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Return a shared circuit breaker instance for a tool name."""
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name)
    return _breakers[name]
