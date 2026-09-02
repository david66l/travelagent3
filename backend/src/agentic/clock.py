"""Reference clock with an optional frozen moment for replayed frozen states.

Live production always uses the wall clock.  When a TRL environment replays
a frozen corpus decision state, date-relative evidence requirements (for
example the ten-day weather-forecast window) and freshness TTLs must be
evaluated against the moment the state was authored — otherwise a frozen
corpus silently expires as the calendar moves (H-002: a state whose trip
started eleven days after authoring stopped replaying the day the trip
entered the ten-day window, because weather suddenly became a required
artifact the recorded prefix never gathered).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime
from typing import Iterator

_frozen_reference_time: ContextVar[datetime | None] = ContextVar(
    "agentic_frozen_reference_time", default=None
)


def reference_now() -> datetime:
    """Current reference time: the frozen moment when one is active."""
    frozen = _frozen_reference_time.get()
    if frozen is not None:
        return frozen
    return datetime.now(UTC)


def reference_today() -> date:
    """Current reference date: the frozen date when one is active."""
    return reference_now().date()


@contextmanager
def frozen_reference_time(moment: datetime | None) -> Iterator[None]:
    """Pin the reference clock for the duration of the context.

    A ``None`` moment is a no-op so callers can pass an unresolved timestamp
    without branching.  The override is scoped to the current async context
    and always restored.
    """
    if moment is None:
        yield
        return
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    token = _frozen_reference_time.set(moment)
    try:
        yield
    finally:
        _frozen_reference_time.reset(token)
