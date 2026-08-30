"""UTC clock helpers with explicit timezone semantics."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return an aware UTC timestamp for timezone-aware database columns."""
    return datetime.now(UTC)


def utc_now_naive() -> datetime:
    """Return UTC without tzinfo for legacy TIMESTAMP WITHOUT TIME ZONE columns."""
    return datetime.now(UTC).replace(tzinfo=None)
