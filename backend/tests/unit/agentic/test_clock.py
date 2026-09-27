"""Tests for the frozen reference clock (H-002)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from agentic.clock import frozen_reference_time, reference_now, reference_today


def test_default_reference_clock_follows_the_wall_clock():
    before = datetime.now(UTC)
    now = reference_now()
    after = datetime.now(UTC)
    assert before <= now <= after
    assert reference_today() == datetime.now(UTC).date()


def test_frozen_context_pins_now_and_today_and_restores():
    frozen = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    with frozen_reference_time(frozen):
        assert reference_now() == frozen
        assert reference_today() == date(2026, 9, 1)
    assert reference_now() >= datetime.now(UTC) - timedelta(seconds=5)


def test_frozen_context_accepts_none_and_naive_moments():
    with frozen_reference_time(None):
        assert reference_now() >= datetime.now(UTC) - timedelta(seconds=5)

    naive = datetime(2026, 9, 1, 8, 30)
    with frozen_reference_time(naive):
        assert reference_now() == naive.replace(tzinfo=UTC)


def test_weather_window_uses_the_frozen_date_when_replaying():
    """H-002 regression: a trip authored eleven days out must not suddenly
    require a weather artifact once the wall clock crosses into the
    ten-day forecast window."""
    from agentic.react import infer_research_requirements
    from agentic.state import GoalLedger

    authored_at = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)
    start_date = (authored_at.date() + timedelta(days=11)).isoformat()
    goal = GoalLedger(
        original_request="Plan a trip",
        hard_constraints={"travel_days": 3, "start_date": start_date},
    )

    # Inside a frozen replay the authored distance (11 days) applies.
    with frozen_reference_time(authored_at):
        replayed = infer_research_requirements(goal)
    assert "weather_snapshot" not in replayed.required_artifact_types
    assert replayed.requires_weather is False

    # Against today's wall clock the same trip may sit inside the window;
    # the live behaviour stays calendar-relative and unchanged.
    live = infer_research_requirements(goal)
    delta_days = (date.fromisoformat(start_date) - datetime.now(UTC).date()).days
    assert live.requires_weather is (0 <= delta_days <= 10)
