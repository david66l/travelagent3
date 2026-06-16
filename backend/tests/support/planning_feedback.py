"""Shared helpers for integration tests."""

from __future__ import annotations

from typing import Any

from core.conversation_state import default_conversation_state, merge_profile
from schemas import ProfilePatch


def feedback_with_trip(
    destination: str,
    travel_days: int,
    travel_dates: str | None = None,
    user_input: str | None = None,
    **extra_trip: Any,
) -> dict[str, Any]:
    """Conversation state for ``PlanningJob.user_feedback`` (WS path → pipeline read)."""
    state = default_conversation_state()
    set_fields: dict[str, Any] = {
        "destination": destination,
        "travel_days": travel_days,
        **extra_trip,
    }
    if travel_dates:
        set_fields["travel_dates"] = travel_dates
    state["profile"] = merge_profile(state["profile"], ProfilePatch(set=set_fields))
    state["phase"] = "planning"
    if user_input:
        from core.conversation_state import append_message

        append_message(state, "user", user_input)
        state["turn"] = 1
    return state
