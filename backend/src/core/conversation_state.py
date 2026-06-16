"""Session-level conversation state — DB-persisted, websocket-cached.

Every planning job carries a ``user_feedback`` JSON column that is the
authoritative state source.  The websocket layer holds a hot-cache copy.

Schema versioning lets us evolve the shape over time without breaking
in-flight sessions.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any

from schemas import ProfilePatch

# Increment when the state shape changes in a non-backwards-compatible way.
CURRENT_SCHEMA_VERSION = 1

# Keep at most this many recent messages in the conversation state.
MAX_RECENT_MESSAGES = 10

# Fields that hold a single value (overwrite on ``set``).
SCALAR_FIELDS = frozenset(
    {
        "destination",
        "travel_days",
        "travel_dates",
        "travelers_count",
        "travelers_type",
        "pace",
        "budget_range",
        "accommodation_preference",
    }
)

# Fields that hold lists (append / remove / clear).
LIST_FIELDS = frozenset(
    {
        "interests",
        "food_preferences",
        "avoid",
        "special_requests",
    }
)

# Fields that belong to the user's long-term personal profile (carry across trips).
PERSONAL_FIELDS = frozenset(
    {
        "food_preferences",
        "interests",
        "pace",
        "avoid",
        "accommodation_preference",
    }
)

# Fields that are scoped to the current trip (reset per trip).
TRIP_FIELDS = frozenset(
    {
        "destination",
        "travel_days",
        "travel_dates",
        "travelers_count",
        "travelers_type",
        "budget_range",
        "special_requests",
    }
)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def default_conversation_state() -> dict[str, Any]:
    """Return a fresh state for a brand-new conversation."""
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "recent_messages": [],
        "profile": {
            # Long-term personal preferences — carry across trips, rarely reset.
            "personal": {
                "food_preferences": [],
                "interests": [],
                "pace": "moderate",
                "avoid": [],
                "accommodation_preference": None,
            },
            # Trip-scoped preferences — reset when starting a new trip.
            "trip": {
                "destination": None,
                "travel_days": None,
                "travel_dates": None,
                "travelers_count": 1,
                "travelers_type": None,
                "budget_range": None,
                "special_requests": [],
            },
        },
        "phase": "gathering",
        "turn": 0,
        "last_intent": None,
        "missing_required": [],
        "revision": 1,
        "updated_at": int(time.time()),
    }


# --------------------------------------------------------------------------- #
# Profile merge
# --------------------------------------------------------------------------- #


def merge_profile(profile: dict[str, Any], patch: ProfilePatch) -> dict[str, Any]:
    """Apply a structured preference delta to *profile* and return a new dict.

    Each field is automatically routed to ``profile.personal`` or
    ``profile.trip`` based on :data:`PERSONAL_FIELDS` / :data:`TRIP_FIELDS`.

    ``patch.set``    → overwrite scalar / list values
    ``patch.add``    → append to list values (deduplicated, order-preserving)
    ``patch.remove`` → remove specific items from list values
    ``patch.clear``  → reset a field to its default
    """
    merged = deepcopy(profile)

    # Ensure personal / trip sub-objects exist (schema migration from flat profile)
    if "personal" not in merged:
        merged["personal"] = {
            "food_preferences": merged.get("food_preferences") or [],
            "interests": merged.get("interests") or [],
            "pace": merged.get("pace") or "moderate",
            "avoid": merged.get("avoid") or [],
            "accommodation_preference": merged.get("accommodation_preference"),
        }
    if "trip" not in merged:
        merged["trip"] = {
            "destination": merged.get("destination"),
            "travel_days": merged.get("travel_days"),
            "travel_dates": merged.get("travel_dates"),
            "travelers_count": merged.get("travelers_count", 1),
            "travelers_type": merged.get("travelers_type"),
            "budget_range": merged.get("budget_range"),
            "special_requests": merged.get("special_requests") or [],
        }

    def _target(key: str) -> dict:
        if key in PERSONAL_FIELDS:
            return merged["personal"]
        if key in TRIP_FIELDS:
            return merged["trip"]
        # Unknown fields fall back to trip scope
        return merged["trip"]

    for key, value in patch.set.items():
        target = _target(key)
        if key in SCALAR_FIELDS and value is not None:
            target[key] = value
        elif key in LIST_FIELDS and isinstance(value, list):
            target[key] = list(value)

    for key, values in patch.add.items():
        if key in LIST_FIELDS and isinstance(values, list):
            target = _target(key)
            existing: list = target.get(key, [])
            target[key] = _dedupe_keep_order(existing + values)

    for key, values in patch.remove.items():
        if key in LIST_FIELDS and isinstance(values, list):
            target = _target(key)
            existing: list = target.get(key, [])
            remove_set = {v.strip() for v in values}
            target[key] = [x for x in existing if x not in remove_set]

    for key in patch.clear:
        target = _target(key)
        if key in LIST_FIELDS:
            target[key] = []
        elif key in SCALAR_FIELDS:
            target[key] = None

    return merged


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        norm = item.strip()
        if norm and norm not in seen:
            result.append(norm)
            seen.add(norm)
    return result


# --------------------------------------------------------------------------- #
# Recent-messages helpers
# --------------------------------------------------------------------------- #


def append_message(
    state: dict[str, Any],
    role: str,
    content: str,
    *,
    max_messages: int = MAX_RECENT_MESSAGES,
) -> None:
    """Append a message and trim the history to *max_messages*."""
    state.setdefault("recent_messages", []).append(
        {
            "role": role,
            "content": content,
            "ts": int(time.time()),
        }
    )
    if len(state["recent_messages"]) > max_messages:
        state["recent_messages"] = state["recent_messages"][-max_messages:]


# --------------------------------------------------------------------------- #
# Convenience: is the profile ready for planning?
# --------------------------------------------------------------------------- #


def is_profile_ready(profile: dict[str, Any]) -> bool:
    """True when the minimum required trip fields are present."""
    trip = profile.get("trip", profile)  # tolerate flat pre-migration profiles
    return bool(trip.get("destination") and trip.get("travel_days"))


def flatten_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Merge personal + trip into a flat dict for downstream consumers (Pipeline, LLM).

    Trip fields take precedence over personal fields when both exist.
    """
    personal = profile.get("personal", {})
    trip = profile.get("trip", {})
    return {**personal, **trip}  # trip overwrites same-key personal fields (shouldn't overlap)
