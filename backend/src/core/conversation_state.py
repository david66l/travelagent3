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
SCALAR_FIELDS = frozenset({
    "destination",
    "travel_days",
    "travel_dates",
    "travelers_count",
    "travelers_type",
    "pace",
    "budget_range",
})

# Fields that hold lists (append / remove / clear).
LIST_FIELDS = frozenset({
    "interests",
    "food_preferences",
    "avoid",
    "special_requests",
})


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def default_conversation_state() -> dict[str, Any]:
    """Return a fresh state for a brand-new conversation."""
    return {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "recent_messages": [],
        "profile": {
            "destination": None,
            "travel_days": None,
            "travel_dates": None,
            "travelers_count": None,
            "travelers_type": None,
            "budget_range": None,
            "pace": None,
            "interests": [],
            "food_preferences": [],
            "avoid": [],
            "special_requests": [],
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

    ``patch.set``    → overwrite scalar / list values
    ``patch.add``    → append to list values (deduplicated, order-preserving)
    ``patch.remove`` → remove specific items from list values
    ``patch.clear``  → reset a field to its default
    """
    merged = deepcopy(profile)

    for key, value in patch.set.items():
        if key in SCALAR_FIELDS and value is not None:
            merged[key] = value
        elif key in LIST_FIELDS and isinstance(value, list):
            merged[key] = list(value)

    for key, values in patch.add.items():
        if key in LIST_FIELDS and isinstance(values, list):
            existing: list = merged.get(key, [])
            merged[key] = _dedupe_keep_order(existing + values)

    for key, values in patch.remove.items():
        if key in LIST_FIELDS and isinstance(values, list):
            existing: list = merged.get(key, [])
            remove_set = {v.strip() for v in values}
            merged[key] = [x for x in existing if x not in remove_set]

    for key in patch.clear:
        if key in LIST_FIELDS:
            merged[key] = []
        elif key in SCALAR_FIELDS:
            merged[key] = None

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
    state.setdefault("recent_messages", []).append({
        "role": role,
        "content": content,
        "ts": int(time.time()),
    })
    if len(state["recent_messages"]) > max_messages:
        state["recent_messages"] = state["recent_messages"][-max_messages:]


# --------------------------------------------------------------------------- #
# Convenience: is the profile ready for planning?
# --------------------------------------------------------------------------- #


def is_profile_ready(profile: dict[str, Any]) -> bool:
    """True when the minimum required fields are present."""
    return bool(profile.get("destination") and profile.get("travel_days"))
