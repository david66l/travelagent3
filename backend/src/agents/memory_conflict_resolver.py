"""Conflict resolver between short-term input and long-term profile."""

from __future__ import annotations

from typing import Any


class MemoryConflictResolver:
    """Resolve conflicts between short-term input and long-term memory.

    Rules:
    - Short-term explicit input wins for dietary taboos, people type, destination.
    - Physical constraints (walk/transit limits, special needs) use the most conservative value.
    - Preferences (pace, transport) use short-term when explicitly stated.
    """

    def resolve(
        self,
        short_term: dict[str, Any],
        long_term: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge short-term and long-term into a resolved profile dict."""
        resolved: dict[str, Any] = {}

        # Destination: explicit short-term wins.
        resolved["destination"] = short_term.get("destination") or long_term.get("destination")

        # Origin & travel dates: explicit short-term wins.
        resolved["origin"] = short_term.get("origin") or long_term.get("origin")
        resolved["travel_dates"] = short_term.get("travel_dates") or long_term.get("travel_dates")

        # Days: explicit short-term wins.
        resolved["travel_days"] = short_term.get("travel_days") or long_term.get("travel_days")

        # Budget: explicit short-term wins.
        resolved["total_budget"] = short_term.get("total_budget") or long_term.get("total_budget")

        # People count: explicit short-term wins.
        resolved["travelers_count"] = short_term.get("travelers_count") or long_term.get(
            "travelers_count"
        )

        # Companion type: explicit short-term wins.
        resolved["travel_companion"] = short_term.get("travel_companion") or long_term.get(
            "travel_companion"
        )

        # Dietary taboos: merge; short-term additions take priority.
        resolved["food_taboos"] = self._merge_lists(
            short_term.get("food_taboos", []),
            long_term.get("food_taboos", []),
        )

        # Food preferences: merge.
        resolved["food_prefs"] = self._merge_lists(
            short_term.get("food_prefs", []),
            long_term.get("food_preferences", long_term.get("food_prefs", [])),
        )

        # Interests: merge.
        resolved["interests"] = self._merge_lists(
            short_term.get("interests", []),
            long_term.get("interests", []),
        )

        # Pace: short-term explicit wins.
        resolved["pace"] = short_term.get("pace") or long_term.get("pace") or "moderate"

        # Transport preference: short-term explicit wins. "any" means unset.
        short_transport = short_term.get("transport_preference")
        long_transport = long_term.get("transport_mode")
        resolved["transport_preference"] = (
            short_transport
            if short_transport and short_transport != "any"
            else long_transport
            if long_transport and long_transport != "any"
            else None
        )

        # Physical constraints: conservative merge.
        resolved["max_walk_minutes"] = self._min_constraint(
            short_term.get("max_walk_minutes"),
            long_term.get("max_walk_minutes"),
            default=180,
        )
        resolved["max_transit_minutes"] = self._min_constraint(
            short_term.get("max_transit_minutes"),
            long_term.get("max_transit_minutes"),
            default=120,
        )

        # Special flags: unknown stays null; explicit values merge conservatively (true wins).
        for key in (
            "has_elderly",
            "has_children",
            "has_pregnant",
            "has_wheelchair",
            "avoid_crowds",
            "prefer_morning",
        ):
            short_val = short_term.get(key)
            long_val = long_term.get(key)
            if short_val is not None and long_val is not None:
                resolved[key] = bool(short_val) or bool(long_val)
            elif short_val is not None:
                resolved[key] = short_val
            elif long_val is not None:
                resolved[key] = long_val
            else:
                resolved[key] = None

        return resolved

    @staticmethod
    def _merge_lists(short: list[Any], long_term: list[Any]) -> list[Any]:
        """Merge lists, deduplicate, preserve short-term order first."""
        result: list[Any] = []
        seen: set[Any] = set()
        for item in list(short) + list(long_term):
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result

    @staticmethod
    def _min_constraint(short: Any, long_term: Any, default: int) -> int:
        """Pick the smallest (most conservative) constraint value."""
        values = [default]
        if short is not None:
            values.append(int(short))
        if long_term is not None:
            values.append(int(long_term))
        return min(values)
