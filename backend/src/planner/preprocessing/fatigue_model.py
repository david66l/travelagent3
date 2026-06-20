"""Daily walk-intensity budget based on cross-day fatigue accumulation."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class FatigueModel:
    """Compute per-day walk intensity limits given trip length and traveller type.

    Model (v4.0):
        fatigue_d = alpha * fatigue_{d-1} + day_walk_d
        where alpha is the recovery coefficient.

    If two consecutive days are high-intensity (>= 0.8 * base_walk),
    the third day is forced to low intensity (<= 0.4 * base_walk).
    """

    # Recovery coefficients per traveller type (v4.0 spec)
    _RECOVERY_RATES = {
        "family_elder": 0.70,
        "family_kid": 0.50,
        "adult": 0.35,
        "solo": 0.35,
        "couple": 0.35,
        "friends": 0.30,
        "young": 0.25,
    }

    def daily_walk_limits(self, constraints) -> list[int]:
        """Return a walk-intensity budget per day."""
        days = constraints.travel_days
        base = constraints.max_walk_km
        if base <= 0:
            base = 8

        alpha = constraints.fatigue_recovery_rate
        if alpha is None:
            alpha = self._RECOVERY_RATES.get(constraints.travelers_type, 0.35)

        limits: list[int] = []
        fatigue = 0.0
        high_streak = 0

        for d in range(days):
            # Effective walk capacity decreases with accumulated fatigue
            fatigue = alpha * fatigue
            effective_max = max(1.0, base - fatigue)

            # Force recovery day after 2 consecutive high-intensity days
            if high_streak >= 2:
                effective_max = min(effective_max, base * 0.4)
                high_streak = 0

            limit = max(1, int(effective_max))
            limits.append(limit)

            # Assume the day will be used at the allowed limit for fatigue propagation
            fatigue += limit
            if limit >= base * 0.8:
                high_streak += 1
            else:
                high_streak = 0

        return limits
