"""Adjust POI play time / windows according to play_mode and best-visit period."""

from __future__ import annotations

import logging
from typing import Literal

from vrp_solver_service.models import POIInput

logger = logging.getLogger(__name__)


def _time_minutes(time_str: str) -> int:
    try:
        h, m = map(int, time_str.split(":"))
        return h * 60 + m
    except Exception:
        return 0


class PlayTimeManager:
    """Apply play-mode duration scaling and time-of-day window heuristics."""

    def adjust(
        self,
        pois: list[POIInput],
        constraints,
    ) -> list[POIInput]:
        """Return adjusted POIs with v4.0 play-mode intervalization."""
        mode: Literal["quick", "standard", "deep"] = constraints.play_mode or "standard"
        for p in pois:
            # 1. Duration intervalization by play_mode (v4.0 table)
            base = p.duration_minutes
            if mode == "quick":
                p.duration_minutes = max(p.min_play_time, 15)
            elif mode == "deep":
                p.duration_minutes = min(p.max_play_time, 480)
            else:  # standard: w_i (clamped to [min, max] for safety)
                p.duration_minutes = max(p.min_play_time, min(p.max_play_time, base))

            # 2. Time-of-day window heuristics
            period = (p.best_visit_period or "").lower()
            if not period:
                continue
            open_min = _time_minutes(p.open_time)
            close_min = _time_minutes(p.close_time)
            if "sunset" in period or "evening" in period:
                open_min = max(open_min, 16 * 60)
            elif "morning" in period or "sunrise" in period:
                close_min = min(close_min, 12 * 60)
            elif "night" in period:
                open_min = max(open_min, 18 * 60)
                close_min = min(close_min, 23 * 60)
            p.open_time = f"{open_min // 60:02d}:{open_min % 60:02d}"
            p.close_time = f"{close_min // 60:02d}:{close_min % 60:02d}"
        return pois
