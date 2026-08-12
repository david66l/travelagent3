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

    # Real recommended dwell time for large landmarks (minutes):
    #   keyword -> (default, min, max).
    # POIInput defaults are duration=60/120, min=15, max=240 (4h). Theme parks
    # and large attractions need a full/half day, otherwise the solver schedules
    # "迪士尼 08:00-12:00" (4h) and wastes a ¥475 ticket, or crams a far-suburb
    # zoo into a slot too short to reach. Matched case-insensitively by name.
    # Theme parks default to a full day extending into the evening so the花车/烟花
    # core experience is not cut (08:00 + 12h ⇒ ~20:00; the day_end is 21:00). Large
    # gardens/zoos/science museums are bumped to match real touring time (a 3h slot
    # left visitors rushing 野生动物园/辰山/科技馆).
    LANDMARK_DURATIONS: dict[str, tuple[int, int, int]] = {
        "迪士尼": (720, 540, 780),
        "disney": (720, 540, 780),
        "环球影城": (720, 540, 780),
        "universal": (720, 540, 780),
        "长隆": (600, 420, 720),
        "欢乐谷": (420, 330, 540),
        "方特": (420, 330, 540),
        "野生动物园": (390, 300, 480),
        "动物园": (240, 180, 330),
        "safari": (390, 300, 480),
        "zoo": (240, 180, 330),
        "海昌": (360, 300, 480),
        "海洋公园": (360, 300, 480),
        "ocean park": (360, 300, 480),
        "水族馆": (180, 120, 270),
        "aquarium": (180, 120, 270),
        "植物园": (210, 150, 300),
        "科技馆": (210, 150, 300),
        "博物馆": (150, 90, 210),
        "museum": (150, 90, 210),
    }

    # Real-world opening hours by category (open, close). POIInput defaults to
    # 08:00-18:00 and RAG/AMap rarely supply real hours, so museums/galleries/zoos
    # were being scheduled at 08:00 — when they are still closed (中华艺术宫 opens
    # 10:00, 博物馆/动物园 09:00). Matched by name keyword, most-specific first.
    # Theme parks are intentionally absent: they are widened to 08:00-21:30 above.
    # Open-air spots (步行街/路/公园/外滩/古镇) are absent too — 08:00 is fine.
    CATEGORY_HOURS: dict[str, tuple[str, str]] = {
        "野生动物园": ("09:00", "17:00"),
        "动物园": ("09:00", "17:00"),
        "safari": ("09:00", "17:00"),
        "zoo": ("09:00", "17:00"),
        "水族馆": ("09:00", "18:00"),
        "aquarium": ("09:00", "18:00"),
        "海洋馆": ("09:00", "18:00"),
        "科技馆": ("09:00", "17:15"),
        "艺术宫": ("10:00", "18:00"),
        "美术馆": ("10:00", "18:00"),
        "art museum": ("10:00", "18:00"),
        # 上海博物馆东馆: 10:00-18:00 (closes Tuesday) — must precede the generic
        # 博物馆 entry so first-match wins.
        "博物馆东馆": ("10:00", "18:00"),
        "上博东馆": ("10:00", "18:00"),
        "博物馆": ("09:00", "17:00"),
        "museum": ("09:00", "17:00"),
        "植物园": ("08:00", "17:00"),
        "豫园": ("08:45", "16:45"),
        # Performance venues: little to see before ~09:00 (exterior only) and the
        # real draw is an evening show — keep them off the 08:00 day-opener slot.
        "大剧院": ("09:00", "21:30"),
        "音乐厅": ("09:00", "21:30"),
        "歌剧院": ("09:00", "21:30"),
        "寺": ("07:30", "17:00"),
        "庙": ("08:00", "17:00"),
        "园林": ("08:30", "17:00"),
    }

    def _landmark_profile(self, name: str) -> tuple[int, int, int] | None:
        """Return (default, min, max) minutes if the POI name matches a landmark."""
        low = (name or "").lower()
        for keyword, profile in self.LANDMARK_DURATIONS.items():
            if keyword in low:
                return profile
        return None

    # Weekly closing day by category (0=Mon … 6=Sun). Chinese museums / galleries /
    # science & exhibition halls almost universally close Monday (周一闭馆); a few
    # close Tuesday (e.g. 上海博物馆东馆) but Monday is the safe general default.
    # Open-air spots, temples, zoos, gardens, aquariums run daily → absent here.
    CATEGORY_CLOSED: dict[str, list[int]] = {
        "艺术宫": [0],
        "美术馆": [0],
        "art museum": [0],
        # 上海博物馆东馆 closes Tuesday (1), not Monday — specific keys first so
        # they win over the generic 博物馆 → Monday rule below.
        "博物馆东馆": [1],
        "上博东馆": [1],
        "博物馆": [0],
        "museum": [0],
        "科技馆": [0],
        "纪念馆": [0],
        "展览馆": [0],
        "陈列馆": [0],
    }

    def _category_hours(self, name: str) -> tuple[str, str] | None:
        """Return (open, close) if the POI name matches a known category."""
        low = (name or "").lower()
        for keyword, hours in self.CATEGORY_HOURS.items():
            if keyword in low:
                return hours
        return None

    def _category_closed(self, name: str) -> list[int]:
        """Return the weekdays (0=Mon) the POI is closed, by name keyword."""
        low = (name or "").lower()
        for keyword, days in self.CATEGORY_CLOSED.items():
            if keyword in low:
                return list(days)
        return []

    def adjust(
        self,
        pois: list[POIInput],
        constraints,
    ) -> list[POIInput]:
        """Return adjusted POIs with v4.0 play-mode intervalization."""
        mode: Literal["quick", "standard", "deep"] = constraints.play_mode or "standard"
        for p in pois:
            # 0. Apply real landmark dwell time before clamping, so theme parks /
            #    large attractions get a full/half-day block instead of the
            #    generic 60-120min default.
            landmark = self._landmark_profile(p.name)
            if landmark is not None:
                lm_default, lm_min, lm_max = landmark
                p.min_play_time = max(p.min_play_time, lm_min)
                p.max_play_time = max(p.max_play_time, lm_max)
                if p.duration_minutes < lm_default:
                    p.duration_minutes = lm_default
                # Theme-park tier (>=9h): the dwell time only fits if the venue's
                # window spans the whole day. Real parks open ~08:00 and run花车/
                # 烟花 to ~21:00, so widen the window to 08:00-21:30; otherwise the
                # default 18:00 close makes A[i]+duration <= close infeasible and
                # the ¥475 landmark is silently dropped.
                if lm_default >= 540:
                    if _time_minutes(p.open_time) > 8 * 60:
                        p.open_time = "08:00"
                    if _time_minutes(p.close_time) < 21 * 60 + 30:
                        p.close_time = "21:30"

            # 0b. Real opening hours by category (only for non-theme-park venues;
            #     theme parks were just widened above). Never schedule before the
            #     venue opens; never run past close. This kills "中华艺术宫 08:00".
            if not (landmark is not None and landmark[0] >= 540):
                hours = self._category_hours(p.name)
                if hours is not None:
                    h_open, h_close = hours
                    if _time_minutes(p.open_time) < _time_minutes(h_open):
                        p.open_time = h_open
                    if _time_minutes(p.close_time) > _time_minutes(h_close):
                        p.close_time = h_close

            # 0c. Weekly closing day (周一闭馆). Only fill from the category table
            #     when real data did not already provide it.
            if not p.closed_weekdays:
                closed = self._category_closed(p.name)
                if closed:
                    p.closed_weekdays = closed

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
            # Night-view POIs (观景台/外滩等) carry a 夜景/观景 tag but rarely an
            # explicit best_visit_period. Without one they get scheduled at 08:00
            # and the "俯瞰璀璨夜景" promise is broken. Default them to evening so
            # the window-narrowing below pushes them to ≥16:00. Pure-观景 daytime
            # towers also benefit (sunset is their best slot).
            if not period:
                tagset = {t for t in (p.tags or [])}
                if "夜景" in tagset or "观景" in tagset:
                    period = "evening"
                    p.best_visit_period = "evening"
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
