"""Transport mode selection and travel time/cost matrix generation."""

from __future__ import annotations

import logging
import math
from typing import Literal

from vrp_solver_service.models import POIInput

logger = logging.getLogger(__name__)


class TransportSelector:
    """Select transport mode per leg and build travel time/cost matrices."""

    MODE_PROFILES: dict[str, tuple[float, float, float]] = {
        "walk": (4.5, 0.0, 0.0),
        "subway": (25.0, 2.0, 0.5),
        "bus": (20.0, 1.5, 0.5),
        "taxi": (40.0, 2.5, 0.0),
        "drive": (40.0, 1.5, 0.0),
    }

    def build_matrices(
        self,
        pois: list[POIInput],
        constraints,
    ) -> tuple[list[list[int]], list[list[float]]]:
        """Return (time_minutes_matrix, transport_cost_matrix)."""
        n = len(pois)
        time_matrix: list[list[int]] = [[0] * n for _ in range(n)]
        cost_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                mode = self._select_mode(pois[i], pois[j], constraints)
                minutes, cost = self._estimate(pois[i], pois[j], mode)
                time_matrix[i][j] = minutes
                cost_matrix[i][j] = cost
        return time_matrix, cost_matrix

    def _select_mode(
        self,
        origin: POIInput,
        dest: POIInput,
        constraints,
    ) -> Literal["walk", "subway", "bus", "taxi", "drive"]:
        """Pick transport mode by distance and constraints."""
        km = self._haversine_km(origin.lat, origin.lng, dest.lat, dest.lng)
        if km <= 1.0:
            return "walk"
        if km <= 5.0:
            return "subway" if dest.category not in ("nature", "suburb") else "taxi"
        if km <= 20.0:
            return "taxi"
        return "drive"

    def _estimate(
        self,
        origin: POIInput,
        dest: POIInput,
        mode: Literal["walk", "subway", "bus", "taxi", "drive"],
    ) -> tuple[int, float]:
        km = self._haversine_km(origin.lat, origin.lng, dest.lat, dest.lng)
        speed, per_km, base = self.MODE_PROFILES.get(mode, (40.0, 2.5, 0.0))
        minutes = max(1, int(km / speed * 60))
        cost = round(base + km * per_km, 2)
        return minutes, cost

    @staticmethod
    def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        lat1, lng1 = math.radians(lat1), math.radians(lng1)
        lat2, lng2 = math.radians(lat2), math.radians(lng2)
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
        return 6371 * 2 * math.asin(math.sqrt(s))
