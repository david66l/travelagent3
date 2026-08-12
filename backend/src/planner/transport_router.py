"""Map service router — primary map API with haversine fallback."""

from __future__ import annotations

import logging
import math
from typing import Literal

import httpx

from core.settings import settings

logger = logging.getLogger(__name__)


class HaversineFallback:
    """Estimate travel time/cost when map API fails."""

    _MODE_PROFILES: dict[str, tuple[float, float]] = {
        "walk": (4.5, 0.0),
        "subway": (25.0, 0.5),
        "bus": (20.0, 0.5),
        "taxi": (40.0, 2.5),
        "drive": (40.0, 1.5),
    }

    def estimate(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        mode: Literal["walk", "subway", "bus", "taxi", "drive"] = "taxi",
    ) -> tuple[int, float]:
        """Return (minutes, cost_cny)."""
        km = self._haversine_km(origin_lat, origin_lng, dest_lat, dest_lng)
        speed, cost_per_km = self._MODE_PROFILES.get(mode, (40.0, 2.5))
        minutes = max(1, int(km / speed * 60))
        cost = round(km * cost_per_km, 2)
        return minutes, cost

    @staticmethod
    def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        lat1, lng1 = math.radians(lat1), math.radians(lng1)
        lat2, lng2 = math.radians(lat2), math.radians(lng2)
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
        return 6371 * 2 * math.asin(math.sqrt(s))


class MapServiceRouter:
    """Route map distance queries to primary provider or haversine fallback."""

    def __init__(self, amap_key: str | None = None):
        self._amap_key = amap_key or settings.amap_key
        self._fallback = HaversineFallback()
        self._failure_count = 0
        self._max_primary_failures = 3

    async def get_distance(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        mode: Literal["walk", "subway", "bus", "taxi", "drive"] = "taxi",
    ) -> tuple[int, float]:
        """Return (minutes, cost_cny) for a leg."""
        if self._failure_count < self._max_primary_failures:
            try:
                result = await self._amap_distance(origin_lat, origin_lng, dest_lat, dest_lng, mode)
                return result
            except Exception as exc:
                logger.warning("Map provider failed: %s", exc)
                self._failure_count += 1

        return self._fallback.estimate(origin_lat, origin_lng, dest_lat, dest_lng, mode)

    async def _amap_distance(
        self,
        origin_lat: float,
        origin_lng: float,
        dest_lat: float,
        dest_lng: float,
        mode: Literal["walk", "subway", "bus", "taxi", "drive"],
    ) -> tuple[int, float]:
        """Call AMap distance API (MVP: falls through to exception if key missing)."""
        if not self._amap_key:
            raise RuntimeError("AMap key not configured")

        mode_map = {"walk": "1", "subway": "2", "bus": "3", "taxi": "4", "drive": "0"}
        url = "https://restapi.amap.com/v3/distance"
        params = {
            "key": self._amap_key,
            "origins": f"{origin_lng},{origin_lat}",
            "destination": f"{dest_lng},{dest_lat}",
            "type": mode_map.get(mode, "0"),
        }
        from core.amap_rate import amap_rate_gate

        async with httpx.AsyncClient(timeout=5.0) as client:
            await amap_rate_gate()
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "1":
                raise RuntimeError(data.get("info", "AMap error"))
            result = data["results"][0]
            minutes = max(1, int(int(result["distance"]) / 1000 / 40 * 60))
            return minutes, 0.0
