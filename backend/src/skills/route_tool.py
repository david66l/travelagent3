"""Route / distance calculation tool with API stub + heuristic fallback."""

import logging


from core.settings import settings
from schemas import Location, RouteInfo, ToolResult
from tools.base import Tool

logger = logging.getLogger(__name__)


class RouteTool(Tool):
    """Calculate route distance and duration between two points."""

    name = "route"
    timeout = 3.0
    retries = 1
    cache_ttl = settings.cache_ttl_route

    async def route(
        self,
        origin: Location,
        destination: Location,
        mode: str = "transit",
    ) -> RouteInfo:
        """Backward-compatible entry point returning a RouteInfo."""
        result = await self.run(
            {
                "origin": origin.model_dump(),
                "destination": destination.model_dump(),
                "mode": mode,
            }
        )
        data = result.data
        if isinstance(data, RouteInfo):
            return data
        if isinstance(data, dict):
            return RouteInfo(**data)
        return RouteInfo(
            origin=origin,
            destination=destination,
            distance_m=0,
            duration_min=0,
            mode=mode,
            data_source="unavailable",
            is_fallback=True,
            fallback_reason="route lookup returned no data",
        )

    async def execute(self, params: dict) -> ToolResult:
        """Try Amap Distance Matrix API; fallback to heuristic formula."""
        origin = Location(**params["origin"])
        destination = Location(**params["destination"])
        mode = params.get("mode", "transit")

        if settings.amap_key:
            try:
                route = await self._fetch_amap_route(origin, destination, mode)
                return ToolResult(
                    data=route,
                    data_source="api",
                    confidence=0.9,
                )
            except Exception as exc:
                logger.warning("Amap route API failed: %s", exc)

        route = self._heuristic_route(origin, destination, mode)
        return ToolResult(
            data=route,
            data_source="fallback",
            confidence=0.7,
            is_fallback=True,
            fallback_reason="route api unavailable, using heuristic estimate",
        )

    async def _fetch_amap_route(
        self, origin: Location, destination: Location, mode: str
    ) -> RouteInfo:
        """Amap Distance Matrix API stub (replace with real endpoint)."""
        raise NotImplementedError("real route API not configured")

    def _heuristic_route(self, origin: Location, destination: Location, mode: str) -> RouteInfo:
        """Estimate distance/duration using planar approximation."""
        distance_km = _distance_km(origin, destination)
        distance_m = int(distance_km * 1000)
        duration_min = _estimate_transit_minutes(distance_km)
        return RouteInfo(
            origin=origin,
            destination=destination,
            distance_m=distance_m,
            duration_min=duration_min,
            mode=mode,
            data_source="fallback",
            confidence=0.7,
            is_fallback=True,
            fallback_reason="heuristic distance estimate",
        )


def _distance_km(a: Location, b: Location) -> float:
    """Approximate distance in kilometers."""
    lat_km = (a.lat - b.lat) * 111
    lng_km = (a.lng - b.lng) * 85
    return (lat_km**2 + lng_km**2) ** 0.5


def _estimate_transit_minutes(distance_km: float) -> int:
    """Estimate transit time based on distance."""
    if distance_km < 1:
        return 10
    if distance_km < 8:
        return int(distance_km / 18 * 60) + 10
    if distance_km < 35:
        return int(distance_km / 24 * 60) + 15
    return int(distance_km / 35 * 60) + 25
