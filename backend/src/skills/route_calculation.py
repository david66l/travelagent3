"""Route Calculation Skill - distance and time estimation."""

import math
from typing import Literal

from schemas import Location, RouteInfo


class RouteCalculationSkill:
    """Calculate routes and travel times between points."""

    WALK_SPEED_M_PER_MIN = 80  # ~5 km/h
    TRANSIT_SPEED_M_PER_MIN = 300  # ~18 km/h average
    TAXI_SPEED_M_PER_MIN = 400  # ~24 km/h average in city

    def calculate_route(
        self,
        origin: Location,
        destination: Location,
        mode: Literal["walk", "transit", "taxi"] = "transit",
    ) -> RouteInfo:
        """Calculate route between two points."""
        distance = self._haversine(origin.lat, origin.lng, destination.lat, destination.lng)

        speed = {
            "walk": self.WALK_SPEED_M_PER_MIN,
            "transit": self.TRANSIT_SPEED_M_PER_MIN,
            "taxi": self.TAXI_SPEED_M_PER_MIN,
        }[mode]

        duration = int(distance / speed)

        return RouteInfo(
            origin=origin,
            destination=destination,
            distance_m=int(distance),
            duration_min=max(duration, 5),  # Minimum 5 minutes
            mode=mode,
        )

    def get_distance_matrix(
        self,
        points: list[Location],
    ) -> list[list[float]]:
        """Calculate distance matrix for all point pairs."""
        n = len(points)
        matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    matrix[i][j] = self._haversine(
                        points[i].lat,
                        points[i].lng,
                        points[j].lat,
                        points[j].lng,
                    )
        return matrix

    @staticmethod
    def _haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate great circle distance in meters."""
        R = 6371000  # Earth's radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lng2 - lng1)

        a = (
            math.sin(delta_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c
