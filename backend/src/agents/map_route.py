"""Map & Route Agent - geographic calculations and route optimization."""

from typing import Optional

from schemas import Location, DayPlan, Activity
from skills.route_calculation import RouteCalculationSkill


class MapRouteAgent:
    """Geographic calculations and route optimization."""

    def __init__(self):
        self.route_skill = RouteCalculationSkill()

    async def get_coordinates(self, poi_name: str, city: str) -> Optional[Location]:
        """Get coordinates for a POI using deterministic city-center offset."""
        city_centers = {
            "北京": (39.9042, 116.4074),
            "上海": (31.2304, 121.4737),
            "广州": (23.1291, 113.2644),
            "深圳": (22.5431, 114.0579),
            "成都": (30.5728, 104.0668),
            "杭州": (30.2741, 120.1551),
            "西安": (34.3416, 108.9398),
            "重庆": (29.5630, 106.5516),
            "苏州": (31.2989, 120.5853),
            "南京": (32.0603, 118.7969),
        }
        base = city_centers.get(city, (30.0, 120.0))
        # Deterministic offset based on hash of poi_name
        h = hash(poi_name) % 1000
        lat_offset = (h % 100 - 50) / 10000.0  # ±0.005° ≈ ±550m
        lng_offset = ((h // 100) % 100 - 50) / 10000.0
        return Location(
            lat=base[0] + lat_offset,
            lng=base[1] + lng_offset,
        )

    async def batch_optimize_routes(
        self,
        itinerary: list[DayPlan],
        hotel_location: Optional[Location] = None,
    ) -> dict[int, list[Activity]]:
        """Optimize routes for each day using 2-opt."""
        optimized = {}
        for day in itinerary:
            if len(day.activities) <= 2:
                optimized[day.day_number] = day.activities
                continue

            # Ensure all activities have coordinates
            activities_with_coords = []
            for act in day.activities:
                if not act.location:
                    # Try to get coordinates from city + poi_name
                    act.location = await self.get_coordinates(act.poi_name, "")
                activities_with_coords.append(act)

            # Build distance matrix
            points = [a.location for a in activities_with_coords if a.location]
            if len(points) < 2:
                optimized[day.day_number] = day.activities
                continue

            dist_matrix = self.route_skill.get_distance_matrix(points)

            # 2-opt optimization
            route_indices = list(range(len(points)))
            route_indices = self._two_opt(route_indices, dist_matrix)

            # Reorder activities
            optimized[day.day_number] = [activities_with_coords[i] for i in route_indices]

        return optimized

    def _two_opt(self, route: list[int], dist_matrix: list[list[float]]) -> list[int]:
        """2-opt route optimization algorithm."""
        n = len(route)
        if n < 3:
            return route

        improved = True
        max_iterations = 100
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            for i in range(n - 1):
                for j in range(i + 2, n):
                    # Calculate delta for 2-opt swap
                    a, b = route[i], route[i + 1]
                    c, d = route[j], route[(j + 1) % n]

                    delta = (
                        dist_matrix[a][c]
                        + dist_matrix[b][d]
                        - dist_matrix[a][b]
                        - dist_matrix[c][d]
                    )

                    if delta < -1:  # Improvement threshold
                        # Reverse segment between i+1 and j
                        route[i + 1 : j + 1] = reversed(route[i + 1 : j + 1])
                        improved = True

        return route

    def calculate_daily_transit(
        self,
        activities: list[Activity],
        hotel: Optional[Location] = None,
    ) -> list[dict]:
        """Calculate transit info between consecutive activities."""
        transit_info = []
        for i in range(len(activities) - 1):
            curr = activities[i]
            next_act = activities[i + 1]

            if curr.location and next_act.location:
                route = self.route_skill.calculate_route(
                    curr.location, next_act.location, "transit"
                )
                transit_info.append(
                    {
                        "from": curr.poi_name,
                        "to": next_act.poi_name,
                        "distance_m": route.distance_m,
                        "duration_min": route.duration_min,
                        "mode": route.mode,
                    }
                )

        return transit_info
