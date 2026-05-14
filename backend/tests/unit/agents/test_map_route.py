"""Tests for MapRouteAgent."""

import pytest
from schemas import Location, DayPlan, Activity
from agents.map_route import MapRouteAgent


class TestMapRouteAgent:
    """Test coordinate resolution and 2-opt route optimization."""

    def setup_method(self):
        self.agent = MapRouteAgent()

    # === Coordinate Resolution Tests ===

    @pytest.mark.asyncio
    async def test_get_coordinates_beijing(self):
        loc = await self.agent.get_coordinates("故宫", "北京")
        assert loc is not None
        assert 39.8 < loc.lat < 40.0
        assert 116.3 < loc.lng < 116.5

    @pytest.mark.asyncio
    async def test_get_coordinates_shanghai(self):
        loc = await self.agent.get_coordinates("外滩", "上海")
        assert loc is not None
        assert 31.1 < loc.lat < 31.4
        assert 121.3 < loc.lng < 121.6

    @pytest.mark.asyncio
    async def test_get_coordinates_unknown_city(self):
        loc = await self.agent.get_coordinates("某景点", "未知城市")
        assert loc is not None  # falls back to default (30.0, 120.0)

    @pytest.mark.asyncio
    async def test_get_coordinates_deterministic(self):
        """Same POI name should always produce same coordinates."""
        loc1 = await self.agent.get_coordinates("测试景点", "北京")
        loc2 = await self.agent.get_coordinates("测试景点", "北京")
        assert loc1.lat == loc2.lat
        assert loc1.lng == loc2.lng

    @pytest.mark.asyncio
    async def test_get_coordinates_different_names_different_coords(self):
        loc1 = await self.agent.get_coordinates("景点A", "北京")
        loc2 = await self.agent.get_coordinates("景点B", "北京")
        # Hash-based offsets should differ for different names
        assert loc1.lat != loc2.lat or loc1.lng != loc2.lng

    # === 2-opt Optimization Tests ===

    def test_two_opt_basic(self):
        dist = [
            [0, 10, 15, 20],
            [10, 0, 35, 25],
            [15, 35, 0, 30],
            [20, 25, 30, 0],
        ]
        route = [0, 1, 2, 3]
        result = self.agent._two_opt(route, dist)
        assert len(result) == 4
        assert set(result) == {0, 1, 2, 3}

    def test_two_opt_three_points(self):
        dist = [
            [0, 5, 10],
            [5, 0, 8],
            [10, 8, 0],
        ]
        route = [0, 1, 2]
        result = self.agent._two_opt(route, dist)
        assert len(result) == 3

    def test_two_opt_no_improvement_already_optimal(self):
        dist = [
            [0, 1, 100],
            [1, 0, 100],
            [100, 100, 0],
        ]
        route = [0, 1, 2]
        result = self.agent._two_opt(route, dist)
        assert result == [0, 1, 2]

    def test_two_opt_two_points(self):
        dist = [[0, 5], [5, 0]]
        route = [0, 1]
        result = self.agent._two_opt(route, dist)
        assert result == [0, 1]

    def test_two_opt_large_route(self):
        n = 10
        dist = [[0 if i == j else abs(i - j) * 10 for j in range(n)] for i in range(n)]
        route = list(range(n))
        result = self.agent._two_opt(route, dist)
        assert len(result) == n
        assert set(result) == set(range(n))

    # === Batch Optimize Tests ===

    @pytest.mark.asyncio
    async def test_batch_optimize_single_activity(self):
        day = DayPlan(
            day_number=1,
            activities=[
                Activity(poi_name="A", category="attraction", location=Location(lat=39.9, lng=116.4)),
            ],
        )
        result = await self.agent.batch_optimize_routes([day])
        assert len(result[1]) == 1

    @pytest.mark.asyncio
    async def test_batch_optimize_multiple_days(self):
        day1 = DayPlan(
            day_number=1,
            activities=[
                Activity(poi_name="A", category="attraction", location=Location(lat=39.9, lng=116.4)),
                Activity(poi_name="B", category="attraction", location=Location(lat=39.95, lng=116.35)),
            ],
        )
        day2 = DayPlan(
            day_number=2,
            activities=[
                Activity(poi_name="C", category="attraction", location=Location(lat=39.85, lng=116.45)),
            ],
        )
        result = await self.agent.batch_optimize_routes([day1, day2])
        assert 1 in result
        assert 2 in result

    @pytest.mark.asyncio
    async def test_batch_optimize_missing_coordinates(self):
        day = DayPlan(
            day_number=1,
            activities=[
                Activity(poi_name="A", category="attraction", location=None),
                Activity(poi_name="B", category="attraction", location=None),
            ],
        )
        result = await self.agent.batch_optimize_routes([day])
        assert len(result[1]) == 2

    # === Daily Transit Tests ===

    def test_calculate_daily_transit(self):
        activities = [
            Activity(poi_name="A", category="attraction", location=Location(lat=39.9, lng=116.4)),
            Activity(poi_name="B", category="attraction", location=Location(lat=39.95, lng=116.35)),
        ]
        transit = self.agent.calculate_daily_transit(activities)
        assert len(transit) == 1
        assert transit[0]["from"] == "A"
        assert transit[0]["to"] == "B"
        assert transit[0]["distance_m"] > 0

    def test_calculate_daily_transit_no_location(self):
        activities = [
            Activity(poi_name="A", category="attraction", location=None),
            Activity(poi_name="B", category="attraction", location=None),
        ]
        transit = self.agent.calculate_daily_transit(activities)
        assert len(transit) == 0
