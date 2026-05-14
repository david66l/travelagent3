"""Tests for RouteCalculationSkill."""

import math
import pytest
from schemas import Location, RouteInfo
from skills.route_calculation import RouteCalculationSkill


class TestRouteCalculationSkill:
    """Test haversine distance and route calculations."""

    def setup_method(self):
        self.skill = RouteCalculationSkill()

    # === Haversine Tests ===

    def test_haversine_same_point(self):
        dist = self.skill._haversine(39.9, 116.4, 39.9, 116.4)
        assert dist == pytest.approx(0, abs=1)

    def test_haversine_known_distance_beijing_shanghai(self):
        """Beijing to Shanghai is approximately 1067 km."""
        dist = self.skill._haversine(39.9042, 116.4074, 31.2304, 121.4737)
        assert dist == pytest.approx(1_067_000, abs=5000)

    def test_haversine_symmetry(self):
        d1 = self.skill._haversine(39.9, 116.4, 31.2, 121.5)
        d2 = self.skill._haversine(31.2, 121.5, 39.9, 116.4)
        assert d1 == pytest.approx(d2, abs=0.1)

    def test_haversine_small_distance(self):
        """Two points 1km apart in Beijing."""
        dist = self.skill._haversine(39.9, 116.4, 39.909, 116.4)
        assert dist == pytest.approx(1000, abs=50)

    # === Route Calculation Tests ===

    def test_calculate_route_walk(self):
        origin = Location(lat=39.9, lng=116.4)
        destination = Location(lat=39.91, lng=116.41)
        route = self.skill.calculate_route(origin, destination, "walk")
        assert route.mode == "walk"
        assert route.duration_min >= 5
        assert route.distance_m > 0

    def test_calculate_route_transit(self):
        origin = Location(lat=39.9, lng=116.4)
        destination = Location(lat=39.91, lng=116.41)
        route = self.skill.calculate_route(origin, destination, "transit")
        assert route.mode == "transit"
        # Transit should be faster than walking
        walk_route = self.skill.calculate_route(origin, destination, "walk")
        assert route.duration_min < walk_route.duration_min

    def test_calculate_route_taxi(self):
        origin = Location(lat=39.9, lng=116.4)
        destination = Location(lat=39.91, lng=116.41)
        route = self.skill.calculate_route(origin, destination, "taxi")
        assert route.mode == "taxi"
        # Taxi should be fastest
        transit_route = self.skill.calculate_route(origin, destination, "transit")
        assert route.duration_min <= transit_route.duration_min

    def test_calculate_route_minimum_duration(self):
        """Even for very close points, duration should be at least 5 minutes."""
        origin = Location(lat=39.9, lng=116.4)
        destination = Location(lat=39.90001, lng=116.40001)
        route = self.skill.calculate_route(origin, destination, "walk")
        assert route.duration_min >= 5

    # === Distance Matrix Tests ===

    def test_distance_matrix_2x2(self):
        points = [
            Location(lat=39.9, lng=116.4),
            Location(lat=39.95, lng=116.35),
        ]
        matrix = self.skill.get_distance_matrix(points)
        assert len(matrix) == 2
        assert len(matrix[0]) == 2
        assert matrix[0][0] == 0
        assert matrix[1][1] == 0
        assert matrix[0][1] > 0
        assert matrix[1][0] > 0

    def test_distance_matrix_5x5(self):
        points = [Location(lat=39.9 + i * 0.01, lng=116.4) for i in range(5)]
        matrix = self.skill.get_distance_matrix(points)
        assert len(matrix) == 5
        for i in range(5):
            assert matrix[i][i] == 0
            for j in range(5):
                if i != j:
                    assert matrix[i][j] > 0

    def test_distance_matrix_diagonal_zero(self):
        points = [
            Location(lat=39.9, lng=116.4),
            Location(lat=39.95, lng=116.35),
            Location(lat=39.85, lng=116.45),
        ]
        matrix = self.skill.get_distance_matrix(points)
        for i in range(3):
            assert matrix[i][i] == 0

    def test_distance_matrix_symmetry(self):
        points = [
            Location(lat=39.9, lng=116.4),
            Location(lat=39.95, lng=116.35),
        ]
        matrix = self.skill.get_distance_matrix(points)
        assert matrix[0][1] == pytest.approx(matrix[1][0], abs=0.1)
