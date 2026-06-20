"""Tests for standalone VRP solver."""

import pytest

from vrp_solver_service.models import ConstraintsInput, POIInput, SolverRequest
from vrp_solver_service.solver import TravelVRPSolver


def _beijing_pois() -> list[POIInput]:
    return [
        POIInput(id="p1", name="故宫", lat=39.9163, lng=116.3972, duration_minutes=180, ticket_price=60.0, score=0.9),
        POIInput(id="p2", name="天坛", lat=39.8830, lng=116.4120, duration_minutes=120, ticket_price=34.0, score=0.8),
        POIInput(id="p3", name="颐和园", lat=39.9990, lng=116.2750, duration_minutes=180, ticket_price=30.0, score=0.85),
    ]


def test_solver_returns_valid_itinerary_for_three_pois():
    pois = _beijing_pois()
    constraints = ConstraintsInput(travel_days=2, total_budget=5000, max_walk_minutes=300, must_visit=["故宫"])
    request = SolverRequest(pois=pois, constraints=constraints)

    solver = TravelVRPSolver()
    response = solver.solve(request)

    assert response.status in ("optimal", "fallback")
    assert len(response.days) == 2
    all_activities = [a for d in response.days for a in d.activities]
    assert any(a.poi_name == "故宫" for a in all_activities)
    # Each activity has valid time range
    for a in all_activities:
        assert a.start_time < a.end_time
        assert a.duration_min > 0


def test_solver_respects_must_visit():
    pois = _beijing_pois()
    constraints = ConstraintsInput(travel_days=1, max_walk_minutes=600, must_visit=["天坛", "颐和园"])
    request = SolverRequest(pois=pois, constraints=constraints)

    solver = TravelVRPSolver()
    response = solver.solve(request)

    all_names = {a.poi_name for d in response.days for a in d.activities}
    assert "天坛" in all_names
    assert "颐和园" in all_names


def test_solver_greedy_strategy_for_small_instance():
    pois = _beijing_pois()
    constraints = ConstraintsInput(travel_days=2, total_budget=5000)
    request = SolverRequest(pois=pois, constraints=constraints, strategy="greedy")

    solver = TravelVRPSolver()
    response = solver.solve(request)

    assert len(response.days) == 2
    all_activities = [a for d in response.days for a in d.activities]
    # Greedy may inject dummy meal nodes when food_day > 0
    assert len(all_activities) <= len(pois) + 2 * constraints.travel_days


def test_solver_handles_single_poi():
    pois = [POIInput(id="p1", name="故宫", lat=39.9163, lng=116.3972, duration_minutes=180, ticket_price=60.0)]
    constraints = ConstraintsInput(travel_days=1)
    request = SolverRequest(pois=pois, constraints=constraints)

    solver = TravelVRPSolver()
    response = solver.solve(request)

    assert len(response.days[0].activities) == 1
    assert response.days[0].activities[0].poi_name == "故宫"
