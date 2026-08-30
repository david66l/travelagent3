"""Tests for standalone VRP solver."""

import vrp_solver_service.solver as solver_module
from vrp_solver_service.models import ConstraintsInput, POIInput, SolverRequest
from vrp_solver_service.solver import TravelVRPSolver


def _beijing_pois() -> list[POIInput]:
    return [
        POIInput(
            id="p1",
            name="故宫",
            lat=39.9163,
            lng=116.3972,
            duration_minutes=180,
            ticket_price=60.0,
            score=0.9,
        ),
        POIInput(
            id="p2",
            name="天坛",
            lat=39.8830,
            lng=116.4120,
            duration_minutes=120,
            ticket_price=34.0,
            score=0.8,
        ),
        POIInput(
            id="p3",
            name="颐和园",
            lat=39.9990,
            lng=116.2750,
            duration_minutes=180,
            ticket_price=30.0,
            score=0.85,
        ),
    ]


def test_solver_returns_valid_itinerary_for_three_pois():
    pois = _beijing_pois()
    constraints = ConstraintsInput(
        travel_days=2, total_budget=5000, max_walk_minutes=300, must_visit=["故宫"]
    )
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
    constraints = ConstraintsInput(
        travel_days=1, max_walk_minutes=600, must_visit=["天坛", "颐和园"]
    )
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


def test_solver_reports_cpsat_internal_greedy_fallback(monkeypatch):
    monkeypatch.setattr(
        solver_module,
        "_cpsat_solve",
        lambda *_args, **_kwargs: ([], "UNKNOWN"),
    )

    response = TravelVRPSolver().solve(
        SolverRequest(
            pois=_beijing_pois(),
            constraints=ConstraintsInput(travel_days=2),
            strategy="cpsat",
        )
    )

    assert response.status == "fallback"
    assert response.message == "Greedy fallback used"
    assert response.metadata["cpsat_status"] == "UNKNOWN"


def test_solver_handles_single_poi():
    pois = [
        POIInput(
            id="p1", name="故宫", lat=39.9163, lng=116.3972, duration_minutes=180, ticket_price=60.0
        )
    ]
    constraints = ConstraintsInput(travel_days=1)
    request = SolverRequest(pois=pois, constraints=constraints)

    solver = TravelVRPSolver()
    response = solver.solve(request)

    assert len(response.days[0].activities) == 1
    assert response.days[0].activities[0].poi_name == "故宫"


def test_solver_applies_live_date_hours_and_temporary_closure():
    pois = [
        POIInput(
            id="closed",
            name="临时闭馆景点",
            score=1,
            closed_dates=["2026-09-01"],
        ),
        POIInput(
            id="late",
            name="下午开放景点",
            score=0.9,
            duration_minutes=60,
            date_opening_hours={"2026-09-01": ("15:00", "18:00")},
        ),
    ]
    response = TravelVRPSolver().solve(
        SolverRequest(
            pois=pois,
            constraints=ConstraintsInput(
                travel_days=1,
                trip_start_date="2026-09-01",
                include_restaurant=False,
                meals_per_day=0,
            ),
            strategy="cpsat",
        )
    )

    activities = [activity for day in response.days for activity in day.activities]
    assert all(activity.poi_id != "closed" for activity in activities)
    late = next(activity for activity in activities if activity.poi_id == "late")
    assert late.start_time >= "15:00"


def test_solver_applies_transport_derived_daily_boundaries():
    response = TravelVRPSolver().solve(
        SolverRequest(
            pois=[
                POIInput(
                    id="museum",
                    name="博物馆",
                    duration_minutes=60,
                    open_time="08:00",
                    close_time="20:00",
                )
            ],
            constraints=ConstraintsInput(
                travel_days=1,
                daily_start_minutes=[13 * 60],
                daily_end_minutes=[17 * 60],
                include_restaurant=False,
                meals_per_day=0,
            ),
            strategy="cpsat",
        )
    )

    activity = response.days[0].activities[0]
    assert activity.start_time >= "13:00"
    assert activity.end_time <= "17:00"
