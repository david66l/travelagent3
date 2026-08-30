"""Meal obligations must remain separated by a real activity."""

from evaluation.validator import ItineraryValidator
from planner.preprocessing.transport_selector import TransportSelector
from vrp_solver_service.models import ConstraintsInput, POIInput, SolverRequest
from vrp_solver_service.solver import TravelVRPSolver


def test_greedy_solver_does_not_schedule_lunch_and_dinner_consecutively():
    pois = [
        POIInput(
            id=f"poi-{index}",
            name=f"Attraction {index}",
            category="attraction",
            duration_minutes=120,
            open_time="08:00",
            close_time="20:00",
            score=1 - index * 0.05,
            lat=31.20 + index * 0.01,
            lng=121.40 + index * 0.01,
        )
        for index in range(6)
    ]
    constraints = ConstraintsInput(
        travel_days=2,
        include_restaurant=True,
        meals_per_day=2,
        total_budget=3000,
    )
    matrix_pois = [
        POIInput(
            id="__hotel",
            name="Hotel",
            category="hotel",
            duration_minutes=0,
            open_time="00:00",
            close_time="23:59",
        ),
        *pois,
    ]
    dist, cost = TransportSelector().build_matrices(matrix_pois, constraints)
    result = TravelVRPSolver().solve(
        SolverRequest(
            pois=pois,
            constraints=constraints,
            dist_matrix=dist,
            tc_matrix=cost,
            strategy="greedy",
        )
    )

    itinerary = [day.model_dump(mode="json") for day in result.days]
    report = ItineraryValidator().validate(
        itinerary,
        constraints={"travel_days": 2, "meals_per_day": 2},
    )
    assert "CONSECUTIVE_DINING_ACTIVITIES" not in {
        violation.code for violation in report.hard_violations
    }
