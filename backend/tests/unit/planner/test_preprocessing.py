"""Tests for VRP preprocessing modules."""

import pytest

from planner.preprocessing import (
    CPSATTuningGuide,
    FatigueModel,
    PlayTimeManager,
    ReservationHandler,
    RestaurantHandler,
    TransportSelector,
)
from vrp_solver_service.models import ConstraintsInput, POIInput, ReservationInput


def test_reservation_handler_marks_must_visit_and_reminds():
    pois = [
        POIInput(id="p1", name="故宫", reservation="需提前 7 天预约"),
        POIInput(id="p2", name="天坛"),
    ]
    reservations = [ReservationInput(poi_id="p1")]
    handler = ReservationHandler()
    result, reminders = handler.filter_and_remind(pois, reservations)

    assert result[0].must_visit is True
    assert any(r.poi_id == "p1" and "提前" in r.message for r in reminders)


def test_reservation_handler_warns_missing_poi():
    pois = [POIInput(id="p1", name="故宫")]
    reservations = [ReservationInput(poi_id="p2")]
    handler = ReservationHandler()
    _, reminders = handler.filter_and_remind(pois, reservations)
    assert any(r.poi_id == "p2" for r in reminders)


def test_play_time_manager_adjusts_evening_window():
    pois = [POIInput(id="p1", name="景山", best_visit_period="sunset", open_time="08:00", close_time="22:00")]
    adjusted = PlayTimeManager().adjust(pois, ConstraintsInput())
    assert adjusted[0].open_time == "16:00"


def test_play_time_manager_quick_mode_clamps_duration():
    pois = [POIInput(id="p1", name="故宫", duration_minutes=180, min_play_time=60, max_play_time=240)]
    adjusted = PlayTimeManager().adjust(pois, ConstraintsInput(play_mode="quick"))
    assert adjusted[0].duration_minutes == 60


def test_play_time_manager_deep_mode_extends_duration():
    pois = [POIInput(id="p1", name="故宫", duration_minutes=180, min_play_time=60, max_play_time=240)]
    adjusted = PlayTimeManager().adjust(pois, ConstraintsInput(play_mode="deep"))
    assert adjusted[0].duration_minutes == 240


def test_restaurant_handler_injects_meals_when_opted_in():
    pois = [POIInput(id="p1", name="故宫")]
    constraints = ConstraintsInput(travel_days=1, food_day=200, include_restaurant=True, meals_per_day=2)
    result = RestaurantHandler().inject(pois, constraints)
    assert len(result) == 3  # original + lunch + dinner
    assert sum(1 for p in result if p.category == "restaurant") == 2


def test_restaurant_handler_skips_when_not_opted_in():
    pois = [POIInput(id="p1", name="故宫")]
    constraints = ConstraintsInput(travel_days=1, food_day=200, include_restaurant=False)
    result = RestaurantHandler().inject(pois, constraints)
    assert len(result) == 1


def test_restaurant_handler_skips_when_no_meals():
    pois = [POIInput(id="p1", name="故宫")]
    constraints = ConstraintsInput(travel_days=1, food_day=200, include_restaurant=True, meals_per_day=0)
    result = RestaurantHandler().inject(pois, constraints)
    assert len(result) == 1


def test_transport_selector_returns_square_matrices():
    pois = [
        POIInput(id="p1", name="故宫", lat=39.9163, lng=116.3972),
        POIInput(id="p2", name="天坛", lat=39.8830, lng=116.4120),
    ]
    dist, tc = TransportSelector().build_matrices(pois, ConstraintsInput())
    assert len(dist) == len(dist[0]) == 2
    assert len(tc) == len(tc[0]) == 2
    assert dist[0][1] > 0


def test_fatigue_model_reduces_budget_over_days():
    limits = FatigueModel().daily_walk_limits(ConstraintsInput(travel_days=3, max_walk_km=10))
    assert limits[0] >= limits[-1]
    assert all(l > 0 for l in limits)


def test_fatigue_model_forces_recovery_day_after_high_streak():
    # Use family_elder (high alpha = slow recovery) to trigger recovery day
    limits = FatigueModel().daily_walk_limits(
        ConstraintsInput(travel_days=5, max_walk_km=10, travelers_type="family_elder")
    )
    # After 2 high-intensity days, the third should be <= 0.4 * base = 4
    assert any(limit <= 4 for limit in limits[2:])


def test_cp_sat_tuning_scales_time_and_uses_single_worker():
    small = CPSATTuningGuide().recommend(ConstraintsInput(travel_days=1), 5)
    large = CPSATTuningGuide().recommend(ConstraintsInput(travel_days=5), 50)
    # Single worker on purpose: multi-worker CP-SAT dead-hangs on macOS+ortools
    # 9.15 before search begins, so the time limit never fires. See tuning guide.
    assert small["num_search_workers"] == large["num_search_workers"] == 1
    assert small["max_time_in_seconds"] < large["max_time_in_seconds"]
