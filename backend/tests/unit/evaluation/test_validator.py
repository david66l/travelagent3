from __future__ import annotations

from evaluation.validator import ItineraryValidator


def _valid_itinerary() -> list[dict]:
    return [
        {
            "day_number": 1,
            "date": "2026-08-11",
            "activities": [
                {
                    "poi_id": "museum-1",
                    "poi_name": "城市博物馆",
                    "category": "attraction",
                    "start_time": "09:00",
                    "end_time": "11:00",
                    "ticket_price": 50,
                    "tags": ["历史"],
                },
                {
                    "poi_id": "park-1",
                    "poi_name": "城市公园",
                    "category": "attraction",
                    "start_time": "13:00",
                    "end_time": "15:00",
                    "ticket_price": 0,
                    "transit_from_prev": {"duration_min": 30},
                    "tags": ["自然"],
                },
            ],
        }
    ]


def test_valid_itinerary_passes_and_emits_versioned_metrics() -> None:
    report = ItineraryValidator().validate(
        _valid_itinerary(),
        constraints={
            "travel_days": 1,
            "total_budget": 500,
            "max_transit_minutes": 120,
            "must_visit": ["museum-1"],
            "interests": ["历史"],
        },
        facts=[
            {
                "id": "museum-1",
                "name": "城市博物馆",
                "open_time": "09:00",
                "close_time": "17:00",
                "closed_weekdays": [],
            }
        ],
    )

    assert report.hard_pass is True
    assert report.validator_version == "travel-validator.v1"
    assert report.metrics["total_cost"] == 50
    assert report.metrics["total_transit_minutes"] == 30
    assert report.soft_scores["preference_match"] == 0.5


def test_validator_returns_stable_hard_violation_codes() -> None:
    itinerary = _valid_itinerary()
    itinerary[0]["activities"][1]["start_time"] = "10:30"
    itinerary[0]["activities"][1]["end_time"] = "12:30"
    itinerary[0]["activities"].append(dict(itinerary[0]["activities"][0]))

    report = ItineraryValidator().validate(
        itinerary,
        constraints={
            "travel_days": 2,
            "total_budget": 10,
            "max_transit_minutes": 10,
            "must_visit": ["missing-poi"],
        },
        facts={
            "museum-1": {
                "name": "城市博物馆",
                "open_time": "10:00",
                "close_time": "17:00",
                "closed_weekdays": [1],
            }
        },
    )
    codes = {violation.code for violation in report.hard_violations}

    assert report.hard_pass is False
    assert {
        "TRAVEL_DAY_COUNT_MISMATCH",
        "ACTIVITY_TIME_OVERLAP",
        "DUPLICATE_POI_VISIT",
        "MUST_VISIT_MISSING",
        "TOTAL_BUDGET_EXCEEDED",
        "MAX_TRANSIT_EXCEEDED",
        "POI_CLOSED_ON_DATE",
    } <= codes


def test_empty_itinerary_never_passes() -> None:
    report = ItineraryValidator().validate([], constraints={"travel_days": 1})

    assert report.hard_pass is False
    assert report.hard_violations[0].code == "EMPTY_ITINERARY"


def test_validator_rejects_forbidden_poi_by_name_alias() -> None:
    itinerary = [
        {
            "day_number": 1,
            "activities": [
                {
                    "poi_id": "bund-1",
                    "poi_name": "上海外滩观景区",
                    "category": "attraction",
                    "start_time": "09:00",
                    "end_time": "11:00",
                }
            ],
        }
    ]

    report = ItineraryValidator().validate(
        itinerary,
        {"travel_days": 1, "must_not_visit": ["外滩"]},
    )

    assert report.hard_pass is False
    assert any(v.code == "MUST_NOT_VISIT_PRESENT" for v in report.hard_violations)


def test_validator_rejects_consecutive_and_excessive_restaurants() -> None:
    itinerary = _valid_itinerary()
    itinerary[0]["activities"] = [
        {
            "poi_name": f"Restaurant {index}",
            "category": "restaurant",
            "start_time": f"{8 + index * 2:02d}:00",
            "end_time": f"{10 + index * 2:02d}:00",
        }
        for index in range(3)
    ]

    report = ItineraryValidator().validate(
        itinerary,
        constraints={"travel_days": 1, "include_restaurant": True, "meals_per_day": 2},
    )

    codes = {violation.code for violation in report.hard_violations}
    assert "CONSECUTIVE_DINING_ACTIVITIES" in codes
    assert "TOO_MANY_DINING_ACTIVITIES" in codes
    assert report.hard_pass is False


def test_validator_checks_live_date_hours_and_transport_day_boundaries() -> None:
    report = ItineraryValidator().validate(
        [
            {
                "day_number": 1,
                "activities": [
                    {
                        "poi_id": "museum",
                        "poi_name": "博物馆",
                        "category": "attraction",
                        "start_time": "12:00",
                        "end_time": "13:00",
                    }
                ],
            }
        ],
        constraints={
            "travel_days": 1,
            "trip_start_date": "2026-09-01",
            "daily_start_minutes": [13 * 60],
            "daily_end_minutes": [17 * 60],
        },
        facts=[
            {
                "id": "museum",
                "name": "博物馆",
                "open_time": "08:00",
                "close_time": "18:00",
                "date_opening_hours": {"2026-09-01": [14 * 60, 16 * 60]},
            }
        ],
    )

    codes = {item.code for item in report.hard_violations}
    assert "DAY_TIME_BOUNDARY_EXCEEDED" in codes
    assert "POI_CLOSED_DURING_VISIT" in codes
