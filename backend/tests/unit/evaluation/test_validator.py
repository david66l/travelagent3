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
