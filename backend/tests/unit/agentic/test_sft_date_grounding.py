"""Trip-date derivation must not turn arbitrary policy arguments into evidence."""
import pytest

from agentic.sft_dataset import _arguments_grounded, _date_derived_from_trip


def context(start="2028-02-23", end="2028-02-25", days=3):
    return {"hard_constraints": {"start_date": start, "end_date": end, "travel_days": days}}


@pytest.mark.parametrize("action", ["get_weather", "search_current_info", "search_transport"])
def test_interior_trip_date_is_grounded_without_literal_occurrence(action):
    assert "2028-02-24" not in str(context())
    assert _arguments_grounded(action, {"date": "2028-02-24"}, context())


@pytest.mark.parametrize("start,end,days,target", [
    ("2028-02-28", "2028-03-01", 3, "2028-02-29"),
    ("2027-02-28", "2027-03-02", 3, "2027-03-01"),
    ("2028-12-31", "2029-01-02", 3, "2029-01-01"),
    ("2028-02-23", "2028-02-23", 1, "2028-02-23"),
    ("2028-02-23", "2028-02-25", 3, "2028-02-25"),
])
def test_calendar_derivation_handles_boundaries(start, end, days, target):
    assert _date_derived_from_trip(target, context(start, end, days))


@pytest.mark.parametrize("target", ["2028-02-22", "2028-02-26", "2028-02-30", "2027-02-29",
    "20280224", "2028-W08-4", "2028-2-24", " 2028-02-24", "2028-02-24T12:00:00",
    "２０２８-０２-２４", None, 20280224, True, ["2028-02-24"], {"value": "2028-02-24"}])
def test_unmentioned_outside_or_malformed_dates_are_rejected(target):
    assert not _date_derived_from_trip(target, context())
    assert not _arguments_grounded("get_weather", {"date": target}, context())


@pytest.mark.parametrize("constraints", [None, {}, "2028-02-23 to 2028-02-25",
    {"start_date": "2028-02-23", "travel_days": 3},
    {"start_date": "2028-02-23", "end_date": "2028-02-25"},
    {"start_date": "2028-02-25", "end_date": "2028-02-23", "travel_days": 3},
    {"start_date": "2028-02-23", "end_date": "2028-02-25", "travel_days": 2},
    {"start_date": "2028-02-23", "end_date": "2028-02-25", "travel_days": "3"},
    {"start_date": "2028-02-23", "end_date": "2028-02-25", "travel_days": True},
    {"start_date": "2028-02-30", "end_date": "2028-03-02", "travel_days": 3},
])
def test_missing_inconsistent_or_untyped_trip_bounds_do_not_authorize_derivation(constraints):
    assert not _arguments_grounded("get_weather", {"date": "2028-02-24"}, {"hard_constraints": constraints})


def test_derivation_cannot_authorize_other_fields_actions_or_unstructured_text():
    assert not _arguments_grounded("search_current_info", {"query": "new query", "date": "2028-02-24", "poi_id": "invented-place"}, context())
    assert not _arguments_grounded("search_pois", {"date": "2028-02-24"}, context())
    assert not _arguments_grounded("get_weather", {"other_date": "2028-02-24"}, context())
    assert not _arguments_grounded("get_weather", {"date": "2028-02-24"}, {"original_request": "2028-02-23 to 2028-02-25, three days"})


def test_existing_literal_evidence_route_is_preserved():
    assert _arguments_grounded("get_weather", {"date": "2028-02-24"}, {"relevant_facts": [{"date": "2028-02-24"}]})
