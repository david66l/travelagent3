"""Unit tests for Fact Guard — per-activity protected-field comparison."""

import copy

from schemas import Activity, Location
from planner.core.fact_guard import (
    activity_fields_match,
    protected_field_differences,
    protected_fields_match,
)


def make_activity() -> Activity:
    """Standard activity with all protected fields populated."""
    return Activity(
        poi_name="故宫",
        category="attraction",
        start_time="09:00",
        end_time="13:00",
        duration_min=240,
        ticket_price=60,
        location=Location(lat=39.916, lng=116.397),
    )


class TestProtectedFieldChanges:
    """Any change to a protected field must be detected."""

    def test_change_start_time_fails(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.start_time = "10:00"
        assert not activity_fields_match(a, b)

    def test_change_poi_name_fails(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.poi_name = "假故宫"
        assert not activity_fields_match(a, b)

    def test_change_ticket_price_fails(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.ticket_price = 999
        assert not activity_fields_match(a, b)

    def test_change_duration_fails(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.duration_min = 999
        assert not activity_fields_match(a, b)

    def test_change_location_fails(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.location = Location(lat=0, lng=0)
        assert not activity_fields_match(a, b)

    def test_change_end_time_fails(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.end_time = "15:00"
        assert not activity_fields_match(a, b)
        assert protected_field_differences(a, b) == ["end_time"]

    def test_add_location_when_original_has_none_fails(self):
        a = make_activity()
        a.location = None
        b = copy.deepcopy(a)
        b.location = Location(lat=39.9, lng=116.4)
        assert not activity_fields_match(a, b)

    def test_remove_location_when_original_has_one_fails(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.location = None
        assert not activity_fields_match(a, b)


class TestAllowedChanges:
    """Decorative fields can be freely modified."""

    def test_add_recommendation_reason_passes(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.recommendation_reason = "世界文化遗产，必去"
        assert activity_fields_match(a, b)

    def test_add_tags_passes(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.tags = ["历史", "文化"]
        assert activity_fields_match(a, b)

    def test_add_close_time_passes(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.close_time = "17:00"
        assert activity_fields_match(a, b)

    def test_modify_category_passes(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.category = "restaurant"
        assert activity_fields_match(a, b)

    def test_modify_meal_cost_passes(self):
        a = make_activity()
        b = copy.deepcopy(a)
        b.meal_cost = 200
        assert activity_fields_match(a, b)


class TestIdentity:
    def test_identical_activity_matches(self):
        a = make_activity()
        assert activity_fields_match(a, a)

    def test_deepcopy_matches(self):
        a = make_activity()
        b = copy.deepcopy(a)
        assert activity_fields_match(a, b)

    def test_interview_alias_matches(self):
        a = make_activity()
        b = copy.deepcopy(a)
        assert protected_fields_match(a, b)
        assert protected_field_differences(a, b) == []
