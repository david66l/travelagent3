"""Unit tests for Phase 2C Writer."""
import copy

import pytest

from schemas import Activity, DayPlan, Location, UserProfile
from planner.core.writer import enrich, enrich_safe
from planner.core.fact_checksum import verify_checksum


@pytest.fixture
def profile_shanghai():
    return UserProfile(
        destination="上海",
        travel_days=2,
        travel_dates="2026-06-01",
        travelers_type="情侣",
        budget_range=1000,
        interests=["历史", "文化"],
        food_preferences=["本帮菜"],
    )


@pytest.fixture
def itinerary(profile_shanghai):
    day1 = DayPlan(
        day_number=1,
        activities=[
            Activity(
                poi_name="外滩", category="attraction",
                start_time="09:00", end_time="11:00", duration_min=120,
                ticket_price=0,
                location=Location(lat=31.24, lng=121.50),
            ),
            Activity(
                poi_name="豫园", category="attraction",
                start_time="11:30", end_time="13:30", duration_min=120,
                ticket_price=40,
                location=Location(lat=31.23, lng=121.49),
            ),
            Activity(
                poi_name="Lunch", category="restaurant",
                start_time="13:30", end_time="15:00", duration_min=90,
                meal_cost=80,
            ),
        ],
        total_cost=120,
    )
    day2 = DayPlan(
        day_number=2,
        activities=[
            Activity(
                poi_name="上海博物馆", category="attraction",
                start_time="09:00", end_time="11:30", duration_min=150,
                ticket_price=0,
                location=Location(lat=31.23, lng=121.47),
            ),
        ],
        total_cost=0,
    )
    return [day1, day2]


# --------------------------------------------------------------------------- #
# Tests: writer preserves facts
# --------------------------------------------------------------------------- #


class TestWriterPreservesFacts:
    def test_enrich_passes_checksum(self, itinerary, profile_shanghai):
        enriched, proposal = enrich(itinerary, profile_shanghai)
        assert verify_checksum(itinerary, enriched)

    def test_enrich_does_not_mutate_original(self, itinerary, profile_shanghai):
        original_checksum = copy.deepcopy(itinerary)
        enrich(itinerary, profile_shanghai)
        assert verify_checksum(original_checksum, itinerary)

    def test_enrich_safe_always_passes_checksum(self, itinerary, profile_shanghai):
        enriched, proposal = enrich_safe(itinerary, profile_shanghai)
        assert verify_checksum(itinerary, enriched)


# --------------------------------------------------------------------------- #
# Tests: writer adds decoration (not facts)
# --------------------------------------------------------------------------- #


class TestWriterDecoration:
    def test_adds_day_themes(self, itinerary, profile_shanghai):
        enriched, _ = enrich(itinerary, profile_shanghai)
        # At least one day should have a theme assigned
        themes = [d.theme for d in enriched if d.theme]
        assert len(themes) > 0

    def test_adds_recommendation_reasons(self, itinerary, profile_shanghai):
        enriched, _ = enrich(itinerary, profile_shanghai)
        reasons = [a.recommendation_reason for d in enriched for a in d.activities if a.recommendation_reason]
        assert len(reasons) > 0

    def test_proposal_includes_destination(self, itinerary, profile_shanghai):
        _, proposal = enrich(itinerary, profile_shanghai)
        assert "上海" in proposal

    def test_proposal_includes_budget(self, itinerary, profile_shanghai):
        _, proposal = enrich(itinerary, profile_shanghai)
        assert "¥" in proposal

    def test_proposal_includes_activity_names(self, itinerary, profile_shanghai):
        _, proposal = enrich(itinerary, profile_shanghai)
        assert "外滩" in proposal
        assert "豫园" in proposal
        assert "上海博物馆" in proposal


# --------------------------------------------------------------------------- #
# Tests: writer cannot change facts
# --------------------------------------------------------------------------- #


class TestWriterCantChangeFacts:
    def test_enrich_result_has_same_poi_names(self, itinerary, profile_shanghai):
        enriched, _ = enrich(itinerary, profile_shanghai)
        orig_names = {(d.day_number, a.poi_name) for d in itinerary for a in d.activities}
        enriched_names = {(d.day_number, a.poi_name) for d in enriched for a in d.activities}
        assert orig_names == enriched_names

    def test_enrich_result_has_same_durations(self, itinerary, profile_shanghai):
        enriched, _ = enrich(itinerary, profile_shanghai)
        for orig_day, enr_day in zip(itinerary, enriched):
            for orig_act, enr_act in zip(orig_day.activities, enr_day.activities):
                assert orig_act.duration_min == enr_act.duration_min


# --------------------------------------------------------------------------- #
# Tests: edge cases
# --------------------------------------------------------------------------- #


class TestWriterEdgeCases:
    def test_empty_itinerary(self, profile_shanghai):
        enriched, proposal = enrich([], profile_shanghai)
        assert enriched == []
        assert len(proposal) > 0

    def test_activity_without_tags(self, profile_shanghai):
        day = DayPlan(day_number=1, activities=[
            Activity(poi_name="某景点", category="attraction",
                     start_time="09:00", end_time="10:00", duration_min=60),
        ])
        enriched, proposal = enrich([day], profile_shanghai)
        assert verify_checksum([day], enriched)
        assert proposal  # not empty
