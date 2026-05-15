"""Unit tests for Rule Validator."""

import pytest

from schemas import DayPlan, Activity, UserProfile
from planner.core.rule_validator import validate
from planner.core.models import RuleViolation


class TestRuleValidator:
    def test_empty_itinerary_passes(self):
        report = validate([], UserProfile(destination="北京", travel_days=1), [])
        assert report.passed is True
        assert report.hard_violations == []

    def test_overlapping_activities_is_hard_violation(self):
        day = DayPlan(
            day_number=1,
            activities=[
                Activity(poi_name="故宫", start_time="09:00", end_time="11:00", duration_min=120),
                Activity(poi_name="天坛", start_time="10:00", end_time="12:00", duration_min=120),
            ],
        )
        report = validate([day], UserProfile(destination="北京", travel_days=1), [])
        assert report.passed is False
        assert any(v.rule == "time_feasibility" for v in report.hard_violations)

    def test_missing_must_see_is_hard_violation(self):
        day = DayPlan(
            day_number=1,
            activities=[
                Activity(poi_name="天坛", start_time="09:00", end_time="11:00", duration_min=120),
            ],
        )
        report = validate([day], UserProfile(destination="北京", travel_days=1), ["故宫"])
        assert report.passed is False
        assert any(v.rule == "must_see_presence" for v in report.hard_violations)

    def test_budget_over_120_percent_is_hard_violation(self):
        day = DayPlan(
            day_number=1,
            activities=[
                Activity(poi_name="故宫", duration_min=120, ticket_price=500),
            ],
            total_cost=500,
        )
        report = validate([day], UserProfile(destination="北京", travel_days=1, budget_range=100), [])
        assert report.passed is False
        assert any(v.rule == "budget_compliance" for v in report.hard_violations)

    def test_opening_hours_missing_data_no_warning(self):
        """If open_time/close_time is missing, no soft warning."""
        day = DayPlan(
            day_number=1,
            activities=[
                Activity(poi_name="故宫", start_time="09:00", end_time="11:00", duration_min=120),
            ],
        )
        report = validate([day], UserProfile(destination="北京", travel_days=1), [])
        assert report.passed is True
        assert not any(v.rule == "opening_hours" for v in report.soft_warnings)

    def test_opening_hours_outside_range_is_soft_warning(self):
        day = DayPlan(
            day_number=1,
            activities=[
                Activity(
                    poi_name="故宫",
                    start_time="09:00",
                    end_time="11:00",
                    duration_min=120,
                    open_time="10:00",
                    close_time="17:00",
                ),
            ],
        )
        report = validate([day], UserProfile(destination="北京", travel_days=1), [])
        assert report.passed is True  # soft only
        assert any(v.rule == "opening_hours" for v in report.soft_warnings)

    def test_hard_vs_soft_separation(self):
        """Hard violations block; soft warnings don't."""
        day = DayPlan(
            day_number=1,
            activities=[
                Activity(
                    poi_name="故宫",
                    start_time="09:00",
                    end_time="11:00",
                    duration_min=120,
                    open_time="10:00",
                    close_time="17:00",
                ),
            ],
        )
        report = validate([day], UserProfile(destination="北京", travel_days=1), [])
        assert report.passed is True
        assert len(report.hard_violations) == 0
        assert len(report.soft_warnings) == 1

    def test_activity_outside_day_bounds_is_hard(self):
        day = DayPlan(
            day_number=1,
            activities=[
                Activity(poi_name="故宫", start_time="22:00", end_time="23:00", duration_min=60),
            ],
        )
        report = validate([day], UserProfile(destination="北京", travel_days=1), [])
        assert report.passed is False
        assert any(v.rule == "time_feasibility" for v in report.hard_violations)
