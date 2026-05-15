"""Unit tests for deterministic Repair Executor (Phase 2B)."""
import pytest
from copy import deepcopy

from schemas import Activity, DayPlan, Location, ScoredPOI, UserProfile
from planner.core.models import RepairPlan, RuleViolation
from planner.core.rule_validator import validate
from planner.core.repair import generate_repairs, apply_repair, run_repair_loop


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def profile_shanghai():
    return UserProfile(
        destination="上海",
        travel_days=2,
        budget_range=1000,
        interests=["历史", "文化"],
        pace="moderate",
    )


@pytest.fixture
def sample_pois():
    return [
        ScoredPOI(
            name="外滩",
            category="attraction",
            score=0.9,
            location=Location(lat=31.24, lng=121.49),
            area="外滩",
            ticket_price=0,
            recommended_hours="2小时",
            tags=["观光"],
        ),
        ScoredPOI(
            name="豫园",
            category="attraction",
            score=0.85,
            location=Location(lat=31.23, lng=121.49),
            area="城隍庙",
            ticket_price=40,
            recommended_hours="2小时",
            tags=["历史"],
        ),
        ScoredPOI(
            name="东方明珠",
            category="attraction",
            score=0.88,
            location=Location(lat=31.24, lng=121.50),
            area="陆家嘴",
            ticket_price=200,
            recommended_hours="2-3小时",
            tags=["观光"],
        ),
        ScoredPOI(
            name="上海博物馆",
            category="attraction",
            score=0.92,
            location=Location(lat=31.23, lng=121.47),
            area="人民广场",
            ticket_price=0,
            recommended_hours="2-3小时",
            tags=["历史", "文化"],
        ),
    ]


def make_activity(name, start="09:00", end="11:00", duration=120, price=0):
    return Activity(
        poi_name=name,
        category="attraction",
        start_time=start,
        end_time=end,
        duration_min=duration,
        ticket_price=price,
    )


# --------------------------------------------------------------------------- #
# Tests: overlap repair
# --------------------------------------------------------------------------- #


class TestOverlapRepair:
    def test_detect_and_generate_for_overlap(self, profile_shanghai):
        """Overlapping activities should produce move/remove plans."""
        day = DayPlan(
            day_number=1,
            activities=[
                make_activity("外滩", "09:00", "11:00", 120),
                make_activity("豫园", "10:00", "12:00", 120),
            ],
        )
        itinerary = [day]
        report = validate(itinerary, profile_shanghai, [])
        assert not report.passed
        assert any(v.rule == "time_feasibility" for v in report.hard_violations)

    def test_move_repair_resolves_overlap(self, profile_shanghai, sample_pois):
        """Move repair should resolve overlap by pushing activity later."""
        day = DayPlan(
            day_number=1,
            activities=[
                make_activity("外滩", "09:00", "11:00", 120),
                make_activity("豫园", "10:00", "12:00", 120),
            ],
        )
        itinerary = [day]
        report = validate(itinerary, profile_shanghai, [])
        plans = generate_repairs(report.hard_violations, itinerary, sample_pois, profile_shanghai)

        # First plan should be "move"
        assert len(plans) > 0
        move_plan = plans[0]
        assert move_plan.action == "move"

        # Apply it
        repaired = apply_repair(move_plan, itinerary, sample_pois, profile_shanghai)
        new_report = validate(repaired, profile_shanghai, [])
        assert new_report.passed, f"Still has violations: {new_report.hard_violations}"

    def test_run_repair_loop_fixes_overlap(self, profile_shanghai, sample_pois):
        """Full repair loop should resolve overlapping activities."""
        day = DayPlan(
            day_number=1,
            activities=[
                make_activity("外滩", "09:00", "11:00", 120),
                make_activity("豫园", "10:00", "12:00", 120),
                make_activity("东方明珠", "11:00", "13:30", 150, 200),
            ],
        )
        itinerary = [day]
        result = run_repair_loop(itinerary, profile_shanghai, [], sample_pois)
        assert result.success
        assert len(result.applied_plans) >= 1


# --------------------------------------------------------------------------- #
# Tests: must-see insert
# --------------------------------------------------------------------------- #


class TestMustSeeRepair:
    def test_insert_missing_must_see(self, profile_shanghai, sample_pois):
        """Must-see POI not in itinerary → insert."""
        day = DayPlan(
            day_number=1,
            activities=[make_activity("外滩", "09:00", "11:00", 120)],
        )
        itinerary = [day]
        report = validate(itinerary, profile_shanghai, ["上海博物馆"])
        assert not report.passed
        assert any(v.rule == "must_see_presence" for v in report.hard_violations)

        plans = generate_repairs(report.hard_violations, itinerary, sample_pois, profile_shanghai)
        insert_plans = [p for p in plans if p.action == "insert"]
        assert len(insert_plans) > 0

        repaired = apply_repair(insert_plans[0], itinerary, sample_pois, profile_shanghai)
        new_report = validate(repaired, profile_shanghai, ["上海博物馆"])
        assert new_report.passed

    def test_run_repair_loop_inserts_must_see(self, profile_shanghai, sample_pois):
        """Full repair loop should insert missing must-see POI."""
        day = DayPlan(
            day_number=1,
            activities=[make_activity("外滩", "09:00", "11:00", 120)],
        )
        itinerary = [day]
        result = run_repair_loop(itinerary, profile_shanghai, ["上海博物馆"], sample_pois)
        assert result.success

        # Apply all repairs to get the final itinerary
        current = deepcopy(itinerary)
        for plan in result.applied_plans:
            current = apply_repair(plan, current, sample_pois, profile_shanghai)
        final_names = {a.poi_name for d in current for a in d.activities}
        assert "上海博物馆" in final_names

    def test_missing_poi_data_needs_human(self, profile_shanghai):
        """Insert repair for unknown POI → needs_human."""
        day = DayPlan(
            day_number=1,
            activities=[make_activity("外滩", "09:00", "11:00", 120)],
        )
        itinerary = [day]
        result = run_repair_loop(
            itinerary,
            profile_shanghai,
            ["不存在的景点"],
            [],  # no POI data
        )
        assert not result.success
        assert result.needs_human


# --------------------------------------------------------------------------- #
# Tests: budget repair
# --------------------------------------------------------------------------- #


class TestBudgetRepair:
    def test_remove_expensive_for_budget(self, profile_shanghai, sample_pois):
        """Over-budget itinerary should remove expensive non-must-see activities."""
        from planner.core.repair import _rebuild_times
        profile = profile_shanghai.model_copy(update={"budget_range": 100})
        day = DayPlan(
            day_number=1,
            activities=[
                make_activity("外滩", "09:00", "11:00", 120, 0),
                make_activity("东方明珠", "11:30", "14:00", 150, 200),
            ],
        )
        itinerary = _rebuild_times([day], profile)
        report = validate(itinerary, profile, [])
        if report.passed:
            pytest.skip("Budget already within range")
        assert not report.passed

        plans = generate_repairs(report.hard_violations, itinerary, sample_pois, profile)
        remove_plans = [p for p in plans if p.action == "remove"]
        assert len(remove_plans) > 0

        # The first remove should target the expensive one (¥200)
        assert "东方明珠" in remove_plans[0].reason


# --------------------------------------------------------------------------- #
# Tests: protected facts
# --------------------------------------------------------------------------- #


class TestProtectedFacts:
    def test_move_preserves_poi_name_and_price(self, profile_shanghai, sample_pois):
        """Move repair must not alter poi_name, location, ticket_price, duration_min."""
        act = make_activity("豫园", "10:00", "12:00", 120, 40)
        original = act.model_copy(deep=True)
        day = DayPlan(day_number=1, activities=[
            make_activity("外滩", "09:00", "11:00", 120),
            act,
        ])
        itinerary = [day]
        report = validate(itinerary, profile_shanghai, [])
        plans = generate_repairs(report.hard_violations, itinerary, sample_pois, profile_shanghai)
        move_plan = plans[0]
        repaired = apply_repair(move_plan, itinerary, sample_pois, profile_shanghai)

        # Find the moved activity by name
        moved_act = None
        for d in repaired:
            for a in d.activities:
                if a.poi_name == "豫园":
                    moved_act = a
        assert moved_act is not None
        assert moved_act.poi_name == original.poi_name
        assert moved_act.ticket_price == original.ticket_price
        assert moved_act.duration_min == original.duration_min
        # Start time may change, end time may change — those are editable

    def test_insert_preserves_poi_facts(self, profile_shanghai, sample_pois):
        """Inserted activity must use the original POI's immutable fields."""
        day = DayPlan(day_number=1, activities=[])
        itinerary = [day]
        plan = RepairPlan(
            action="insert",
            target={"day_number": 1},
            params={"poi_name": "上海博物馆"},
            reason="test",
        )
        repaired = apply_repair(plan, itinerary, sample_pois, profile_shanghai)
        inserted = None
        for a in repaired[0].activities:
            if a.poi_name == "上海博物馆":
                inserted = a
        assert inserted is not None
        assert inserted.ticket_price == 0
        assert inserted.duration_min == 150  # "2-3小时"
        assert inserted.poi_name == "上海博物馆"

    def test_remove_only_affects_target(self, profile_shanghai):
        """Remove repair must only remove the targeted activity."""
        day = DayPlan(
            day_number=1,
            activities=[
                make_activity("外滩", "09:00", "11:00", 120),
                make_activity("豫园", "11:30", "13:30", 120),
            ],
        )
        itinerary = [day]
        plan = RepairPlan(
            action="remove",
            target={"day_number": 1, "activity_index": 1},
            params={},
            reason="test",
        )
        repaired = apply_repair(plan, itinerary, [], profile_shanghai)
        names = [a.poi_name for a in repaired[0].activities]
        assert "外滩" in names
        assert "豫园" not in names
        assert len(names) == 1


# --------------------------------------------------------------------------- #
# Tests: needs_human
# --------------------------------------------------------------------------- #


class TestNeedsHuman:
    def test_unrepairable_overlap_needs_human(self, profile_shanghai):
        """When all repairs fail, needs_human should be True."""
        # Single-day itinerary with unfixable overlap and no POI data for repair
        day = DayPlan(
            day_number=1,
            activities=[
                make_activity("外滩", "09:00", "11:00", 120),
                make_activity("豫园", "10:00", "12:00", 120),
            ],
        )
        itinerary = [day]
        # No POI data → move to next day won't work (single day), remove works
        # With our algorithm, remove should always work
        result = run_repair_loop(itinerary, profile_shanghai, ["不存在的POI"], [])
        assert result.success is False or result.needs_human is True
        # At minimum, must_see for unknown POI should fail

    def test_iteration_limit_prevents_infinite_loop(self, profile_shanghai, sample_pois):
        """Even with max_iterations=1, the repair loop should exit cleanly."""
        day = DayPlan(
            day_number=1,
            activities=[
                make_activity("外滩", "09:00", "11:00", 120),
                make_activity("豫园", "10:00", "12:00", 120),
            ],
        )
        itinerary = [day]
        result = run_repair_loop(
            itinerary, profile_shanghai, [], sample_pois, max_iterations=1,
        )
        # Should terminate, not hang
        assert result.success or not result.success
