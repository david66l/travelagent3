"""Unit tests for FeasibilityChecker."""

from agents.feasibility import FeasibilityChecker
from models.travel_slots import TravelSlots


class TestFeasibilityChecker:
    def test_feasible_when_basic_info_present(self):
        slots = TravelSlots(destination="成都", travel_days=3, travelers_count=2)
        report = FeasibilityChecker.check(slots)
        assert report["feasible"] is True
        assert report["budget_fit"] == "ok"

    def test_infeasible_when_missing_destination(self):
        slots = TravelSlots(travel_days=3)
        report = FeasibilityChecker.check(slots)
        assert report["feasible"] is False
        assert "缺少目的地" in report["issues"]

    def test_infeasible_when_missing_days(self):
        slots = TravelSlots(destination="成都")
        report = FeasibilityChecker.check(slots)
        assert report["feasible"] is False
        assert "缺少旅行天数" in report["issues"]

    def test_budget_over(self):
        slots = TravelSlots(
            destination="北京",
            travel_days=5,
            travelers_count=2,
            total_budget=2000,
        )
        report = FeasibilityChecker.check(slots)
        assert report["budget_fit"] == "over"
        assert not report["feasible"]
        assert "预算偏低" in report["issues"][0]

    def test_budget_tight(self):
        slots = TravelSlots(
            destination="北京",
            travel_days=2,
            travelers_count=1,
            total_budget=1000,
        )
        report = FeasibilityChecker.check(slots)
        assert report["budget_fit"] == "tight"
        assert "预算较紧" in report["warnings"][0]

    def test_pregnant_with_family_warns(self):
        slots = TravelSlots(
            destination="三亚",
            travel_days=4,
            has_pregnant=True,
            travel_companion="family",
        )
        report = FeasibilityChecker.check(slots)
        assert any("孕妇" in w for w in report["warnings"])

    def test_elderly_with_intensive_is_infeasible(self):
        slots = TravelSlots(
            destination="北京",
            travel_days=3,
            has_elderly=True,
            pace="intensive",
        )
        report = FeasibilityChecker.check(slots)
        assert not report["feasible"]
        assert any("老人" in i for i in report["issues"])

    def test_wheelchair_walk_warning(self):
        slots = TravelSlots(
            destination="北京",
            travel_days=3,
            has_wheelchair=True,
            max_walk_minutes=180,
        )
        report = FeasibilityChecker.check(slots)
        assert any("轮椅" in w for w in report["warnings"])
