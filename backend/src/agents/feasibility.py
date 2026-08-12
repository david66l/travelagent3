"""Feasibility checker for travel demand slots."""

from __future__ import annotations

import logging
from typing import Any, Optional

from models.travel_slots import TravelSlots

logger = logging.getLogger(__name__)


class FeasibilityReport(dict):
    """Lightweight feasibility report dict.

    Keys:
        - feasible: bool
        - issues: list[str]
        - warnings: list[str]
        - budget_fit: str  # ok / tight / over
        - crowd_risk: str  # low / medium / high
    """

    def __init__(self, **kwargs: Any):
        defaults = {
            "feasible": True,
            "issues": [],
            "warnings": [],
            "budget_fit": "ok",
            "crowd_risk": "low",
        }
        defaults.update(kwargs)
        super().__init__(defaults)


class FeasibilityChecker:
    """Check whether user demand slots are feasible."""

    # Approximate daily cost per person (CNY) — MVP hard-coded lookup.
    # In production this should come from city_info.daily_avg_cost.
    CITY_DAILY_COST: dict[str, float] = {
        "北京": 800,
        "上海": 900,
        "广州": 700,
        "深圳": 800,
        "成都": 600,
        "杭州": 700,
        "西安": 550,
        "重庆": 600,
        "苏州": 650,
        "南京": 650,
        "厦门": 700,
        "青岛": 600,
        "大理": 500,
        "丽江": 500,
        "三亚": 1000,
        "长沙": 550,
        "武汉": 550,
        "昆明": 500,
        "桂林": 500,
        "拉萨": 700,
    }

    @classmethod
    def check(
        cls,
        slots: TravelSlots,
        *,
        city_daily_cost: Optional[float] = None,
        must_visit_spots: Optional[list[dict[str, Any]]] = None,
        travel_dates: Optional[str] = None,
    ) -> FeasibilityReport:
        """Run feasibility checks and return a report."""
        report = FeasibilityReport()

        cls._check_basic(slots, report)
        cls._check_budget(slots, report, city_daily_cost)
        cls._check_people(slots, report)
        cls._check_pace(slots, report)
        cls._check_reservations(slots, report, must_visit_spots or [])
        cls._check_seasonal_closures(slots, report, must_visit_spots or [], travel_dates)

        report["feasible"] = not report["issues"]
        return report

    @classmethod
    def _check_basic(cls, slots: TravelSlots, report: FeasibilityReport) -> None:
        if not slots.destination:
            report["issues"].append("缺少目的地")
        if not slots.travel_days:
            report["issues"].append("缺少旅行天数")

    @classmethod
    def _check_budget(
        cls,
        slots: TravelSlots,
        report: FeasibilityReport,
        city_daily_cost: Optional[float],
    ) -> None:
        if not slots.travelers_count or not slots.travel_days:
            return

        daily = city_daily_cost or cls.CITY_DAILY_COST.get(slots.destination, 700)
        travelers = slots.travelers_count
        estimated_total = daily * slots.travel_days * travelers

        budget = slots.total_budget
        if budget is None and slots.budget_per_person is not None:
            budget = slots.budget_per_person * travelers

        if budget is None:
            report["warnings"].append("未提供预算，无法做预算匹配")
            return

        ratio = budget / estimated_total if estimated_total > 0 else 1.0
        if ratio < 0.6:
            report["budget_fit"] = "over"
            report["issues"].append(
                f"预算偏低：{slots.destination} {slots.travel_days}天预计至少"
                f" {estimated_total:.0f} 元，当前预算 {budget:.0f} 元"
            )
        elif ratio < 0.85:
            report["budget_fit"] = "tight"
            report["warnings"].append(
                f"预算较紧：{slots.destination} {slots.travel_days}天预计 {estimated_total:.0f} 元"
            )

    @classmethod
    def _check_people(cls, slots: TravelSlots, report: FeasibilityReport) -> None:
        if slots.has_pregnant and slots.travel_companion == "family":
            report["warnings"].append("孕妇同行，建议选择轻松节奏并避免高海拔/激烈项目")
        if slots.has_elderly and slots.pace == "intensive":
            report["issues"].append("老人同行时不适合特种兵式紧凑行程")
        if slots.has_wheelchair and slots.max_walk_minutes and slots.max_walk_minutes > 120:
            report["warnings"].append("轮椅出行建议单日步行控制在 120 分钟以内")

    @classmethod
    def _check_pace(cls, slots: TravelSlots, report: FeasibilityReport) -> None:
        if slots.travel_days and slots.travel_days > 7 and slots.pace == "intensive":
            report["warnings"].append("多日紧凑行程易造成疲劳，建议适当放缓")
        if slots.fatigue_preference == "low" and slots.pace == "intensive":
            report["issues"].append("疲劳接受度低与紧凑行程冲突")

    @classmethod
    def _check_reservations(
        cls,
        slots: TravelSlots,
        report: FeasibilityReport,
        must_visit_spots: list[dict[str, Any]],
    ) -> None:
        if not must_visit_spots:
            return

        reservation_spots: list[str] = []
        for spot in must_visit_spots:
            name = spot.get("name") or spot.get("title") or "未知景点"
            need = (
                spot.get("need_reservation")
                or spot.get("reservation_required")
                or spot.get("reservation_advance_days")
            )
            if need:
                advance = spot.get("reservation_advance_days") or 1
                reservation_spots.append(f"{name}（需提前{advance}天预约）")

        if reservation_spots:
            report["warnings"].append("必去景点中以下需要预约：" + "、".join(reservation_spots))

    @classmethod
    def _check_seasonal_closures(
        cls,
        slots: TravelSlots,
        report: FeasibilityReport,
        must_visit_spots: list[dict[str, Any]],
        travel_dates: Optional[str],
    ) -> None:
        if not must_visit_spots:
            return

        closed_spots: list[str] = []
        for spot in must_visit_spots:
            name = spot.get("name") or spot.get("title") or "未知景点"
            restriction = spot.get("season_restriction")
            temp_closure = spot.get("temp_closure_dates")
            if restriction:
                closed_spots.append(f"{name}（季节限制：{restriction}）")
            elif temp_closure:
                closed_spots.append(f"{name}（临时闭园：{temp_closure}）")

        if closed_spots:
            msg = "必去景点存在闭园/季节限制：" + "、".join(closed_spots)
            if travel_dates:
                report["warnings"].append(msg + "，请核对出行日期")
            else:
                report["warnings"].append(msg)
