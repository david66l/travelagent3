"""Versioned, programmatic itinerary validation.

The validator reads itinerary and environment facts only. It never trusts a
model-authored claim that constraints passed, which makes its output suitable
for online gates, offline evaluation and reward computation.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta
from statistics import pstdev
from typing import Any

from pydantic import BaseModel, Field


VALIDATOR_VERSION = "travel-validator.v1"
_NON_POI_CATEGORIES = {"meal", "restaurant", "hotel", "transport", "rest"}
_GENERIC_NAMES = {"早餐", "午餐", "晚餐", "酒店", "休息", "自由活动"}


class ConstraintViolation(BaseModel):
    code: str
    message: str
    day_number: int | None = None
    poi_id: str | None = None
    poi_name: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    validator_version: str = VALIDATOR_VERSION
    hard_pass: bool
    hard_violations: list[ConstraintViolation] = Field(default_factory=list)
    soft_scores: dict[str, float] = Field(default_factory=dict)
    metrics: dict[str, int | float] = Field(default_factory=dict)


def _minutes(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        parsed = datetime.strptime(str(value), "%H:%M")
    except ValueError:
        return None
    return parsed.hour * 60 + parsed.minute


def _activity_key(activity: dict[str, Any]) -> str:
    return str(activity.get("poi_id") or activity.get("poi_name") or "").strip()


def _identity(value: Any) -> str:
    return "".join(str(value or "").strip().lower().split())


def _iso_date(value: Any) -> date | None:
    if not value:
        return None
    normalized = str(value).strip().replace("年", "-").replace("月", "-").replace("日", "")
    normalized = normalized.replace("/", "-").replace(".", "-")
    try:
        return date.fromisoformat(normalized)
    except ValueError:
        return None


def _facts_by_key(facts: dict[str, Any] | list[dict[str, Any]] | None) -> dict[str, dict]:
    if isinstance(facts, dict):
        items = facts.get("pois", facts)
        if isinstance(items, dict):
            return {str(key): value for key, value in items.items() if isinstance(value, dict)}
    elif isinstance(facts, list):
        items = facts
    else:
        items = []

    indexed: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        for key in (item.get("id"), item.get("poi_id"), item.get("name"), item.get("poi_name")):
            if key:
                indexed[str(key)] = item
    return indexed


class ItineraryValidator:
    """Validate hard constraints and calculate deterministic quality features."""

    version = VALIDATOR_VERSION

    def validate(
        self,
        itinerary: list[dict[str, Any]],
        constraints: dict[str, Any] | BaseModel | None = None,
        facts: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> ValidationReport:
        config = (
            constraints.model_dump()
            if isinstance(constraints, BaseModel)
            else dict(constraints or {})
        )
        fact_index = _facts_by_key(facts)
        violations: list[ConstraintViolation] = []

        expected_days = int(config.get("travel_days") or 0)
        if not itinerary:
            violations.append(ConstraintViolation(code="EMPTY_ITINERARY", message="行程不能为空"))
        elif expected_days and len(itinerary) != expected_days:
            violations.append(
                ConstraintViolation(
                    code="TRAVEL_DAY_COUNT_MISMATCH",
                    message=f"计划天数 {len(itinerary)} 与要求 {expected_days} 不一致",
                    details={"actual": len(itinerary), "expected": expected_days},
                )
            )

        default_day_start = int(config.get("day_start_min") or 8 * 60)
        default_day_end = int(config.get("day_end_min") or 21 * 60)
        daily_starts = [int(value) for value in config.get("daily_start_minutes") or []]
        daily_ends = [int(value) for value in config.get("daily_end_minutes") or []]
        trip_start = _iso_date(config.get("trip_start_date"))
        all_poi_keys: list[str] = []
        scheduled_keys: set[str] = set()
        scheduled_identities: set[str] = set()
        scheduled_records: dict[str, dict[str, Any]] = {}
        total_transit = 0
        total_cost = 0.0
        daily_loads: list[float] = []
        categories: list[str] = []
        matched_interests = 0
        interest_activities = 0
        interests = set(config.get("interests") or [])

        for fallback_day_number, day in enumerate(itinerary, 1):
            day_number = int(day.get("day_number") or fallback_day_number)
            day_index = max(0, day_number - 1)
            day_start = (
                daily_starts[day_index]
                if day_index < len(daily_starts) and daily_starts[day_index] > 0
                else default_day_start
            )
            day_end = (
                daily_ends[day_index]
                if day_index < len(daily_ends) and daily_ends[day_index] > 0
                else default_day_end
            )
            actual_day_date = _iso_date(day.get("date"))
            if actual_day_date is None and trip_start is not None:
                actual_day_date = trip_start + timedelta(days=day_index)
            activities = [a for a in day.get("activities", []) if isinstance(a, dict)]
            intervals: list[tuple[int, int, dict[str, Any]]] = []
            computed_day_cost = 0.0
            day_transit = int(day.get("total_transit_time_min") or 0)
            day_load = float(day.get("total_walking_minutes") or 0)
            dining_activities = 0
            previous_dining_end: int | None = None

            for activity in activities:
                poi_name = str(activity.get("poi_name") or "").strip()
                poi_id = str(activity.get("poi_id") or "").strip() or None
                category = str(activity.get("category") or "attraction").lower()
                key = _activity_key(activity)
                fact = fact_index.get(str(poi_id or "")) or fact_index.get(poi_name) or {}

                start = _minutes(activity.get("start_time"))
                end = _minutes(activity.get("end_time"))
                if category in {"restaurant", "meal"}:
                    dining_activities += 1
                    # Lunch and dinner can legitimately be adjacent in the
                    # activity list when several hours of free time separate
                    # them. Only flag overlapping or nearly back-to-back meals.
                    if (
                        previous_dining_end is not None
                        and start is not None
                        and start - previous_dining_end < 60
                    ):
                        violations.append(
                            ConstraintViolation(
                                code="CONSECUTIVE_DINING_ACTIVITIES",
                                message="同一天不能紧接着安排多个餐饮活动",
                                day_number=day_number,
                                poi_id=poi_id,
                                poi_name=poi_name or None,
                            )
                        )
                    if end is not None:
                        previous_dining_end = end
                if start is None or end is None or end <= start:
                    violations.append(
                        ConstraintViolation(
                            code="INVALID_ACTIVITY_TIME",
                            message=f"{poi_name or '活动'} 的起止时间无效",
                            day_number=day_number,
                            poi_id=poi_id,
                            poi_name=poi_name or None,
                        )
                    )
                else:
                    intervals.append((start, end, activity))
                    if start < day_start or end > day_end:
                        violations.append(
                            ConstraintViolation(
                                code="DAY_TIME_BOUNDARY_EXCEEDED",
                                message=f"{poi_name} 超出每日活动时间边界",
                                day_number=day_number,
                                poi_id=poi_id,
                                poi_name=poi_name,
                                details={"start": start, "end": end},
                            )
                        )
                    date_hours = {}
                    if actual_day_date is not None:
                        date_hours = (fact.get("date_opening_hours") or {}).get(
                            actual_day_date.isoformat()
                        ) or {}
                    if isinstance(date_hours, (list, tuple)) and len(date_hours) == 2:
                        open_min = _minutes(date_hours[0])
                        close_min = _minutes(date_hours[1])
                    else:
                        open_min = _minutes(fact.get("open_time") or activity.get("open_time"))
                        close_min = _minutes(fact.get("close_time") or activity.get("close_time"))
                    if (
                        open_min is not None
                        and close_min is not None
                        and (start < open_min or end > close_min)
                    ):
                        violations.append(
                            ConstraintViolation(
                                code="POI_CLOSED_DURING_VISIT",
                                message=f"{poi_name} 的活动时间不在营业时间内",
                                day_number=day_number,
                                poi_id=poi_id,
                                poi_name=poi_name,
                            )
                        )

                if fact.get("closed_weekdays") and actual_day_date is not None:
                    weekday = actual_day_date.weekday()
                    if weekday is not None and weekday in fact["closed_weekdays"]:
                        violations.append(
                            ConstraintViolation(
                                code="POI_CLOSED_ON_DATE",
                                message=f"{poi_name} 在计划日期闭馆",
                                day_number=day_number,
                                poi_id=poi_id,
                                poi_name=poi_name,
                            )
                        )

                if actual_day_date is not None and actual_day_date.isoformat() in set(
                    fact.get("closed_dates") or []
                ):
                    violations.append(
                        ConstraintViolation(
                            code="POI_CLOSED_ON_DATE",
                            message=f"{poi_name} 在计划日期临时闭馆",
                            day_number=day_number,
                            poi_id=poi_id,
                            poi_name=poi_name,
                        )
                    )

                if fact.get("reservation_required") and fact.get("reservation_available") is False:
                    violations.append(
                        ConstraintViolation(
                            code="REQUIRED_RESERVATION_UNAVAILABLE",
                            message=f"{poi_name} 需要预约但当前不可预约",
                            day_number=day_number,
                            poi_id=poi_id,
                            poi_name=poi_name,
                        )
                    )

                if key:
                    scheduled_keys.add(key)
                    scheduled_identities.update(
                        identity
                        for identity in (_identity(poi_id), _identity(poi_name))
                        if identity
                    )
                    scheduled_records[key] = {
                        "day_number": day_number,
                        "start": start,
                        "end": end,
                        "poi_name": poi_name,
                    }
                if key and category not in _NON_POI_CATEGORIES and poi_name not in _GENERIC_NAMES:
                    all_poi_keys.append(key)
                if category:
                    categories.append(category)
                tags = set(activity.get("tags") or fact.get("tags") or [])
                if interests and category not in _NON_POI_CATEGORIES:
                    interest_activities += 1
                    if interests & tags:
                        matched_interests += 1

                transit = activity.get("transit_from_prev") or {}
                if isinstance(transit, dict):
                    day_transit += int(transit.get("duration_min") or 0)
                computed_day_cost += sum(
                    float(activity.get(field) or 0)
                    for field in ("ticket_price", "meal_cost", "transport_cost")
                )
                if start is not None and end is not None and end > start:
                    day_load += end - start

            intervals.sort(key=lambda item: item[0])
            for previous, current in zip(intervals, intervals[1:]):
                if current[0] < previous[1]:
                    current_activity = current[2]
                    violations.append(
                        ConstraintViolation(
                            code="ACTIVITY_TIME_OVERLAP",
                            message="同一天存在活动时间重叠",
                            day_number=day_number,
                            poi_id=current_activity.get("poi_id"),
                            poi_name=current_activity.get("poi_name"),
                        )
                    )

            max_transit = int(config.get("max_transit_minutes") or 0)
            if max_transit and day_transit > max_transit:
                violations.append(
                    ConstraintViolation(
                        code="MAX_TRANSIT_EXCEEDED",
                        message=f"第 {day_number} 天通勤 {day_transit} 分钟超过上限 {max_transit}",
                        day_number=day_number,
                        details={"actual": day_transit, "limit": max_transit},
                    )
                )
            max_meals = int(config.get("meals_per_day") or 0)
            if max_meals and dining_activities > max_meals:
                violations.append(
                    ConstraintViolation(
                        code="TOO_MANY_DINING_ACTIVITIES",
                        message=(
                            f"第 {day_number} 天安排了 {dining_activities} 个餐饮活动，"
                            f"超过上限 {max_meals}"
                        ),
                        day_number=day_number,
                        details={"actual": dining_activities, "limit": max_meals},
                    )
                )
            total_transit += day_transit
            total_cost += float(day.get("total_cost") or computed_day_cost)
            daily_loads.append(day_load)

        duplicate_count = sum(count - 1 for count in Counter(all_poi_keys).values() if count > 1)
        if duplicate_count:
            violations.append(
                ConstraintViolation(
                    code="DUPLICATE_POI_VISIT",
                    message=f"存在 {duplicate_count} 次非必要重复 POI 访问",
                    details={"count": duplicate_count},
                )
            )

        must_visit = {str(item) for item in config.get("must_visit") or []}
        missing = sorted(must_visit - scheduled_keys)
        for key in missing:
            violations.append(
                ConstraintViolation(
                    code="MUST_VISIT_MISSING",
                    message=f"必去 POI 未安排：{key}",
                    poi_id=key,
                )
            )

        for forbidden in config.get("must_not_visit") or []:
            forbidden_identity = _identity(forbidden)
            if forbidden_identity and any(
                forbidden_identity in actual or actual in forbidden_identity
                for actual in scheduled_identities
            ):
                violations.append(
                    ConstraintViolation(
                        code="MUST_NOT_VISIT_PRESENT",
                        message=f"禁去 POI 被安排：{forbidden}",
                        poi_name=str(forbidden),
                    )
                )

        for reservation in config.get("user_reservations") or []:
            if not isinstance(reservation, dict) or not reservation.get("poi_id"):
                continue
            poi_id = str(reservation["poi_id"])
            scheduled = scheduled_records.get(poi_id)
            if scheduled is None:
                continue  # MUST_VISIT_MISSING above is the canonical absence error.
            expected_date = _iso_date(reservation.get("date"))
            if trip_start is not None and expected_date is not None:
                actual_date = trip_start + timedelta(days=int(scheduled["day_number"]) - 1)
                if actual_date != expected_date:
                    violations.append(
                        ConstraintViolation(
                            code="FIXED_EVENT_DATE_MISMATCH",
                            message=f"固定活动 {scheduled['poi_name']} 被安排到了错误日期",
                            day_number=int(scheduled["day_number"]),
                            poi_id=poi_id,
                            details={
                                "actual": actual_date.isoformat(),
                                "expected": expected_date.isoformat(),
                            },
                        )
                    )
            expected_start = _minutes(reservation.get("start_time"))
            if expected_start is not None and scheduled.get("start") != expected_start:
                violations.append(
                    ConstraintViolation(
                        code="FIXED_EVENT_TIME_MISMATCH",
                        message=f"固定活动 {scheduled['poi_name']} 的开始时间被改变",
                        day_number=int(scheduled["day_number"]),
                        poi_id=poi_id,
                        details={"actual": scheduled.get("start"), "expected": expected_start},
                    )
                )
            expected_end = _minutes(reservation.get("end_time"))
            if expected_end is not None and int(scheduled.get("end") or 0) > expected_end:
                violations.append(
                    ConstraintViolation(
                        code="FIXED_EVENT_END_EXCEEDED",
                        message=f"固定活动 {scheduled['poi_name']} 超过已知结束时间",
                        day_number=int(scheduled["day_number"]),
                        poi_id=poi_id,
                        details={"actual": scheduled.get("end"), "expected": expected_end},
                    )
                )

        budget = float(config.get("total_budget") or config.get("budget_range") or 0)
        budget_error_rate = max(0.0, (total_cost - budget) / budget) if budget else 0.0
        if budget and total_cost > budget:
            violations.append(
                ConstraintViolation(
                    code="TOTAL_BUDGET_EXCEEDED",
                    message=f"预计费用 ¥{total_cost:.0f} 超过预算 ¥{budget:.0f}",
                    details={"actual": total_cost, "limit": budget},
                )
            )

        preference_match = (
            matched_interests / interest_activities
            if interest_activities
            else (1.0 if not interests else 0.0)
        )
        max_transit_total = int(config.get("max_transit_minutes") or 0) * max(len(itinerary), 1)
        route_efficiency = (
            max(0.0, 1.0 - total_transit / max_transit_total) if max_transit_total else 1.0
        )
        mean_load = sum(daily_loads) / len(daily_loads) if daily_loads else 0.0
        fatigue_balance = (
            max(0.0, 1.0 - pstdev(daily_loads) / mean_load)
            if mean_load and len(daily_loads) > 1
            else 1.0
        )
        diversity = len(set(categories)) / len(categories) if categories else 0.0

        return ValidationReport(
            hard_pass=not violations,
            hard_violations=violations,
            soft_scores={
                "preference_match": round(preference_match, 4),
                "route_efficiency": round(route_efficiency, 4),
                "fatigue_balance": round(fatigue_balance, 4),
                "diversity": round(diversity, 4),
            },
            metrics={
                "budget_error_rate": round(budget_error_rate, 4),
                "total_cost": round(total_cost, 2),
                "total_transit_minutes": total_transit,
                "duplicate_poi_count": duplicate_count,
                "scheduled_days": len(itinerary),
                "activity_count": sum(len(day.get("activities", [])) for day in itinerary),
            },
        )
