"""Rule Validator — deterministic hard/soft constraint validation.

Hard violations block itinerary_final; soft warnings are metadata only.
"""

from datetime import time
from typing import Optional

from schemas import DayPlan, Activity, Location, UserProfile
from planner.core.models import RuleViolation, ValidationReport


def validate(
    itinerary: list[DayPlan],
    profile: UserProfile,
    must_see: list[str],
) -> ValidationReport:
    """Run all validation checks and return a classified report."""
    hard_violations: list[RuleViolation] = []
    soft_warnings: list[RuleViolation] = []

    # Hard checks
    hard_violations.extend(_check_time_feasibility(itinerary))
    hard_violations.extend(_check_transit_feasibility(itinerary))
    hard_violations.extend(_check_route_conflict(itinerary))
    hard_violations.extend(_check_must_see_presence(itinerary, must_see))
    hard_violations.extend(_check_budget_compliance(itinerary, profile))
    hard_violations.extend(_check_opening_hours(itinerary))
    hard_violations.extend(_check_holiday_hotel_full(itinerary, profile))
    hard_violations.extend(_check_cross_city_feasibility(itinerary, profile))

    # Soft checks
    soft_warnings.extend(_check_distance_sanity(itinerary))
    soft_warnings.extend(_check_preference_coverage(itinerary, profile))

    return ValidationReport(
        hard_violations=hard_violations,
        soft_warnings=soft_warnings,
        passed=len(hard_violations) == 0,
    )


# --------------------------------------------------------------------------- #
# Hard Violation Checks
# --------------------------------------------------------------------------- #


def _check_time_feasibility(itinerary: list[DayPlan]) -> list[RuleViolation]:
    """Check for overlapping activities and activities outside day bounds."""
    violations = []
    day_start = time(9, 0)
    day_end = time(21, 0)

    for day in itinerary:
        activities = day.activities
        for i, activity in enumerate(activities):
            start, end = _parse_time_range(activity.start_time, activity.end_time)
            if start is None or end is None:
                if activity.start_time or activity.end_time:
                    violations.append(
                        RuleViolation(
                            rule="time_feasibility",
                            severity="hard",
                            message=f"{activity.poi_name} 时间格式无效",
                            day_number=day.day_number,
                            activity_index=i,
                            poi_name=activity.poi_name,
                        )
                    )
                continue

            # Outside day bounds
            if start < day_start or end > day_end:
                violations.append(
                    RuleViolation(
                        rule="time_feasibility",
                        severity="hard",
                        message=f"{activity.poi_name} 时间超出09:00-21:00范围",
                        day_number=day.day_number,
                        activity_index=i,
                        poi_name=activity.poi_name,
                    )
                )

            # Overlapping with next activity
            if i + 1 < len(activities):
                next_act = activities[i + 1]
                next_start, _ = _parse_time_range(next_act.start_time, next_act.end_time)
                if next_start and next_start < end:
                    violations.append(
                        RuleViolation(
                            rule="time_feasibility",
                            severity="hard",
                            message=f"{activity.poi_name} 与 {next_act.poi_name} 时间重叠",
                            day_number=day.day_number,
                            activity_index=i,
                            poi_name=activity.poi_name,
                            suggested_fix={
                                "action": "move",
                                "target": {
                                    "day_number": day.day_number,
                                    "activity_index": i + 1,
                                },
                                "params": {"direction": "forward"},
                                "reason": f"将 {next_act.poi_name} 后移以消除重叠",
                            },
                        )
                    )

    return violations


def _check_must_see_presence(itinerary: list[DayPlan], must_see: list[str]) -> list[RuleViolation]:
    """Check that all must-see POIs appear in the itinerary."""
    if not must_see:
        return []

    present = {a.poi_name for day in itinerary for a in day.activities}
    violations = []
    for poi_name in must_see:
        if poi_name not in present:
            violations.append(
                RuleViolation(
                    rule="must_see_presence",
                    severity="hard",
                    message=f"必须景点 {poi_name} 未在行程中安排",
                    poi_name=poi_name,
                    suggested_fix={
                        "action": "insert",
                        "target": {"poi_name": poi_name},
                        "reason": f"在行程中插入必须景点 {poi_name}",
                    },
                )
            )
    return violations


def _check_transit_feasibility(itinerary: list[DayPlan]) -> list[RuleViolation]:
    """Check whether gaps between located activities can cover transit time."""
    violations = []
    for day in itinerary:
        previous: tuple[int, Activity, time] | None = None
        for i, activity in enumerate(day.activities):
            if not activity.location:
                continue

            start, end = _parse_time_range(activity.start_time, activity.end_time)
            if start is None or end is None:
                continue

            if previous:
                prev_idx, prev_activity, prev_end = previous
                gap_min = _minutes(start) - _minutes(prev_end)
                required_min = _estimate_transit_minutes(
                    prev_activity.location,
                    activity.location,
                )
                if gap_min < required_min:
                    violations.append(
                        RuleViolation(
                            rule="transit_feasibility",
                            severity="hard",
                            message=(
                                f"{prev_activity.poi_name} 到 {activity.poi_name} "
                                f"预留交通 {gap_min} 分钟不足，预计至少 {required_min} 分钟"
                            ),
                            day_number=day.day_number,
                            activity_index=prev_idx,
                            poi_name=prev_activity.poi_name,
                            suggested_fix={
                                "action": "split_day",
                                "target": {
                                    "day_number": day.day_number,
                                    "activity_index": i,
                                },
                                "params": {"to_day_number": day.day_number + 1},
                                "reason": f"将 {activity.poi_name} 拆分至其他天以留出足够交通时间",
                            },
                        )
                    )

            previous = (i, activity, end)

    return violations


def _check_route_conflict(itinerary: list[DayPlan]) -> list[RuleViolation]:
    """Hard violation if two activities on the same half-day are >30km apart."""
    violations = []
    for day in itinerary:
        acts = day.activities
        for i in range(len(acts)):
            for j in range(i + 1, len(acts)):
                a, b = acts[i], acts[j]
                if not a.location or not b.location:
                    continue
                if not _same_half_day(a.start_time, b.start_time):
                    continue
                dist = _distance_km(a.location, b.location)
                if dist > 30:
                    violations.append(
                        RuleViolation(
                            rule="route_conflict",
                            severity="hard",
                            message=f"{a.poi_name} 与 {b.poi_name} 同半天直线距离 {dist:.1f}km 超过 30km",
                            day_number=day.day_number,
                            activity_index=i,
                            poi_name=a.poi_name,
                            suggested_fix={
                                "action": "split_day",
                                "target": {
                                    "day_number": day.day_number,
                                    "activity_index": j,
                                },
                                "params": {"to_day_number": day.day_number + 1},
                                "reason": f"将 {b.poi_name} 拆分至其他天以避免远距离奔波",
                            },
                        )
                    )
    return violations


def _same_half_day(start_a: Optional[str], start_b: Optional[str]) -> bool:
    """Return True if both activities start in the same half-day."""
    ta = _parse_time(start_a)
    tb = _parse_time(start_b)
    if ta is None or tb is None:
        return False
    # Morning: before 12:00, afternoon: 12:00-17:59, evening: after 18:00
    bucket_a = "morning" if ta.hour < 12 else "afternoon" if ta.hour < 18 else "evening"
    bucket_b = "morning" if tb.hour < 12 else "afternoon" if tb.hour < 18 else "evening"
    return bucket_a == bucket_b


def _check_budget_compliance(itinerary: list[DayPlan], profile: UserProfile) -> list[RuleViolation]:
    """Check total cost against budget. Hard if > 1.2x budget."""
    if profile.budget_range is None or profile.budget_range <= 0:
        return []

    total_cost = sum(day.total_cost for day in itinerary)
    if total_cost > profile.budget_range * 1.2:
        return [
            RuleViolation(
                rule="budget_compliance",
                severity="hard",
                message=f"总费用 {total_cost:.0f} 超出预算 {profile.budget_range:.0f} 的20%",
                suggested_fix={
                    "action": "remove",
                    "reason": "按优先级删除非必须高消费项目以满足预算",
                },
            )
        ]
    return []


# --------------------------------------------------------------------------- #
# Soft Warning Checks
# --------------------------------------------------------------------------- #


def _check_opening_hours(itinerary: list[DayPlan]) -> list[RuleViolation]:
    """Hard check if activity time is outside known opening hours."""
    violations = []
    for day in itinerary:
        for i, activity in enumerate(day.activities):
            if not activity.open_time or not activity.close_time:
                continue

            act_start, act_end = _parse_time_range(activity.start_time, activity.end_time)
            open_t = _parse_time(activity.open_time)
            close_t = _parse_time(activity.close_time)

            if act_start is None or act_end is None or open_t is None or close_t is None:
                continue

            if act_start < open_t or act_end > close_t:
                violations.append(
                    RuleViolation(
                        rule="opening_hours",
                        severity="hard",
                        message=(
                            f"{activity.poi_name} 安排时间 "
                            f"({activity.start_time}-{activity.end_time}) "
                            f"不在营业时间 ({activity.open_time}-{activity.close_time}) 内"
                        ),
                        day_number=day.day_number,
                        activity_index=i,
                        poi_name=activity.poi_name,
                        suggested_fix={
                            "action": "reschedule",
                            "target": {
                                "day_number": day.day_number,
                                "activity_index": i,
                            },
                            "params": {
                                "open_time": activity.open_time,
                                "close_time": activity.close_time,
                            },
                            "reason": (
                                f"将 {activity.poi_name} 调整到营业时间 "
                                f"{activity.open_time}-{activity.close_time} 内"
                            ),
                        },
                    )
                )
    return violations


def _check_holiday_hotel_full(
    itinerary: list[DayPlan],
    profile: UserProfile,
) -> list[RuleViolation]:
    """Detect missing or overpriced lodging on multi-day trips."""
    if not profile.travel_days or profile.travel_days <= 1:
        return []
    if len(itinerary) < 2:
        # Partial single-day drafts during repair should not trigger lodging rules yet.
        return []

    has_hotel = any(
        act.category == "hotel" or "酒店" in act.poi_name
        for day in itinerary
        for act in day.activities
    )
    if has_hotel:
        return []

    needs_lodging = profile.accommodation_preference or profile.travel_days > 1
    if not needs_lodging:
        return []

    return [
        RuleViolation(
            rule="holiday_hotel_full",
            severity="hard",
            message="多日行程缺少酒店安排，可能存在节假日满房风险",
            suggested_fix={
                "action": "insert",
                "target": {"day_number": 1},
                "params": {"category": "hotel"},
                "reason": "插入市中心酒店占位",
            },
        )
    ]


def _extract_cities(profile: UserProfile) -> list[str]:
    dest = (profile.destination or "").strip()
    if not dest:
        return []
    normalized = dest
    for sep in ["+", "、", "/", "转", "和", "至"]:
        normalized = normalized.replace(sep, ",")
    return [c.strip() for c in normalized.split(",") if c.strip()]


def _check_cross_city_feasibility(
    itinerary: list[DayPlan],
    profile: UserProfile,
) -> list[RuleViolation]:
    """Ensure multi-city trips have enough days and transit capacity."""
    cities = _extract_cities(profile)
    if len(cities) <= 1:
        return []

    violations: list[RuleViolation] = []
    if len(itinerary) < len(cities):
        violations.append(
            RuleViolation(
                rule="cross_city_feasibility",
                severity="hard",
                message=f"跨城行程 {len(cities)} 个城市但仅安排 {len(itinerary)} 天",
                day_number=1,
                suggested_fix={
                    "action": "insert",
                    "target": {"day_number": 1},
                    "params": {"category": "transit"},
                    "reason": "增加城际交通节点",
                },
            )
        )

    for day in itinerary:
        transit_count = sum(
            1 for act in day.activities if act.category == "transit" or "交通" in act.poi_name
        )
        if len(cities) > 1 and transit_count == 0 and day.day_number < len(itinerary):
            violations.append(
                RuleViolation(
                    rule="cross_city_feasibility",
                    severity="hard",
                    message=f"第{day.day_number}天跨城行程缺少城际交通安排",
                    day_number=day.day_number,
                    suggested_fix={
                        "action": "insert",
                        "target": {"day_number": day.day_number},
                        "params": {"category": "transit"},
                        "reason": "插入城际交通节点",
                    },
                )
            )
    return violations


def _check_distance_sanity(itinerary: list[DayPlan]) -> list[RuleViolation]:
    """Warn if daily activity locations span unrealistic distances."""
    warnings = []
    # Simplified: warn if a day has activities in > 3 distinct areas
    for day in itinerary:
        areas = set()
        for activity in day.activities:
            if activity.location and hasattr(activity.location, "area"):
                areas.add(activity.location.area)
        if len(areas) > 3:
            warnings.append(
                RuleViolation(
                    rule="distance_sanity",
                    severity="soft",
                    message=f"第{day.day_number}天涉及{len(areas)}个不同区域，移动距离可能较大",
                    day_number=day.day_number,
                )
            )
    return warnings


def _check_preference_coverage(
    itinerary: list[DayPlan], profile: UserProfile
) -> list[RuleViolation]:
    """Warn if user interests are poorly covered."""
    if not profile.interests:
        return []

    matched = set()
    total_activities = 0
    for day in itinerary:
        for activity in day.activities:
            total_activities += 1
            if activity.tags:
                matched.update(set(activity.tags) & set(profile.interests))

    if total_activities == 0:
        return []

    coverage = len(matched) / len(profile.interests)
    if coverage < 0.5:
        return [
            RuleViolation(
                rule="preference_coverage",
                severity="soft",
                message=f"兴趣匹配度仅{coverage:.0%}，建议增加与{profile.interests}相关的景点",
            )
        ]
    return []


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _parse_time_range(
    start_str: Optional[str], end_str: Optional[str]
) -> tuple[Optional[time], Optional[time]]:
    start = _parse_time(start_str)
    end = _parse_time(end_str)
    return start, end


def _parse_time(t: Optional[str]) -> Optional[time]:
    if not t:
        return None
    try:
        parts = t.split(":")
        return time(int(parts[0]), int(parts[1]))
    except (ValueError, IndexError):
        return None


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def _estimate_transit_minutes(origin: Optional[Location], destination: Optional[Location]) -> int:
    if not origin or not destination:
        return 0

    distance_km = _distance_km(origin, destination)
    if distance_km < 1:
        return 10
    if distance_km < 8:
        return int(distance_km / 18 * 60) + 10
    if distance_km < 35:
        return int(distance_km / 24 * 60) + 15
    return int(distance_km / 35 * 60) + 25


def _distance_km(a: Location, b: Location) -> float:
    lat_km = (a.lat - b.lat) * 111
    lng_km = (a.lng - b.lng) * 85
    return (lat_km**2 + lng_km**2) ** 0.5
