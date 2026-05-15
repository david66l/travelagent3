"""Rule Validator — deterministic hard/soft constraint validation.

Hard violations block itinerary_final; soft warnings are metadata only.
"""

from datetime import time
from typing import Optional

from schemas import DayPlan, Activity, UserProfile
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
    hard_violations.extend(_check_must_see_presence(itinerary, must_see))
    hard_violations.extend(_check_budget_compliance(itinerary, profile))

    # Soft checks
    soft_warnings.extend(_check_opening_hours(itinerary))
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
                        )
                    )

    return violations


def _check_must_see_presence(
    itinerary: list[DayPlan], must_see: list[str]
) -> list[RuleViolation]:
    """Check that all must-see POIs appear in the itinerary."""
    if not must_see:
        return []

    present = {
        a.poi_name
        for day in itinerary
        for a in day.activities
    }
    violations = []
    for poi_name in must_see:
        if poi_name not in present:
            violations.append(
                RuleViolation(
                    rule="must_see_presence",
                    severity="hard",
                    message=f"必须景点 {poi_name} 未在行程中安排",
                    poi_name=poi_name,
                )
            )
    return violations


def _check_budget_compliance(
    itinerary: list[DayPlan], profile: UserProfile
) -> list[RuleViolation]:
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
            )
        ]
    return []


# --------------------------------------------------------------------------- #
# Soft Warning Checks
# --------------------------------------------------------------------------- #


def _check_opening_hours(itinerary: list[DayPlan]) -> list[RuleViolation]:
    """Warn if activity time is outside known opening hours.

    If open_time/close_time is missing, skip check (no warning).
    """
    warnings = []
    for day in itinerary:
        for i, activity in enumerate(day.activities):
            if not activity.open_time or not activity.close_time:
                continue  # Missing data → no warning

            act_start, act_end = _parse_time_range(
                activity.start_time, activity.end_time
            )
            open_t = _parse_time(activity.open_time)
            close_t = _parse_time(activity.close_time)

            if act_start is None or act_end is None or open_t is None or close_t is None:
                continue

            if act_start < open_t or act_end > close_t:
                warnings.append(
                    RuleViolation(
                        rule="opening_hours",
                        severity="soft",
                        message=f"{activity.poi_name} 安排时间 ({activity.start_time}-{activity.end_time}) 不在营业时间 ({activity.open_time}-{activity.close_time}) 内",
                        day_number=day.day_number,
                        activity_index=i,
                        poi_name=activity.poi_name,
                    )
                )
    return warnings


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
