"""Deterministic Repair Executor — fix hard violations without LLM.

Each repair action only modifies start_time / end_time / day assignment /
activity presence.  poi_name, location, ticket_price, and duration_min are
treated as immutable facts.
"""

import logging

from schemas import Activity, DayPlan, ScoredPOI, UserProfile
from planner.core.models import RepairPlan, RepairResult, RuleViolation
from planner.core.rule_validator import validate

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def generate_repairs(
    violations: list[RuleViolation],
    itinerary: list[DayPlan],
    pois: list[ScoredPOI],
    profile: UserProfile,
    *,
    must_see: list[str] | None = None,
) -> list[RepairPlan]:
    """Generate deterministic repair plans for a list of hard violations."""
    plans: list[RepairPlan] = []
    pois_by_name = {p.name: p for p in pois}
    _must_see = must_see or []

    for v in violations:
        if v.rule == "time_feasibility" and v.day_number is not None:
            plans.extend(_repair_time_feasibility(v, itinerary, pois_by_name, profile, _must_see))
        elif v.rule == "transit_feasibility":
            plans.extend(_repair_transit_feasibility(v, itinerary))
        elif v.rule == "route_conflict":
            plans.extend(_repair_route_conflict(v, itinerary))
        elif v.rule == "must_see_presence":
            plans.extend(_repair_must_see(v, itinerary, pois_by_name))
        elif v.rule == "budget_compliance":
            plans.extend(_repair_budget(v, itinerary, pois_by_name, profile, _must_see))
        elif v.rule == "opening_hours":
            plans.extend(_repair_opening_hours(v, itinerary, pois_by_name))
        elif v.rule == "holiday_hotel_full":
            plans.extend(_repair_holiday_hotel(v, itinerary, profile))
        elif v.rule == "cross_city_feasibility":
            plans.extend(_repair_cross_city(v, itinerary, profile))

    return plans


def apply_repair(
    plan: RepairPlan,
    itinerary: list[DayPlan],
    pois: list[ScoredPOI],
    profile: UserProfile,
) -> list[DayPlan]:
    """Apply a single repair plan, returning a new itinerary list."""
    from copy import deepcopy

    result = deepcopy(itinerary)
    pois_by_name = {p.name: p for p in pois}

    if plan.action == "move":
        result = _apply_move(plan, result)
    elif plan.action == "insert":
        result = _apply_insert(plan, result, pois_by_name)
    elif plan.action == "remove":
        result = _apply_remove(plan, result)
    elif plan.action == "swap":
        result = _apply_swap(plan, result)

    # Rebuild times after structural change
    result = _rebuild_times(result, profile)
    return result


def run_repair_loop(
    itinerary: list[DayPlan],
    profile: UserProfile,
    must_see: list[str],
    pois: list[ScoredPOI],
    *,
    max_iterations: int = 10,
) -> RepairResult:
    """Validate → repair → validate loop until clean or stuck."""
    from copy import deepcopy

    current = deepcopy(itinerary)
    applied: list[RepairPlan] = []
    rejected: list[RepairPlan] = []

    for _ in range(max_iterations):
        report = validate(current, profile, must_see)
        if report.passed:
            return RepairResult(
                success=True,
                applied_plans=applied,
                rejected_plans=rejected,
                new_violations=[],
                needs_human=False,
                itinerary=current,
            )

        plans = generate_repairs(
            report.hard_violations,
            current,
            pois,
            profile,
            must_see=must_see,
        )
        if not plans:
            return RepairResult(
                success=False,
                applied_plans=applied,
                rejected_plans=rejected,
                new_violations=report.hard_violations,
                needs_human=True,
                itinerary=current,
            )

        # Apply the first applicable plan
        plan = plans[0]
        try:
            current = apply_repair(plan, current, pois, profile)
            applied.append(plan)
        except Exception as exc:
            logger.exception("Repair plan failed for %s: %s", plan.action, exc)
            rejected.append(plan)

    # Exhausted iterations
    final_report = validate(current, profile, must_see)
    return RepairResult(
        success=final_report.passed,
        applied_plans=applied,
        rejected_plans=rejected,
        new_violations=final_report.hard_violations,
        needs_human=not final_report.passed,
        itinerary=current,
    )


# --------------------------------------------------------------------------- #
# Time rebuilder
# --------------------------------------------------------------------------- #


def _rebuild_times(days: list[DayPlan], profile: UserProfile) -> list[DayPlan]:
    """Recalculate start/end times without changing activity order or facts."""
    day_start = 540  # 09:00
    lunch_start, lunch_end = 690, 810  # 11:30-13:30
    dinner_start, dinner_end = 1050, 1170  # 17:30-19:30

    for day in days:
        current = day_start
        last_meal = -1000
        new_activities: list[Activity] = []

        for act in day.activities:
            # Try to insert lunch before this activity
            if lunch_start <= current <= lunch_end and current - last_meal >= 210:
                meal = _build_meal("lunch", profile, current)
                new_activities.append(meal)
                current += meal.duration_min + 30
                last_meal = current

            # Try to insert dinner
            if dinner_start <= current <= dinner_end and current - last_meal >= 210:
                meal = _build_meal("dinner", profile, current)
                new_activities.append(meal)
                current += meal.duration_min + 30
                last_meal = current

            dur = act.duration_min
            new_act = act.model_copy(
                update={
                    "start_time": _min_to_time(current),
                    "end_time": _min_to_time(current + dur),
                }
            )
            new_activities.append(new_act)
            current += dur + 30  # transit buffer

        day.activities = new_activities
        day.total_cost = sum((a.ticket_price or 0) + (a.meal_cost or 0) for a in day.activities)

    return days


# --------------------------------------------------------------------------- #
# Repair generators
# --------------------------------------------------------------------------- #


def _repair_time_feasibility(
    v: RuleViolation,
    itinerary: list[DayPlan],
    pois_by_name: dict[str, ScoredPOI],
    profile: UserProfile,
    must_see: list[str],
) -> list[RepairPlan]:
    """Generate repair plans for time_feasibility violations."""
    plans: list[RepairPlan] = []
    day_idx = v.day_number - 1  # type: ignore[operator]
    if day_idx < 0 or day_idx >= len(itinerary):
        return plans
    day = itinerary[day_idx]

    act_idx = v.activity_index
    if act_idx is None or act_idx >= len(day.activities):
        return plans

    act_name = day.activities[act_idx].poi_name

    # --- Move: push the later activity to start after the earlier one ---
    if act_idx + 1 < len(day.activities):
        plans.append(
            RepairPlan(
                action="move",
                target={"day_number": v.day_number, "activity_index": act_idx + 1},
                params={"direction": "forward", "after_activity_index": act_idx},
                reason=f"将 {day.activities[act_idx + 1].poi_name} 后移以消除重叠",
            )
        )

    # --- Move: the violating activity to the next day ---
    if day_idx + 1 < len(itinerary):
        plans.append(
            RepairPlan(
                action="move",
                target={"day_number": v.day_number, "activity_index": act_idx},
                params={"to_day_number": v.day_number + 1},
                reason=f"将 {act_name} 移至第{v.day_number + 1}天",
            )
        )

    # --- Remove (last resort, never remove must-see or meals) ---
    if act_name not in must_see and not act_name.startswith(("Lunch", "Dinner")):
        plans.append(
            RepairPlan(
                action="remove",
                target={"day_number": v.day_number, "activity_index": act_idx},
                params={},
                reason=f"移除 {act_name}（无法调整时间）",
            )
        )

    return plans


def _repair_transit_feasibility(v: RuleViolation, itinerary: list[DayPlan]) -> list[RepairPlan]:
    """Generate split-day repair for insufficient transit gap."""
    plans: list[RepairPlan] = []
    day_idx = (v.day_number or 1) - 1
    act_idx = v.activity_index
    if act_idx is None or day_idx < 0 or day_idx >= len(itinerary):
        return plans
    day = itinerary[day_idx]
    if act_idx >= len(day.activities):
        return plans
    act_name = day.activities[act_idx].poi_name

    if day_idx + 1 < len(itinerary):
        plans.append(
            RepairPlan(
                action="move",
                target={"day_number": v.day_number, "activity_index": act_idx},
                params={"to_day_number": v.day_number + 1},
                reason=f"将 {act_name} 移至第{v.day_number + 1}天以留出交通时间",
            )
        )
    return plans


def _repair_route_conflict(v: RuleViolation, itinerary: list[DayPlan]) -> list[RepairPlan]:
    """Generate split-day repair for distant POIs on the same half-day."""
    plans: list[RepairPlan] = []
    day_idx = (v.day_number or 1) - 1
    act_idx = v.activity_index
    if act_idx is None or day_idx < 0 or day_idx >= len(itinerary):
        return plans
    day = itinerary[day_idx]
    if act_idx >= len(day.activities):
        return plans
    act_name = day.activities[act_idx].poi_name

    if day_idx + 1 < len(itinerary):
        plans.append(
            RepairPlan(
                action="move",
                target={"day_number": v.day_number, "activity_index": act_idx},
                params={"to_day_number": v.day_number + 1},
                reason=f"将 {act_name} 移至第{v.day_number + 1}天以避免远距离冲突",
            )
        )
    return plans


def _repair_opening_hours(
    v: RuleViolation,
    itinerary: list[DayPlan],
    pois_by_name: dict[str, ScoredPOI],
) -> list[RepairPlan]:
    """Generate reschedule/swap repair for activities outside opening hours."""
    plans: list[RepairPlan] = []
    day_idx = (v.day_number or 1) - 1
    act_idx = v.activity_index
    if act_idx is None or day_idx < 0 or day_idx >= len(itinerary):
        return plans
    day = itinerary[day_idx]
    if act_idx >= len(day.activities):
        return plans
    act = day.activities[act_idx]

    # Try moving the violating activity to the start of the same day.
    plans.append(
        RepairPlan(
            action="move",
            target={"day_number": v.day_number, "activity_index": act_idx},
            params={"to_day_number": v.day_number, "to_position": 0},
            reason=f"将 {act.poi_name} 调整至当天最早时段以匹配营业时间",
        )
    )

    # Try swapping with another activity in the same day
    for other_idx, other in enumerate(day.activities):
        if other_idx == act_idx:
            continue
        # Skip swapping with another activity that is also known to be closed at its slot
        if other.open_time and other.close_time:
            continue
        plans.append(
            RepairPlan(
                action="swap",
                target={"day_number": v.day_number, "activity_index": act_idx},
                params={"with_activity_index": other_idx},
                reason=f"将 {act.poi_name} 与 {other.poi_name} 交换顺序以匹配营业时间",
            )
        )
        break  # one swap candidate is enough

    return plans


def _repair_must_see(
    v: RuleViolation,
    itinerary: list[DayPlan],
    pois_by_name: dict[str, ScoredPOI],
) -> list[RepairPlan]:
    """Generate insert repair for missing must-see POI."""
    poi_name = v.poi_name
    if not poi_name or poi_name not in pois_by_name:
        return []

    # Pick the day with the fewest activities
    target_day = min(
        range(len(itinerary)),
        key=lambda i: len(itinerary[i].activities),
    )

    return [
        RepairPlan(
            action="insert",
            target={"day_number": target_day + 1},
            params={"poi_name": poi_name},
            reason=f"在行程中插入必须景点 {poi_name}",
        )
    ]


def _repair_budget(
    v: RuleViolation,
    itinerary: list[DayPlan],
    pois_by_name: dict[str, ScoredPOI],
    profile: UserProfile,
    must_see: list[str],
) -> list[RepairPlan]:
    """Generate remove repair for budget overrun.

    Removes lower-priority activities first while protecting must-see items:
    priority order (lowest first): P2 > P1 > P0; within same priority remove
    the most expensive first.  One plan per candidate so the repair loop can
    re-check.
    """
    priority_rank = {"P2": 3, "P1": 2, "P0": 1}

    candidates: list[tuple[int, float, int, int, str]] = []
    for di, day in enumerate(itinerary):
        for ai, act in enumerate(day.activities):
            if act.poi_name in must_see:
                continue
            if act.category == "restaurant":
                continue
            poi = pois_by_name.get(act.poi_name)
            if poi is None:
                continue
            priority = getattr(poi, "priority", "P2") or "P2"
            rank = priority_rank.get(priority, 99)
            cost = act.ticket_price or 0.0
            candidates.append((rank, cost, di, ai, act.poi_name))

    # Lower priority first, then higher cost first
    candidates.sort(key=lambda x: (-x[0], -x[1]))

    plans: list[RepairPlan] = []
    for rank, cost, di, ai, name in candidates:
        plans.append(
            RepairPlan(
                action="remove",
                target={"day_number": di + 1, "activity_index": ai},
                params={},
                reason=f"移除 {name}（¥{cost:.0f}）以满足预算",
            )
        )

    return plans


def _repair_holiday_hotel(
    v: RuleViolation,
    itinerary: list[DayPlan],
    profile: UserProfile,
) -> list[RepairPlan]:
    """Insert a placeholder hotel stay when lodging is missing on multi-day trips."""
    if not itinerary:
        return []
    target_day = min(range(len(itinerary)), key=lambda i: len(itinerary[i].activities))
    return [
        RepairPlan(
            action="insert",
            target={"day_number": target_day + 1},
            params={
                "poi_name": "市中心酒店",
                "category": "hotel",
                "duration_min": 60,
                "ticket_price": 0,
            },
            reason="节假日/旺季住宿紧张，插入市中心酒店占位",
        )
    ]


def _repair_cross_city(
    v: RuleViolation,
    itinerary: list[DayPlan],
    profile: UserProfile,
) -> list[RepairPlan]:
    """Insert inter-city transit when multi-city pacing is too tight."""
    day_idx = (v.day_number or 1) - 1
    if day_idx < 0 or day_idx >= len(itinerary):
        return []
    return [
        RepairPlan(
            action="insert",
            target={"day_number": day_idx + 1},
            params={
                "poi_name": "城际交通",
                "category": "transit",
                "duration_min": 120,
                "ticket_price": 0,
            },
            reason="跨城行程需预留城际交通时间",
        )
    ]


# --------------------------------------------------------------------------- #
# Repair appliers
# --------------------------------------------------------------------------- #


def _apply_move(plan: RepairPlan, itinerary: list[DayPlan]) -> list[DayPlan]:
    """Move an activity within or across days."""
    tgt = plan.target
    day_idx = tgt["day_number"] - 1
    act_idx = tgt["activity_index"]
    act = itinerary[day_idx].activities.pop(act_idx)

    if "to_day_number" in plan.params:
        to_day = plan.params["to_day_number"] - 1
        if to_day >= len(itinerary):
            raise ValueError(f"Day {to_day + 1} does not exist")
        to_pos = plan.params.get("to_position")
        if to_pos is None:
            itinerary[to_day].activities.append(act)
        else:
            itinerary[to_day].activities.insert(to_pos, act)
    else:
        # Move within the same day: push to end
        itinerary[day_idx].activities.append(act)

    return itinerary


def _apply_insert(
    plan: RepairPlan,
    itinerary: list[DayPlan],
    pois_by_name: dict[str, ScoredPOI],
) -> list[DayPlan]:
    """Insert a must-see POI or synthetic placeholder at the start of the target day."""
    day_idx = plan.target["day_number"] - 1
    poi_name = plan.params.get("poi_name")
    if not poi_name:
        return itinerary

    if poi_name in pois_by_name:
        poi = pois_by_name[poi_name]
        dur = _resolve_duration(poi)
        activity = Activity(
            poi_name=poi.name,
            category=poi.category,
            duration_min=dur,
            ticket_price=poi.ticket_price,
            location=poi.location,
            recommendation_reason=poi.description or "",
            tags=poi.tags,
            open_time=poi.open_time,
            close_time=poi.close_time,
            time_constraint=getattr(poi, "time_constraint", "flexible"),
        )
    else:
        dur = int(plan.params.get("duration_min", 60))
        activity = Activity(
            poi_name=poi_name,
            category=plan.params.get("category", "attraction"),
            duration_min=dur,
            ticket_price=plan.params.get("ticket_price", 0),
            recommendation_reason=plan.reason or "自动修复插入",
        )

    itinerary[day_idx].activities.insert(0, activity)
    return itinerary


def _apply_remove(plan: RepairPlan, itinerary: list[DayPlan]) -> list[DayPlan]:
    """Remove an activity from the itinerary."""
    tgt = plan.target
    day_idx = tgt["day_number"] - 1
    act_idx = tgt["activity_index"]
    itinerary[day_idx].activities.pop(act_idx)
    return itinerary


def _apply_swap(plan: RepairPlan, itinerary: list[DayPlan]) -> list[DayPlan]:
    """Swap two activities."""
    tgt = plan.target
    day_idx = tgt["day_number"] - 1
    i, j = tgt["activity_index"], plan.params["with_activity_index"]
    acts = itinerary[day_idx].activities
    acts[i], acts[j] = acts[j], acts[i]
    return itinerary


# --------------------------------------------------------------------------- #
# Helpers (shared with daily_scheduler)
# --------------------------------------------------------------------------- #


def _resolve_duration(poi: ScoredPOI) -> int:
    hours_map = {
        "1小时": 60,
        "1-2小时": 90,
        "2小时": 120,
        "2-3小时": 150,
        "半天": 240,
        "全天": 360,
    }
    if poi.recommended_hours and poi.recommended_hours in hours_map:
        return hours_map[poi.recommended_hours]
    if poi.category == "restaurant":
        return 90
    return 120


def _min_to_time(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _build_meal(meal_type: str, profile: UserProfile, start_min: int) -> Activity:
    food_hint = (
        f"（偏好：{','.join(profile.food_preferences)}）" if profile.food_preferences else ""
    )
    return Activity(
        poi_name=f"{meal_type.capitalize()}{food_hint}",
        category="restaurant",
        start_time=_min_to_time(start_min),
        end_time=_min_to_time(start_min + 90),
        duration_min=90,
        meal_cost=80,
        recommendation_reason="就近推荐",
    )
