"""Deterministic Repair Executor — fix hard violations without LLM.

Each repair action only modifies start_time / end_time / day assignment /
activity presence.  poi_name, location, ticket_price, and duration_min are
treated as immutable facts.
"""

from typing import Optional

from schemas import Activity, DayPlan, ScoredPOI, UserProfile
from planner.core.models import RepairPlan, RepairResult, RuleViolation
from planner.core.rule_validator import validate

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
            plans.extend(
                _repair_time_feasibility(v, itinerary, pois_by_name, profile, _must_see)
            )
        elif v.rule == "must_see_presence":
            plans.extend(_repair_must_see(v, itinerary, pois_by_name))
        elif v.rule == "budget_compliance":
            plans.extend(_repair_budget(v, itinerary, pois_by_name, profile, _must_see))

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
            )

        plans = generate_repairs(
            report.hard_violations, current, pois, profile, must_see=must_see,
        )
        if not plans:
            return RepairResult(
                success=False,
                applied_plans=applied,
                rejected_plans=rejected,
                new_violations=report.hard_violations,
                needs_human=True,
            )

        # Apply the first applicable plan
        plan = plans[0]
        try:
            current = apply_repair(plan, current, pois, profile)
            applied.append(plan)
        except Exception:
            rejected.append(plan)

    # Exhausted iterations
    final_report = validate(current, profile, must_see)
    return RepairResult(
        success=final_report.passed,
        applied_plans=applied,
        rejected_plans=rejected,
        new_violations=final_report.hard_violations,
        needs_human=not final_report.passed,
    )


# --------------------------------------------------------------------------- #
# Time rebuilder
# --------------------------------------------------------------------------- #


def _rebuild_times(
    days: list[DayPlan], profile: UserProfile
) -> list[DayPlan]:
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
        day.total_cost = sum(
            (a.ticket_price or 0) + (a.meal_cost or 0) for a in day.activities
        )

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

    Targets the activity with highest ticket_price among non-must-see
    activities — removes one at a time so the repair loop can re-check.
    """
    # Collect all removable activities (attractions with ticket_price, not must-see)
    candidates: list[tuple[float, int, int, str]] = []
    for di, day in enumerate(itinerary):
        for ai, act in enumerate(day.activities):
            if act.poi_name in must_see:
                continue
            if act.category != "restaurant" and (act.ticket_price or 0) > 0:
                candidates.append((act.ticket_price or 0, di, ai, act.poi_name))

    # Sort by cost descending (remove most expensive first)
    candidates.sort(key=lambda x: x[0], reverse=True)

    plans: list[RepairPlan] = []
    for cost, di, ai, name in candidates:
        plans.append(
            RepairPlan(
                action="remove",
                target={"day_number": di + 1, "activity_index": ai},
                params={},
                reason=f"移除 {name}（¥{cost:.0f}）以满足预算",
            )
        )

    return plans


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
        # Insert at end of target day
        itinerary[to_day].activities.append(act)
    else:
        # Move within the same day: push to end
        itinerary[day_idx].activities.append(act)

    return itinerary


def _apply_insert(
    plan: RepairPlan,
    itinerary: list[DayPlan],
    pois_by_name: dict[str, ScoredPOI],
) -> list[DayPlan]:
    """Insert a must-see POI at the start of the target day.

    Inserting at position 0 gives the activity the earliest possible time
    slot after _rebuild_times runs — avoids pushing it past 21:00.
    """
    poi_name = plan.params["poi_name"]
    day_idx = plan.target["day_number"] - 1
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
        f"（偏好：{','.join(profile.food_preferences)}）"
        if profile.food_preferences
        else ""
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
