"""Deterministic daily scheduler — extracted and hardened from _plan_with_algorithm.

Runs in < 500ms, no LLM, no external API calls.
"""

from typing import Optional

from schemas import ScoredPOI, WeatherDay, UserProfile, DayPlan, Activity, Location
from planner.core.models import Strategy


def build_schedule(
    strategy: Strategy,
    pois: list[ScoredPOI],
    weather: list[WeatherDay],
    profile: UserProfile,
) -> list[DayPlan]:
    """Build a full daily schedule from strategy + POIs + weather.

    Steps:
        1. Score POIs by preference match
        2. Group POIs by area
        3. Mark hard time constraints
        4. Assign POIs to days by area group
        5. Optimize daily routes (nearest neighbor)
        6. Build daily schedule with meal insertion
    """
    if not pois:
        return []

    travel_days = profile.travel_days or 1

    # Step 1: Score POIs
    scored = _score_pois(pois, profile)

    # Step 2: Group by area
    groups = _group_pois_by_area(scored)

    # Step 3: Mark time constraints
    constrained = _mark_time_constraints(scored)

    # Step 4: Assign to days
    day_assignments = _assign_days(constrained, groups, travel_days)

    # Step 5: Optimize routes per day
    optimized = _optimize_daily_routes(day_assignments)

    # Step 6: Build schedule
    schedule = _build_day_plans(optimized, weather, profile)

    return schedule


def _score_pois(pois: list[ScoredPOI], profile: UserProfile) -> list[ScoredPOI]:
    """Score POIs by preference match."""
    interests = set(profile.interests)
    food_prefs = set(profile.food_preferences)

    for poi in pois:
        base = poi.score
        interest_match = len(set(poi.tags) & interests) * 0.2
        food_match = len(set(poi.tags) & food_prefs) * 0.3
        pace_bonus = 0.1 if profile.pace != "intensive" else 0.0
        poi.score = min(base + interest_match + food_match + pace_bonus, 1.0)

    pois.sort(key=lambda p: p.score, reverse=True)
    return pois


def _group_pois_by_area(pois: list[ScoredPOI]) -> dict[str, list[ScoredPOI]]:
    """Group POIs by area/region for efficient daily planning."""
    groups: dict[str, list[ScoredPOI]] = {}
    for poi in pois:
        area = poi.area or "其他"
        groups.setdefault(area, []).append(poi)
    # Sort groups by size descending so largest areas get dedicated days first
    return dict(sorted(groups.items(), key=lambda x: len(x[1]), reverse=True))


def _mark_time_constraints(pois: list[ScoredPOI]) -> list[ScoredPOI]:
    """Mark hard time constraints."""
    for poi in pois:
        if poi.category == "attraction" and "夜景" in poi.tags:
            poi.time_constraint = "evening_only"
        elif poi.category == "restaurant" and "早茶" in poi.tags:
            poi.time_constraint = "morning_only"
        else:
            poi.time_constraint = "flexible"
    return pois


def _assign_days(
    pois: list[ScoredPOI],
    groups: dict[str, list[ScoredPOI]],
    travel_days: int,
) -> list[list[ScoredPOI]]:
    """Assign POIs to days by area group."""
    days: list[list[ScoredPOI]] = [[] for _ in range(travel_days)]
    assigned = set()

    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
    for day_idx, (_, group_pois) in enumerate(sorted_groups[:travel_days]):
        for poi in group_pois:
            if poi.name not in assigned:
                days[day_idx].append(poi)
                assigned.add(poi.name)

    remaining = [p for p in pois if p.name not in assigned]
    day_idx = 0
    for poi in remaining:
        placed = False
        for _ in range(travel_days):
            if len(days[day_idx]) < 5:
                days[day_idx].append(poi)
                assigned.add(poi.name)
                day_idx = (day_idx + 1) % travel_days
                placed = True
                break
            day_idx = (day_idx + 1) % travel_days

        if not placed:
            # The 5-POI cap is a balancing preference, not a hard constraint.
            # Once every day is full, place overflow on the shortest day rather
            # than spinning forever.
            target_idx = min(range(travel_days), key=lambda idx: len(days[idx]))
            days[target_idx].append(poi)
            assigned.add(poi.name)
            day_idx = (target_idx + 1) % travel_days

    return days


def _optimize_daily_routes(
    day_assignments: list[list[ScoredPOI]]
) -> list[list[ScoredPOI]]:
    """Optimize daily routes with nearest-neighbor ordering."""
    optimized = []
    for day_pois in day_assignments:
        if len(day_pois) <= 2:
            optimized.append(day_pois)
            continue
        ordered = _nearest_neighbor(day_pois)
        optimized.append(ordered)
    return optimized


def _nearest_neighbor(pois: list[ScoredPOI]) -> list[ScoredPOI]:
    """Greedy nearest neighbor ordering."""
    if not pois:
        return []

    unvisited = set(range(len(pois)))
    route = [0]
    unvisited.remove(0)

    while unvisited:
        last = route[-1]
        last_loc = pois[last].location
        nearest = min(
            unvisited,
            key=lambda i: _distance(last_loc, pois[i].location),
        )
        route.append(nearest)
        unvisited.remove(nearest)

    return [pois[i] for i in route]


def _distance(a: Optional[Location], b: Optional[Location]) -> float:
    if not a or not b:
        return float("inf")
    # Simple Euclidean approximation for sorting (sufficient for nearest-neighbor)
    return ((a.lat - b.lat) ** 2 + (a.lng - b.lng) ** 2) ** 0.5


def _build_day_plans(
    day_pois: list[list[ScoredPOI]],
    weather: list[WeatherDay],
    profile: UserProfile,
) -> list[DayPlan]:
    """Build daily schedule with meal insertion and time allocation."""
    schedule = []
    day_start_min = 9 * 60  # 09:00

    for day_idx, pois in enumerate(day_pois):
        day = DayPlan(day_number=day_idx + 1)

        if day_idx < len(weather):
            day.date = weather[day_idx].date

        current_time = day_start_min
        last_meal_time = -1000

        for poi in pois:
            # Lunch insertion window 11:30-13:30
            if 11 * 60 + 30 <= current_time <= 13 * 60 + 30:
                if current_time - last_meal_time >= 3.5 * 60:
                    meal = _create_meal_activity(day_idx, "lunch", profile)
                    day.activities.append(meal)
                    current_time += 90
                    last_meal_time = current_time

            # Dinner insertion window 17:30-19:30
            if 17 * 60 + 30 <= current_time <= 19 * 60 + 30:
                if current_time - last_meal_time >= 3.5 * 60:
                    meal = _create_meal_activity(day_idx, "dinner", profile)
                    day.activities.append(meal)
                    current_time += 90
                    last_meal_time = current_time

            duration = _resolve_duration(poi)
            activity = Activity(
                poi_name=poi.name,
                poi_id=poi.name,
                category=poi.category,
                start_time=_min_to_time(current_time),
                end_time=_min_to_time(current_time + duration),
                duration_min=duration,
                location=poi.location,
                recommendation_reason=poi.description or f"推荐游览{poi.name}",
                ticket_price=poi.ticket_price,
                time_constraint=poi.time_constraint,
                tags=poi.tags,
                open_time=poi.open_time,
                close_time=poi.close_time,
            )
            day.activities.append(activity)
            current_time += duration
            current_time += 30  # transit buffer

        day.total_cost = sum(
            (a.ticket_price or 0) + (a.meal_cost or 0) for a in day.activities
        )
        schedule.append(day)

    return schedule


def _resolve_duration(poi: ScoredPOI) -> int:
    """Resolve activity duration from POI recommended_hours or defaults."""
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


def _create_meal_activity(day_idx: int, meal_type: str, profile: UserProfile) -> Activity:
    """Create a meal activity placeholder."""
    food_hint = (
        f"（偏好：{','.join(profile.food_preferences)}）"
        if profile.food_preferences
        else ""
    )
    return Activity(
        poi_name=f"{meal_type.capitalize()}{food_hint}",
        category="restaurant",
        duration_min=90,
        meal_cost=80,
        recommendation_reason=f"在附近找一家{'辣' if '辣' in profile.food_preferences else '口碑好'}的餐厅",
    )


def _min_to_time(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"
