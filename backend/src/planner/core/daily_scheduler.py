"""Deterministic daily scheduler — extracted and hardened from _plan_with_algorithm.

Runs in < 500ms, no LLM, no external API calls.
"""

import logging
from typing import Optional

from schemas import ScoredPOI, WeatherDay, UserProfile, DayPlan, Activity, Location
from planner.core.models import Strategy


logger = logging.getLogger(__name__)

DAY_START = 9 * 60  # 09:00
DAY_END = 21 * 60  # 21:00
DAY_AVAILABLE_MINUTES = DAY_END - DAY_START  # 12 hours
MEAL_DURATION = 90  # minutes


def build_schedule(
    strategy: Strategy,
    pois: list[ScoredPOI],
    weather: list[WeatherDay],
    profile: UserProfile,
) -> list[DayPlan]:
    """Build a full daily schedule from strategy + POIs + weather.

    Steps:
        1. Score POIs by preference match
        2. Group POIs by proximity (fallback to area)
        3. Detect remote POIs and classify excursions
        4. Assign POIs to days with embedded budget/time/diversity constraints
        5. Optimize daily routes (multi-start NN + 2-opt)
        6. Assign time slots with real transit times and opening hours
        7. Insert real restaurant meals
        8. Append transport & accommodation budget
        9. Run lightweight sanity check (no repair)
    """
    if not pois:
        return []

    travel_days = profile.travel_days or 1

    # Step 1: Score POIs
    scored = _score_pois(pois, profile)

    # Step 2: Group by proximity, fallback to area
    groups = _group_pois_by_proximity(scored) or _group_pois_by_area(scored)

    # Step 3: Detect remote POIs
    remote_class, cross_city, center = detect_remote_pois(scored)

    # Step 4: Assign to days with constraints
    day_assignments = _assign_days_constrained(
        scored,
        groups,
        travel_days,
        profile,
        strategy.must_see,
        remote_class,
        cross_city,
        center,
    )

    # Step 5: Optimize routes per day
    optimized = _optimize_daily_routes(day_assignments)

    # Step 6-7: Build schedule with real restaurants
    all_restaurants = [p for p in scored if p.category == "restaurant"]
    nearby_by_day = []
    for day in optimized:
        day_names = {p.name for p in day}
        nearby_by_day.append([r for r in all_restaurants if r.name not in day_names])

    schedule: list[DayPlan] = []
    for day_idx, day_pois in enumerate(optimized):
        day = DayPlan(day_number=day_idx + 1)
        if day_idx < len(weather):
            day.date = weather[day_idx].date

        nearby = (
            nearby_by_day[day_idx]
            if day_idx < len(nearby_by_day)
            else []
        )
        day.activities = _assign_time_slots(
            day_pois, day_idx, weather, profile, nearby
        )
        day.total_cost = sum(
            (a.ticket_price or 0) + (a.meal_cost or 0) for a in day.activities
        )
        schedule.append(day)

    # Step 8: Append transport & accommodation budget
    _append_travel_budget(schedule, profile)

    # Step 9: Sanity check (no repair)
    for warning in _sanity_check(schedule, profile):
        logger.warning(warning)

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


def _group_pois_by_proximity(pois: list[ScoredPOI]) -> dict[str, list[ScoredPOI]]:
    """Group POIs by geographic proximity (≤3km from cluster seed).

    Returns empty dict when not enough POIs have location data, so callers
    can fall back to area grouping.
    """
    located = [p for p in pois if p.location]
    if len(located) < 2:
        return {}

    ungrouped = set(range(len(located)))
    groups: dict[str, list[ScoredPOI]] = {}

    while ungrouped:
        seed_idx = min(ungrouped)
        seed = located[seed_idx]
        cluster = [seed]
        ungrouped.remove(seed_idx)

        close = [
            i
            for i in ungrouped
            if _distance_km(seed.location, located[i].location) <= 3.0  # type: ignore[arg-type]
        ]
        for i in close:
            cluster.append(located[i])
            ungrouped.remove(i)

        groups[seed.name] = cluster

    return groups


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


def detect_remote_pois(
    pois: list[ScoredPOI],
) -> tuple[dict[str, str], set[str], Optional[Location]]:
    """Detect and classify remote POIs by distance from the median center.

    Returns:
        - remote_class: poi_name -> "half_day" / "full_day" / "cross_city"
        - cross_city: set of poi_names that should be excluded entirely
        - center: median center location or None
    """
    remote_class: dict[str, str] = {}
    cross_city: set[str] = set()
    center: Optional[Location] = None

    located = [p for p in pois if p.location]
    if len(located) < 3:
        return remote_class, cross_city, center

    # Median center (resistant to extreme outliers dragging the mean)
    lats = sorted(p.location.lat for p in located if p.location)  # type: ignore[union-attr]
    lngs = sorted(p.location.lng for p in located if p.location)  # type: ignore[union-attr]
    center = Location(
        lat=lats[len(lats) // 2],
        lng=lngs[len(lngs) // 2],
    )
    distances = [_distance_km(center, p.location) for p in located if p.location]
    median_distance = sorted(distances)[(len(distances) - 1) // 2]
    threshold = max(30.0, median_distance * 2.0)
    remote_names_all = {
        p.name for p in located if p.location and _distance_km(center, p.location) >= threshold
    }

    for p in located:
        if p.name not in remote_names_all:
            continue
        dist = _distance_km(center, p.location)  # type: ignore[arg-type]
        one_way_h = dist / 50.0
        activity_h = _resolve_duration(p) / 60.0

        if dist > 100.0 or one_way_h * 2 > 5:  # >100km or round-trip > 5h → cross_city
            remote_class[p.name] = "cross_city"
            cross_city.add(p.name)
        elif one_way_h * 2 + activity_h > 9:  # round-trip + 游玩 > 9h → full_day
            remote_class[p.name] = "full_day"
        elif _resolve_duration(p) >= 180:  # ≥ 3h → full_day
            remote_class[p.name] = "full_day"
        else:
            remote_class[p.name] = "half_day"

    # Upgrade to full_day when same-area total ≥ 300 min
    for p in located:
        if remote_class.get(p.name) not in ("full_day", "cross_city", None):
            same_area = [pp for pp in pois if pp.area == p.area and pp.name != p.name]
            total = _resolve_duration(p) + sum(_resolve_duration(pp) for pp in same_area)
            if total >= 300:
                remote_class[p.name] = "full_day"

    return remote_class, cross_city, center


def _assign_days_constrained(
    pois: list[ScoredPOI],
    groups: dict[str, list[ScoredPOI]],
    travel_days: int,
    profile: UserProfile,
    must_see: list[str],
    remote_class: dict[str, str],
    cross_city: set[str],
    center: Optional[Location],
) -> list[list[ScoredPOI]]:
    """Assign POIs to days with embedded budget/time/must-see/diversity constraints.

    Each POI is only placed when `_can_assign` confirms the day still has enough
    budget, time, and category diversity. Must-see POIs are forced in if no day
    satisfies the constraints.
    """
    days: list[list[ScoredPOI]] = [[] for _ in range(travel_days)]

    # Distribute total budget across days, reserving transport & accommodation
    total_budget = profile.budget_range or float("inf")
    if total_budget != float("inf"):
        transport_total = travel_days * 30
        accommodation_total = max(0, travel_days - 1) * 200
        daily_poi_budget = max(
            0.0, (total_budget - transport_total - accommodation_total) / travel_days
        )
    else:
        daily_poi_budget = float("inf")

    day_budget = [daily_poi_budget] * travel_days
    day_minutes = [DAY_AVAILABLE_MINUTES] * travel_days
    day_categories = [set() for _ in range(travel_days)]
    assigned: set[str] = set()

    # Cross-city POIs are excluded from scheduling
    for name in cross_city:
        assigned.add(name)

    # 1. Must-see first — force them in, but respect constraints when possible
    must_see_pois = [p for p in pois if p.name in must_see and p.name not in assigned]
    for poi in must_see_pois:
        best_day = None
        best_remaining = -1
        for day_idx in range(travel_days):
            if _can_assign(
                poi,
                days[day_idx],
                day_budget[day_idx],
                day_minutes[day_idx],
                day_categories[day_idx],
            ):
                if day_minutes[day_idx] > best_remaining:
                    best_remaining = day_minutes[day_idx]
                    best_day = day_idx
        if best_day is None:
            # Force onto the day with the most remaining time
            best_day = max(range(travel_days), key=lambda i: day_minutes[i])

        days[best_day].append(poi)
        assigned.add(poi.name)
        day_budget[best_day] -= _resolve_poi_cost(poi)
        day_minutes[best_day] -= _resolve_duration(poi) + _estimate_transit(poi, center)
        day_categories[best_day].add(poi.category)

    # 2. Remote POIs to empty days (preserve dedicated excursion days)
    active_remote = {n: c for n, c in remote_class.items() if c != "cross_city"}
    remote_areas: dict[str, list[ScoredPOI]] = {}
    for p in pois:
        if p.name in active_remote and p.name not in assigned:
            area = p.area or p.name
            remote_areas.setdefault(area, []).append(p)

    for _, group in remote_areas.items():
        for day_idx in range(travel_days):
            if not days[day_idx]:
                for poi in group:
                    if poi.name in assigned:
                        continue
                    if _can_assign(
                        poi,
                        days[day_idx],
                        day_budget[day_idx],
                        day_minutes[day_idx],
                        day_categories[day_idx],
                    ):
                        days[day_idx].append(poi)
                        assigned.add(poi.name)
                        day_budget[day_idx] -= _resolve_poi_cost(poi)
                        day_minutes[day_idx] -= _resolve_duration(poi) + _estimate_transit(poi, center)
                        day_categories[day_idx].add(poi.category)
                break

    # Track days that are dedicated to full-day remote excursions
    day_full_remote = [
        any(remote_class.get(p.name) == "full_day" for p in day)
        for day in days
    ]

    # 3. Remaining POIs — constraint check for EACH assignment
    remaining = [p for p in pois if p.name not in assigned]
    remaining.sort(key=lambda p: p.score, reverse=True)

    for poi in remaining:
        # Prefer the day with the most remaining time to spread POIs evenly
        candidate_days = sorted(
            range(travel_days),
            key=lambda i: day_minutes[i],
            reverse=True,
        )
        for day_idx in candidate_days:
            # Full-day remote days stay dedicated to their remote excursion
            if day_full_remote[day_idx] and remote_class.get(poi.name) != "full_day":
                continue
            if _can_assign(
                poi,
                days[day_idx],
                day_budget[day_idx],
                day_minutes[day_idx],
                day_categories[day_idx],
            ):
                days[day_idx].append(poi)
                assigned.add(poi.name)
                day_budget[day_idx] -= _resolve_poi_cost(poi)
                day_minutes[day_idx] -= _resolve_duration(poi) + _estimate_transit(poi, center)
                day_categories[day_idx].add(poi.category)
                break

    return days


def _can_assign(
    poi: ScoredPOI,
    day: list[ScoredPOI],
    budget_remaining: float,
    minutes_remaining: int,
    categories_in_day: set[str],
) -> bool:
    """Check whether ``poi`` can be added to ``day`` without breaking constraints."""
    cost = _resolve_poi_cost(poi)
    if budget_remaining - cost < 0:
        return False  # over budget

    duration = _resolve_duration(poi)
    if minutes_remaining - duration < 60:
        return False  # not enough time (keep at least 60 min buffer)

    same_category = sum(1 for p in day if p.category == poi.category)
    if same_category >= 2:
        return False  # max 2 POIs of the same category per day

    return True


def _estimate_transit(poi: ScoredPOI, center: Optional[Location]) -> int:
    """Estimate one-way transit time in minutes for scheduling purposes."""
    if poi.location and center:
        return int(_distance_km(center, poi.location) / 50.0 * 60.0)
    return 30  # conservative default


def _assign_time_slots(
    day_pois: list[ScoredPOI],
    day_idx: int,
    weather: list[WeatherDay],
    profile: UserProfile,
    nearby_restaurants: list[ScoredPOI],
) -> list[Activity]:
    """Assign concrete start/end times using real transit and opening hours.

    Transit between consecutive POIs is computed from their locations
    (``distance_km / 50 * 60`` minutes). Meals are inserted using real
    nearby restaurant POIs when the schedule enters a lunch/dinner window.
    """
    activities: list[Activity] = []
    current_time = DAY_START
    last_meal_time = -1000
    day_names = {p.name for p in day_pois}
    nearby = [r for r in nearby_restaurants if r.name not in day_names]

    for i, poi in enumerate(day_pois):
        # Real transit from previous POI
        if i > 0 and poi.location and day_pois[i - 1].location:
            dist = _distance_km(day_pois[i - 1].location, poi.location)
            current_time += int(dist / 50.0 * 60.0)
        elif i > 0:
            current_time += 15  # small buffer when locations are missing

        duration = _resolve_duration(poi)

        # Opening hours check
        open_min = _time_to_minutes(poi.open_time)
        close_min = _time_to_minutes(poi.close_time)
        if open_min is not None and current_time < open_min:
            current_time = open_min  # wait until open
        if close_min is not None and current_time + duration > close_min:
            logger.warning(
                "%s ends after closing time on day %d", poi.name, day_idx + 1
            )

        # Meal insertion (reuse real restaurant matching logic)
        has_upcoming_restaurant = any(
            p.category == "restaurant" for p in day_pois[i:]
        )

        # Lunch window 11:30-13:30
        if 11 * 60 + 30 <= current_time <= 13 * 60 + 30:
            if current_time - last_meal_time >= 3.5 * 60 and not has_upcoming_restaurant:
                meal = _create_meal_activity(
                    day_idx, "lunch", profile, current_time, nearby
                )
                if meal is not None:
                    activities.append(meal)
                    current_time += MEAL_DURATION
                    last_meal_time = current_time

        # Dinner window 17:30-19:30
        if 17 * 60 + 30 <= current_time <= 19 * 60 + 30:
            if current_time - last_meal_time >= 3.5 * 60 and not has_upcoming_restaurant:
                meal = _create_meal_activity(
                    day_idx, "dinner", profile, current_time, nearby
                )
                if meal is not None:
                    activities.append(meal)
                    current_time += MEAL_DURATION
                    last_meal_time = current_time

        if current_time + duration > DAY_END:
            logger.warning("Cannot fit %s into day %d", poi.name, day_idx + 1)
            break

        # Avoid generic recommendation text in scheduler output
        if poi.description:
            reason = poi.description
        elif poi.category == "restaurant":
            reason = f"在{poi.name}品味地道风味"
        else:
            reason = f"体验{poi.name}的当地特色"

        activity = Activity(
            poi_name=poi.name,
            poi_id=poi.name,
            category=poi.category,
            start_time=_min_to_time(current_time),
            end_time=_min_to_time(current_time + duration),
            duration_min=duration,
            location=poi.location,
            recommendation_reason=reason,
            ticket_price=poi.ticket_price,
            meal_cost=80 if poi.category == "restaurant" else None,
            time_constraint=poi.time_constraint,
            tags=poi.tags,
            open_time=poi.open_time,
            close_time=poi.close_time,
        )
        activities.append(activity)
        current_time += duration + 15  # buffer between activities

    # Cap consecutive restaurant items at 2
    activities = _limit_consecutive_restaurants(activities, max_consecutive=2)
    return activities


def _time_to_minutes(time_str: Optional[str]) -> Optional[int]:
    """Convert a HH:MM string to minutes since midnight."""
    if not time_str:
        return None
    try:
        h, m = map(int, str(time_str).split(":"))
        return h * 60 + m
    except (ValueError, TypeError):
        return None


def _sanity_check(schedule: list[DayPlan], profile: UserProfile) -> list[str]:
    """Lightweight final check: report issues but never repair."""
    warnings: list[str] = []
    budget_limit = profile.budget_range or float("inf")

    for day in schedule:
        if not day.activities:
            warnings.append(f"Day {day.day_number} has no activities")
            continue

        # Time overlap check
        for i in range(len(day.activities) - 1):
            a1 = day.activities[i]
            a2 = day.activities[i + 1]
            if a1.end_time and a2.start_time and a1.end_time > a2.start_time:
                warnings.append(
                    f"Day {day.day_number} time overlap between {a1.poi_name} and {a2.poi_name}"
                )

    # Total trip budget check
    total_cost = sum(day.total_cost for day in schedule)
    if total_cost > budget_limit:
        warnings.append(
            f"Total trip cost {total_cost:.0f} exceeds budget {budget_limit:.0f}"
        )

    return warnings


def _ensure_daily_diversity(
    day_assignments: list[list[ScoredPOI]],
    all_pois: list[ScoredPOI],
) -> list[list[ScoredPOI]]:
    """Replace excess same-category POIs with a different-category alternative.

    If a day has ≥3 POIs of the same category, swap one of them with the
    highest-scoring unscheduled POI of a different category.
    """
    for day in day_assignments:
        if len(day) < 3:
            continue
        # Skip remote/sparse days: if the day has ≤2 non-restaurant POIs
        # it's likely a remote-excursion day that should stay isolated.
        non_restaurant = [p for p in day if p.category != "restaurant"]
        if len(non_restaurant) <= 2:
            continue
        day_names = {p.name for p in day}
        categories: dict[str, list[ScoredPOI]] = {}
        for p in day:
            categories.setdefault(p.category, []).append(p)

        for cat, cat_pois in categories.items():
            if len(cat_pois) >= 3:
                alternatives = [
                    p for p in all_pois if p.category != cat and p.name not in day_names
                ]
                if alternatives:
                    alternatives.sort(key=lambda p: p.score, reverse=True)
                    replacement = alternatives[0]
                    # Remove the lowest-scoring POI of the excess category
                    cat_pois.sort(key=lambda p: p.score)
                    day.remove(cat_pois[0])
                    day.append(replacement)
                    day_names.add(replacement.name)
    return day_assignments


def _max_pois_per_day(profile: UserProfile) -> int:
    """Keep generated days feasible before meals and transit buffers."""
    if profile.pace == "intensive":
        return 5
    if profile.pace == "relaxed":
        return 3
    return 4


def _find_empty_day(days: list[list[ScoredPOI]]) -> Optional[int]:
    for idx, day in enumerate(days):
        if not day:
            return idx
    return None


def _optimize_daily_routes(day_assignments: list[list[ScoredPOI]]) -> list[list[ScoredPOI]]:
    """Optimize daily routes with multi-start nearest-neighbor + 2-opt."""
    optimized = []
    for day_pois in day_assignments:
        if len(day_pois) <= 2:
            optimized.append(day_pois)
            continue

        best_route = _nearest_neighbor(day_pois)
        best_dist = _route_total_distance(best_route)

        # Multi-start NN: try every POI as the starting point
        for start_idx in range(1, len(day_pois)):
            route = _nearest_neighbor(day_pois, start_index=start_idx)
            route = _optimize_route_2opt(route)
            dist = _route_total_distance(route)
            if dist < best_dist:
                best_dist = dist
                best_route = route

        optimized.append(best_route)
    return optimized


def _nearest_neighbor(pois: list[ScoredPOI], start_index: int = 0) -> list[ScoredPOI]:
    """Greedy nearest neighbor ordering with a configurable start index."""
    if not pois:
        return []

    unvisited = set(range(len(pois)))
    route = [start_index % len(pois)]
    unvisited.remove(route[0])

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


def _route_total_distance(pois: list[ScoredPOI]) -> float:
    """Total travel distance for a route in approximate kilometers."""
    total = 0.0
    for i in range(len(pois) - 1):
        a, b = pois[i].location, pois[i + 1].location
        if a and b:
            total += _distance_km(a, b)
        else:
            total += float("inf")
    return total


def _optimize_route_2opt(
    pois: list[ScoredPOI],
    max_iterations: int = 100,
) -> list[ScoredPOI]:
    """2-opt local search to shorten a route.

    Iteratively swaps two edges (i, i+1) and (j, j+1) if reversing the
    segment between them reduces total distance.
    """
    route = list(pois)
    n = len(route)
    if n < 4:
        return route

    for _ in range(max_iterations):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n - 1):
                a, b = route[i], route[i + 1]
                c, d = route[j], route[j + 1]
                if not all(p.location for p in (a, b, c, d)):
                    continue
                # type: ignore[arg-type] is handled by the all() check above
                before = _distance_km(a.location, b.location) + _distance_km(c.location, d.location)  # type: ignore[arg-type]
                after = _distance_km(a.location, c.location) + _distance_km(b.location, d.location)  # type: ignore[arg-type]
                if after < before:
                    route[i + 1 : j + 1] = reversed(route[i + 1 : j + 1])
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return route


def _distance(a: Optional[Location], b: Optional[Location]) -> float:
    if not a or not b:
        return float("inf")
    # Simple Euclidean approximation for sorting (sufficient for nearest-neighbor)
    return ((a.lat - b.lat) ** 2 + (a.lng - b.lng) ** 2) ** 0.5


def _distance_km(a: Location, b: Location) -> float:
    """Approximate distance in kilometers."""
    lat_km = (a.lat - b.lat) * 111
    lng_km = (a.lng - b.lng) * 85
    return (lat_km**2 + lng_km**2) ** 0.5


def _limit_consecutive_restaurants(
    activities: list[Activity], max_consecutive: int = 2
) -> list[Activity]:
    """Remove excess restaurant activities so no more than ``max_consecutive"
    restaurant items appear in a row."""
    result: list[Activity] = []
    consecutive = 0
    for act in activities:
        if act.category == "restaurant":
            consecutive += 1
            if consecutive > max_consecutive:
                continue
        else:
            consecutive = 0
        result.append(act)
    return result


def _resolve_duration(poi: ScoredPOI) -> int:
    """Resolve activity duration from POI recommended_hours or defaults."""
    hours_map = {
        "1小时": 60,
        "1.5小时": 90,
        "1-2小时": 90,
        "2小时": 120,
        "2-3小时": 150,
        "3小时": 180,
        "3-4小时": 210,
        "半天": 240,
        "全天": 360,
    }
    if poi.recommended_hours and poi.recommended_hours in hours_map:
        return hours_map[poi.recommended_hours]
    if poi.category == "restaurant":
        return 90
    return 120


def _resolve_poi_cost(poi: ScoredPOI) -> float:
    """Resolve the activity cost used for budget constraints.

    Restaurant POIs carry an implicit meal cost in addition to any ticket price.
    """
    cost = poi.ticket_price or 0
    if poi.category == "restaurant":
        cost += 80
    return cost


def _create_meal_activity(
    day_idx: int,
    meal_type: str,
    profile: UserProfile,
    start_min: int,
    nearby_pois: Optional[list[ScoredPOI]] = None,
) -> Activity:
    """Create a meal activity, preferring a real nearby restaurant POI."""
    nearby_pois = nearby_pois or []
    restaurants = [p for p in nearby_pois if p.category == "restaurant"]

    # Sort by food-preference tag match, then by score
    if profile.food_preferences and restaurants:
        food_prefs = set(profile.food_preferences)
        restaurants.sort(
            key=lambda r: (-len(set(r.tags) & food_prefs), -r.score, r.name)
        )

    food_hint = (
        f"（偏好：{','.join(profile.food_preferences)}）" if profile.food_preferences else ""
    )

    if restaurants:
        r = restaurants[0]
        return Activity(
            poi_name=f"{r.name}{food_hint}",
            poi_id=r.name,
            category="restaurant",
            start_time=_min_to_time(start_min),
            end_time=_min_to_time(start_min + 90),
            duration_min=90,
            location=r.location,
            meal_cost=80,
            recommendation_reason=r.description
            or f"在{r.name}享用{','.join(profile.food_preferences) if profile.food_preferences else '当地'}美食",
            tags=r.tags,
        )

    # Fallback placeholder
    return Activity(
        poi_name=f"{meal_type.capitalize()}{food_hint}",
        category="restaurant",
        start_time=_min_to_time(start_min),
        end_time=_min_to_time(start_min + 90),
        duration_min=90,
        meal_cost=80,
        recommendation_reason=f"在附近找一家{'辣' if '辣' in profile.food_preferences else '口碑好'}的餐厅",
    )


def _min_to_time(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _append_travel_budget(schedule: list[DayPlan], profile: UserProfile) -> None:
    """Append transport and accommodation estimates to the daily budget."""
    travel_days = profile.travel_days or len(schedule) or 1
    transport_total = travel_days * 30
    accommodation_total = max(0, travel_days - 1) * 200

    if not schedule:
        return

    for day in schedule:
        day.total_cost += transport_total / travel_days

    # Add accommodation cost to the first day (simple aggregation)
    schedule[0].total_cost += accommodation_total

    # P4: build per-day budget breakdown note
    for day in schedule:
        tickets_meals = sum(
            (a.ticket_price or 0) + (a.meal_cost or 0) for a in day.activities
        )
        transport = transport_total / travel_days
        accommodation = accommodation_total if day == schedule[0] else 0
        total = tickets_meals + transport + accommodation
        lines = [
            "**费用明细**:",
            f"  门票+餐饮: ¥{tickets_meals:.0f}",
            f"  市内交通: ¥{transport:.0f} ({travel_days}天×¥30)",
        ]
        if accommodation > 0:
            lines.append(f"  住宿: ¥{accommodation:.0f} ({travel_days - 1}晚×¥200)")
        lines.append(f"  **合计: ¥{total:.0f}**")
        day.budget_note = "\n".join(lines)
