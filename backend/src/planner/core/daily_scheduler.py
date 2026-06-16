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
    day_assignments = _assign_days(
        constrained,
        groups,
        travel_days,
        max_pois_per_day=_max_pois_per_day(profile),
    )

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
    max_pois_per_day: int = 4,
) -> list[list[ScoredPOI]]:
    """Assign POIs to days by area group.

    Remote excursions (outliers by distance) are detected and classified
    into half_day / full_day / cross_city.  Cross-city POIs are excluded
    entirely.  Half-day remote days accept extra non-remote POIs using
    remaining daylight; full-day remote days only accept same-area POIs.
    """
    days: list[list[ScoredPOI]] = [[] for _ in range(travel_days)]
    assigned: set[str] = set()
    day_full_remote = [False] * travel_days
    remote_count = [0] * travel_days

    # ------------------------------------------------------------------ #
    # 1. Detect & classify remote excursions
    # ------------------------------------------------------------------ #
    remote_class: dict[str, str] = {}
    cross_city: set[str] = set()
    center: Optional[Location] = None

    located = [p for p in pois if p.location]
    if len(located) >= 3:
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

            if one_way_h * 2 > 5:  # round-trip > 5h → cross_city
                remote_class[p.name] = "cross_city"
                cross_city.add(p.name)
            elif one_way_h * 2 + activity_h > 9:  # round-trip +游玩 > 9h → full_day
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

    # ------------------------------------------------------------------ #
    # 2. Assign remote groups (by area) to empty days (A + G)
    # ------------------------------------------------------------------ #
    active_remote = {n: c for n, c in remote_class.items() if c != "cross_city"}
    remote_areas: dict[str, list[ScoredPOI]] = {}
    for p in pois:
        if p.name in active_remote:
            area = p.area or p.name
            remote_areas.setdefault(area, []).append(p)

    for _, group in remote_areas.items():
        for day_idx in range(travel_days):
            if not days[day_idx]:
                for poi in group:
                    days[day_idx].append(poi)
                    assigned.add(poi.name)
                    if remote_class.get(poi.name) == "full_day":
                        day_full_remote[day_idx] = True
                    if poi.name in active_remote:
                        remote_count[day_idx] += 1
                break

    # Cross-city POIs are excluded
    for name in cross_city:
        assigned.add(name)

    # ------------------------------------------------------------------ #
    # 3. Effective daily capacity (B – dynamic)
    # ------------------------------------------------------------------ #
    DAY_START = 9 * 60  # 09:00
    DAY_END = 21 * 60  # 21:00
    LUNCH_MIN = 90  # lunch slot in minutes

    def _effective_max(day_idx: int) -> int:
        if day_full_remote[day_idx]:
            total_remote = sum(_resolve_duration(p) for p in days[day_idx])
            lunch_dur = LUNCH_MIN if total_remote >= 180 else 0
            remote_end = DAY_START + total_remote + lunch_dur
            # Account for return trip
            for p in days[day_idx]:
                if p.name in active_remote and center and p.location:
                    dist = _distance_km(center, p.location)
                    if dist > 20:
                        remote_end += int(dist / 50 * 60)
                        break
            remaining = max(0, DAY_END - remote_end)
            extra = max(0, int(remaining / 150))  # ~2h activity + 0.5h buffer
            return min(remote_count[day_idx] + extra, max_pois_per_day)

        if remote_count[day_idx] > 0:  # half_day
            total_remote = sum(_resolve_duration(p) for p in days[day_idx])
            lunch_dur = LUNCH_MIN if total_remote >= 180 else 0
            max_dist = max(
                (
                    _distance_km(center, p.location)
                    for p in days[day_idx]
                    if p.name in active_remote and p.location and center
                ),
                default=0,
            )
            travel_back = int(max_dist / 50 * 60) if max_dist > 20 else 0
            remote_end = DAY_START + total_remote + lunch_dur + travel_back
            remaining = max(0, DAY_END - remote_end)
            extra = max(0, int(remaining / 150))
            return min(remote_count[day_idx] + extra, max_pois_per_day)

        return max_pois_per_day

    # ------------------------------------------------------------------ #
    # 4. Distribute non-remote POIs by area group (F – score-aware)
    # ------------------------------------------------------------------ #
    candidate_pool = [p for p in pois if p.name not in assigned]
    ngroups: dict[str, list[ScoredPOI]] = {}
    for p in candidate_pool:
        ngroups.setdefault(p.area or "其他", []).append(p)

    sorted_ngroups = sorted(
        ngroups.items(),
        key=lambda item: (
            sum(pp.score for pp in item[1]) / len(item[1]) * len(item[1]),
            len(item[1]),
        ),
        reverse=True,
    )

    for group_idx, (_, group_pois) in enumerate(sorted_ngroups):
        start_idx = group_idx % travel_days
        for poi in group_pois:
            for offset in range(travel_days):
                idx = (start_idx + offset) % travel_days
                if len(days[idx]) < _effective_max(idx):
                    days[idx].append(poi)
                    assigned.add(poi.name)
                    break

    # ------------------------------------------------------------------ #
    # 5. Remaining POIs — skip, never abort the whole tail (C)
    # ------------------------------------------------------------------ #
    tail = [p for p in pois if p.name not in assigned]
    di = 0
    for poi in tail:
        for _ in range(travel_days):
            if len(days[di]) < _effective_max(di):
                days[di].append(poi)
                assigned.add(poi.name)
                di = (di + 1) % travel_days
                break
            di = (di + 1) % travel_days

    # ------------------------------------------------------------------ #
    # 6. Rebalance sparse days (H – dynamic)
    # ------------------------------------------------------------------ #
    total_placed = sum(len(d) for d in days)
    avg = total_placed / travel_days
    for _ in range(3):
        sparse = [i for i in range(travel_days) if len(days[i]) < avg * 0.5]
        full = [i for i in range(travel_days) if len(days[i]) > avg * 1.3]
        for si in sparse:
            for fi in full:
                if len(days[fi]) > max(1, int(avg)) and len(days[si]) < _effective_max(si):
                    days[si].append(days[fi].pop())

    return days


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


def _distance_km(a: Location, b: Location) -> float:
    """Approximate distance in kilometers."""
    lat_km = (a.lat - b.lat) * 111
    lng_km = (a.lng - b.lng) * 85
    return (lat_km**2 + lng_km**2) ** 0.5


def _build_day_plans(
    day_pois: list[list[ScoredPOI]],
    weather: list[WeatherDay],
    profile: UserProfile,
) -> list[DayPlan]:
    """Build daily schedule with meal insertion and time allocation."""
    schedule = []
    day_start_min = 9 * 60  # 09:00
    day_end_min = 21 * 60  # 21:00

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
                    meal = _create_meal_activity(day_idx, "lunch", profile, current_time)
                    day.activities.append(meal)
                    current_time += 90
                    last_meal_time = current_time

            # Dinner insertion window 17:30-19:30
            if 17 * 60 + 30 <= current_time <= 19 * 60 + 30:
                if current_time - last_meal_time >= 3.5 * 60:
                    meal = _create_meal_activity(day_idx, "dinner", profile, current_time)
                    day.activities.append(meal)
                    current_time += 90
                    last_meal_time = current_time

            duration = _resolve_duration(poi)
            if current_time + duration > day_end_min:
                break

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

        day.total_cost = sum((a.ticket_price or 0) + (a.meal_cost or 0) for a in day.activities)
        schedule.append(day)

    return schedule


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


def _create_meal_activity(
    day_idx: int,
    meal_type: str,
    profile: UserProfile,
    start_min: int,
) -> Activity:
    """Create a meal activity placeholder."""
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
        recommendation_reason=f"在附近找一家{'辣' if '辣' in profile.food_preferences else '口碑好'}的餐厅",
    )


def _min_to_time(minutes: int) -> str:
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"
