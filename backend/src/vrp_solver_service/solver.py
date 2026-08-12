"""Standalone VRP-TW solver aligned with v4.0 production-grade design."""

from __future__ import annotations

import logging
import math
import re
import time
from typing import Any

from ortools.sat.python import cp_model

from planner.preprocessing import (
    CPSATTuningGuide,
    FatigueModel,
    PlayTimeManager,
    ReservationHandler,
    RestaurantHandler,
    TransportSelector,
)
from vrp_solver_service.callback import TimeoutCallback
from vrp_solver_service.models import (
    ActivityOutput,
    ConstraintsInput,
    DayPlanOutput,
    POIInput,
    ReminderOutput,
    SolverRequest,
    SolverResponse,
)

logger = logging.getLogger(__name__)

HOTEL_ID = "__hotel"
DAY_START_MIN = 8 * 60  # 08:00
DAY_END_MIN = 18 * 60  # 18:00
MAX_POI_PER_DAY = 5


class TravelVRPSolver:
    """VRP-TW itinerary solver with preprocessing and CP-SAT + greedy fallback."""

    def __init__(self) -> None:
        self._reservation_handler = ReservationHandler()
        self._play_time_manager = PlayTimeManager()
        self._restaurant_handler = RestaurantHandler()
        self._transport_selector = TransportSelector()
        self._fatigue_model = FatigueModel()
        self._tuning_guide = CPSATTuningGuide()

    def solve(self, request: SolverRequest) -> SolverResponse:
        """Run full preprocessing → solve → build response."""
        start_ts = time.time()
        pois = list(request.pois)
        constraints = request.constraints

        if not pois:
            return SolverResponse(status="infeasible", message="No POIs provided")

        # 1. Reservation filtering and reminders
        pois, reminders = self._reservation_handler.filter_and_remind(
            pois,
            constraints.user_reservations,
        )

        # 2. Play time adjustment
        pois = self._play_time_manager.adjust(pois, constraints)

        # 3. Restaurant injection (opt-in)
        pois = self._restaurant_handler.inject(pois, constraints)

        # 4. Hotel node injection (index 0)
        pois = self._inject_hotel(pois)

        # 5. Transport matrices (hotel is virtual start/end -> zero commute)
        if request.dist_matrix is not None and request.tc_matrix is not None:
            dist = request.dist_matrix
            tc = request.tc_matrix
        else:
            dist, tc = self._transport_selector.build_matrices(pois, constraints, request.amap_minutes)
            for i in range(len(pois)):
                dist[0][i] = 0
                dist[i][0] = 0
                tc[0][i] = 0.0
                tc[i][0] = 0.0

        # 6. Fatigue-based walk limits
        walk_limits = self._fatigue_model.daily_walk_limits(constraints)

        # 7. Strategy selection
        n_real = len(pois) - 1  # excluding hotel
        use_greedy = request.strategy == "greedy" or (
            request.strategy == "auto" and n_real <= 15 and constraints.travel_days <= 3
        )

        try:
            if use_greedy:
                logger.info("Using greedy heuristic (%d POIs, %d days)", n_real, constraints.travel_days)
                days = _greedy_solve(pois, constraints, dist, tc, walk_limits)
                status = "fallback"
            else:
                logger.info("Using CP-SAT (%d POIs, %d days)", n_real, constraints.travel_days)
                params = self._tuning_guide.recommend(constraints, len(pois))
                days = _cpsat_solve(
                    pois,
                    constraints,
                    dist,
                    tc,
                    walk_limits,
                    params,
                )
                status = "optimal"
        except Exception as exc:
            logger.exception("Solver failed: %s", exc)
            days = _greedy_solve(pois, constraints, dist, tc, walk_limits)
            status = "fallback"
            reminders.append(
                ReminderOutput(
                    type="reservation",
                    message=f"求解器异常，已降级为贪心求解：{exc}",
                )
            )

        # Over-capacity advisory: each far-suburb landmark eats a whole day to one
        # of long one-way commutes, so too many for the trip length means some get
        # dropped or every day is a slog. We do NOT silently decide which to cut —
        # we surface the tension (and name any suburb that was dropped) and let the
        # user cut one or add a day.
        # A day is "consumed" by a far suburb OR by a full-day landmark (theme park
        # ~6h+): both leave no room for other sightseeing that day. 迪士尼/环球 sit
        # ~16km from centre so they are not geographic suburbs, but still own a day.
        day_consuming = set(_suburban_indices(pois, constraints))
        day_consuming |= {
            i
            for i in range(1, len(pois))
            if not pois[i].id.startswith("__meal_d")
            and pois[i].duration_minutes >= 360
        }
        if day_consuming:
            scheduled_ids = {a.poi_id for day in days for a in day.activities}
            cand = [pois[i].name for i in day_consuming]
            dropped = [pois[i].name for i in day_consuming if pois[i].id not in scheduled_ids]
            # ceil(days / 2): suburbs filling half the trip (or more) is the warning
            # line — 3 suburbs in 5 days, 2 in 3 days, etc.
            threshold = (constraints.travel_days + 1) // 2
            if len(cand) >= 2 and len(cand) >= threshold:
                msg = (
                    f"行程含 {len(cand)} 个远郊景点（{', '.join(cand)}），"
                    f"各需独立一天且互相通勤很长，{constraints.travel_days} 天内偏紧。"
                )
                if dropped:
                    msg += f"其中 {', '.join(dropped)} 未能排入。"
                msg += "建议删减其一或延长行程。"
                reminders.append(ReminderOutput(type="capacity", message=msg))

        solve_time_ms = int((time.time() - start_ts) * 1000)
        return SolverResponse(
            status=status,
            days=days,
            reminders=reminders,
            solve_time_ms=solve_time_ms,
            message=None if status != "fallback" else "Greedy fallback used",
            metadata={
                "poi_count": n_real,
                "travel_days": constraints.travel_days,
                "hotel_injected": True,
            },
        )

    def greedy_solve(self, request: SolverRequest) -> SolverResponse:
        """Fast greedy solve for fallback scenarios."""
        start_ts = time.time()
        pois = list(request.pois)
        constraints = request.constraints

        # Mirror the full-solve preprocessing so the fallback path also gets
        # realistic landmark dwell times and meal breaks (not just raw POIs).
        pois, reminders = self._reservation_handler.filter_and_remind(
            pois,
            constraints.user_reservations,
        )
        pois = self._play_time_manager.adjust(pois, constraints)
        pois = self._restaurant_handler.inject(pois, constraints)
        pois = self._inject_hotel(pois)

        if request.dist_matrix is not None and request.tc_matrix is not None:
            dist = request.dist_matrix
            tc = request.tc_matrix
        else:
            dist, tc = self._transport_selector.build_matrices(pois, constraints, request.amap_minutes)

        # The virtual hotel is a start/end anchor; zero its commute so the
        # solver does not reject every POI because the hotel is at (0, 0).
        for i in range(len(pois)):
            dist[0][i] = 0
            dist[i][0] = 0
            tc[0][i] = 0.0
            tc[i][0] = 0.0

        walk_limits = self._fatigue_model.daily_walk_limits(constraints)
        days = _greedy_solve(pois, constraints, dist, tc, walk_limits)
        return SolverResponse(
            status="fallback",
            days=days,
            reminders=reminders,
            solve_time_ms=int((time.time() - start_ts) * 1000),
            message="Greedy fallback",
            metadata={"poi_count": len(pois) - 1, "travel_days": constraints.travel_days},
        )

    @staticmethod
    def _inject_hotel(pois: list[POIInput]) -> list[POIInput]:
        """Add a virtual hotel node at index 0."""
        hotel = POIInput(
            id=HOTEL_ID,
            name="酒店",
            category="hotel",
            lat=0.0,
            lng=0.0,
            score=0.0,
            ticket_price=0.0,
            duration_minutes=0,
            open_time="00:00",
            close_time="23:59",
            walk_intensity=0,
        )
        return [hotel] + pois


def _time_to_minutes(time_str: str | None) -> int:
    if not time_str:
        return DAY_START_MIN
    parts = time_str.split(":")
    try:
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return DAY_START_MIN


def _fmt_time(minutes: int) -> str:
    minutes = max(0, minutes)
    h = minutes // 60
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _geo_day_clusters(
    pois: list[POIInput],
    days_count: int,
    dist: list[list[int]],
    max_per_day: int,
) -> dict[int, list[int]]:
    """Partition real attractions into ``days_count`` proximity clusters.

    Used only as a CP-SAT warm-start hint. Seeds with the mutually farthest POIs
    (greedy max-min dispersion → one seed per geographic area), then assigns the
    rest to the nearest seed's day with spare capacity. Meal/hotel nodes are
    excluded (meals are pinned by hard constraints). Travel time ``dist`` drives
    the grouping, so a tight cluster (陆家嘴) lands on one day instead of scattering.
    """
    if days_count <= 0:
        return {}
    attr = [i for i in range(1, len(pois)) if not pois[i].id.startswith("__meal_d")]
    if not attr:
        return {}
    k = min(days_count, len(attr))
    seeds = [attr[0]]
    while len(seeds) < k:
        best, best_d = None, -1
        for i in attr:
            if i in seeds:
                continue
            nearest = min(dist[i][s] for s in seeds)
            if nearest > best_d:
                best, best_d = i, nearest
        if best is None:
            break
        seeds.append(best)
    clusters: dict[int, list[int]] = {d: [] for d in range(days_count)}
    assigned: set[int] = set()
    for d, s in enumerate(seeds):
        clusters[d].append(s)
        assigned.add(s)
    for i in attr:
        if i in assigned:
            continue
        for d in sorted(range(len(seeds)), key=lambda d: dist[i][seeds[d]]):
            if len(clusters[d]) < max_per_day:
                clusters[d].append(i)
                assigned.add(i)
                break
    return clusters


def _suburban_indices(pois: list[POIInput], constraints: ConstraintsInput) -> set[int]:
    """Indices of attractions that lie far outside the itinerary's centre.

    Uses the *median* coordinate of all real attractions (robust to the very
    outliers we want to flag) and returns those beyond suburb_radius_km. Hotel and
    meal dummy nodes are ignored. Empty when the radius is disabled or coords are
    missing — callers then skip suburb isolation entirely.
    """
    radius = getattr(constraints, "suburb_radius_km", 0) or 0
    if radius <= 0:
        return set()
    real = [
        i
        for i in range(1, len(pois))
        if not pois[i].id.startswith("__meal_d") and (pois[i].lat or pois[i].lng)
    ]
    if len(real) < 3:
        return set()
    lats = sorted(pois[i].lat for i in real)
    lngs = sorted(pois[i].lng for i in real)
    mid = len(real) // 2
    c_lat = lats[mid] if len(real) % 2 else (lats[mid - 1] + lats[mid]) / 2
    c_lng = lngs[mid] if len(real) % 2 else (lngs[mid - 1] + lngs[mid]) / 2
    centroid = POIInput(id="__centroid", name="", lat=c_lat, lng=c_lng)
    return {i for i in real if _haversine_km(pois[i], centroid) > radius}


def _haversine_km(a: POIInput, b: POIInput) -> float:
    lat1, lng1 = math.radians(a.lat), math.radians(a.lng)
    lat2, lng2 = math.radians(b.lat), math.radians(b.lng)
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    s = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(s))


def _cpsat_solve(
    pois: list[POIInput],
    constraints: ConstraintsInput,
    dist: list[list[int]],
    tc: list[list[float]],
    walk_limits: list[int],
    params: dict[str, Any],
) -> list[DayPlanOutput]:
    n = len(pois)
    # Meal dummy nodes are time-window obligations, NOT travel stops. They are kept
    # out of the routing circuit (no X arcs): when meals were routed nodes with
    # zero-cost edges (coords 0,0) the tour could go 景点A→午餐→景点B for free,
    # teleporting past dist(A,B) and zeroing the whole travel objective — which is
    # why geographic clustering never took effect. The circuit now routes only the
    # hotel + real attractions (real travel signal intact); meals are sequenced
    # into the day by a no-overlap constraint and counted in the time/cost budgets.
    meal_set = {i for i in range(1, n) if pois[i].id.startswith("__meal_d")}
    route_nodes = [i for i in range(n) if i not in meal_set]  # hotel + real POIs

    days_count = constraints.travel_days
    day_start = constraints.day_start_min or DAY_START_MIN
    day_end = constraints.day_end_min or DAY_END_MIN
    day_available = day_end - day_start
    total_budget = constraints.total_budget or float("inf")
    day_budget = total_budget / days_count if total_budget != float("inf") else float("inf")
    max_transit = constraints.max_transit_minutes
    rest_day = constraints.rest_day
    food_day = constraints.food_day

    model = cp_model.CpModel()

    # Node 0 is hotel; real POIs are 1..n-1
    durations = [p.duration_minutes for p in pois]
    costs = [int(p.ticket_price) for p in pois]
    walks = [p.walk_intensity for p in pois]
    open_times = [_time_to_minutes(p.open_time) for p in pois]
    close_times = [_time_to_minutes(p.close_time) for p in pois]

    # Decision variables
    X: dict[tuple[int, int, int], cp_model.IntVar] = {}
    V: dict[tuple[int, int], cp_model.IntVar] = {}
    A: dict[tuple[int, int], cp_model.IntVar] = {}
    DC: dict[int, cp_model.IntVar] = {}
    DW: dict[int, cp_model.IntVar] = {}

    # Per-arc slack so the day is not packed minute-to-minute: a fixed transition
    # buffer (walk to station / wait / photos / rest) plus an extra queue pad when
    # arriving at a peak-crowd POI. Only applied between real POIs (hotel arcs are
    # zeroed start/end anchors). Used in the time-propagation and day-duration
    # constraints; NOT in the max_transit feasibility test, which checks raw road
    # time so the buffer can never forbid an otherwise-reachable arc.
    _buf = max(0, constraints.transition_buffer_min or 0)
    _peak_pad = max(0, constraints.peak_queue_pad_min or 0)

    def _arc_extra(i: int, j: int) -> int:
        if i == 0 or j == 0:
            return 0
        return _buf + (_peak_pad if pois[j].is_peak else 0)

    M_time = max(close_times)
    # Big-M for the commute time-propagation constraints. It must dominate the
    # largest possible value of (A[i] + duration[i] + travel) so that a *deselected*
    # arc (X == 0) leaves A[j] unconstrained. Using only max(dist)+max(duration)
    # (~100) is far too small — A can reach day_end (~1200), so deselected arcs
    # imposed impossible bounds and the whole model went infeasible (the second
    # reason "CP-SAT" silently fell back to greedy on every request).
    _max_travel = max((max(row) for row in dist), default=0)
    M_travel = day_end + max(durations) + _max_travel + _buf + _peak_pad + 1

    for d in range(days_count):
        for i in range(n):
            V[(d, i)] = model.NewBoolVar(f"V_{d}_{i}")
            A[(d, i)] = model.NewIntVar(0, day_end, f"A_{d}_{i}")
        # X arcs only between routed nodes (hotel + real POIs); meals never route.
        for i in route_nodes:
            for j in route_nodes:
                if i != j:
                    X[(d, i, j)] = model.NewBoolVar(f"X_{d}_{i}_{j}")
        DC[d] = model.NewIntVar(0, int(day_budget * 2) if day_budget != float("inf") else 100000, f"DC_{d}")
        DW[d] = model.NewIntVar(0, max(walk_limits) * 2, f"DW_{d}")

    # Constraint 1: AddCircuit for each day over routed nodes (hotel + real POIs)
    for d in range(days_count):
        arcs = []
        for i in route_nodes:
            for j in route_nodes:
                if i == j:
                    continue
                arcs.append((i, j, X[(d, i, j)]))
            self_loop = model.NewBoolVar(f"sl_{d}_{i}")
            arcs.append((i, i, self_loop))
        model.AddCircuit(arcs)

    # Constraint 2: visit-edge linkage for routed POIs (excluding hotel)
    for d in range(days_count):
        for i in route_nodes:
            if i == 0:
                continue
            model.Add(sum(X[(d, j, i)] for j in route_nodes if j != i) == V[(d, i)])
            model.Add(sum(X[(d, i, j)] for j in route_nodes if j != i) == V[(d, i)])

    # Hotel is start/end each day
    for d in range(days_count):
        model.Add(V[(d, 0)] == 1)
        model.Add(A[(d, 0)] == day_start)

    # Constraint 2b: each real POI visited at most once
    for i in range(1, n):
        model.Add(sum(V[(d, i)] for d in range(days_count)) <= 1)

    # Constraint 2c: must-visit POIs must appear exactly once
    must_indices = []
    for target in constraints.must_visit:
        for idx, p in enumerate(pois):
            if idx == 0:
                continue
            if p.id == target or p.name == target:
                must_indices.append(idx)
                break
    for i in must_indices:
        model.Add(sum(V[(d, i)] for d in range(days_count)) == 1)

    # Full-day landmarks (theme parks etc.) occupy essentially the whole day, so
    # their open hours overlap the lunch window — a separate lunch node cannot
    # coexist and the landmark would be dropped. They are must-visit and the day
    # that hosts one is exempt from the mandatory meal nodes (you eat inside the
    # park). FULLDAY_MIN ~= 6h catches Disney/Universal/长隆/欢乐谷 but not a zoo.
    FULLDAY_MIN = 360
    fullday_landmarks = [
        i
        for i in range(1, n)
        if durations[i] >= FULLDAY_MIN and not pois[i].id.startswith("__meal_d")
    ]
    for i in fullday_landmarks:
        model.Add(sum(V[(d, i)] for d in range(days_count)) == 1)

    # has_fullday[d] == 1 iff day d hosts a full-day landmark.
    has_fullday: dict[int, Any] = {}
    for d in range(days_count):
        flag = model.NewBoolVar(f"fullday_{d}")
        if fullday_landmarks:
            model.AddMaxEquality(flag, [V[(d, i)] for i in fullday_landmarks])
        else:
            model.Add(flag == 0)
        has_fullday[d] = flag

    # Constraint 2d: meals are mandatory and bound to their target day. Meal dummy
    # nodes are named "__meal_d{day}_m{meal}_..." so each must be visited on its
    # own day, guaranteeing exactly meals_per_day meals (lunch + dinner) per day —
    # UNLESS that day hosts a full-day landmark, in which case meals happen inside
    # and the separate node is dropped so the landmark fits.
    for idx, p in enumerate(pois):
        if idx == 0:
            continue
        meal_match = re.match(r"__meal_d(\d+)_", p.id)
        if not meal_match:
            continue
        target_day = int(meal_match.group(1)) - 1  # "__meal_d2_..." -> day index 1
        if not 0 <= target_day < days_count:
            continue
        for d in range(days_count):
            if d != target_day:
                model.Add(V[(d, idx)] == 0)
        # On its own day: required normally, but optional (off) on a landmark day.
        model.Add(V[(target_day, idx)] == 1).OnlyEnforceIf(has_fullday[target_day].Not())
        model.Add(V[(target_day, idx)] == 0).OnlyEnforceIf(has_fullday[target_day])

    # Constraint 2e: remote-pair same-day exclusion. Two attractions whose
    # road-network time exceeds remote_pair_min cannot share a day, so a far-suburb
    # POI (松江辰山) is isolated onto its own light day instead of being bundled
    # with a downtown cluster (陆家嘴) — the source of 3h+ same-day cross-city
    # folding. Meal dummy nodes sit at (0,0) so their matrix distance is meaningless;
    # they are excluded. must-visit pairs are still excluded from each other (you
    # genuinely cannot do both well in one day), which simply spreads them apart.
    _remote_min = max(0, constraints.remote_pair_min or 0)
    if _remote_min:
        _real = [
            i for i in range(1, n) if not pois[i].id.startswith("__meal_d")
        ]
        for a_idx in range(len(_real)):
            for b_idx in range(a_idx + 1, len(_real)):
                i, j = _real[a_idx], _real[b_idx]
                if dist[i][j] > _remote_min:
                    for d in range(days_count):
                        model.Add(V[(d, i)] + V[(d, j)] <= 1)

    # Constraint 2f: geographic suburb isolation. Driving time underestimates the
    # transit pain of reaching a far suburb (松江辰山, 惠南野生动物园, 川沙迪士尼),
    # so such a POI can sneak under remote_pair_min and still bundle with downtown.
    # Classify by straight-line distance from the itinerary centroid instead: a
    # suburb POI may share its day only with a genuinely adjacent attraction
    # (<= suburb_nearby_min driving), giving each far landmark its own day.
    suburban_idx = _suburban_indices(pois, constraints)
    if suburban_idx:
        _nearby = max(0, constraints.suburb_nearby_min or 0)
        _real_attr = [
            i for i in range(1, n) if not pois[i].id.startswith("__meal_d")
        ]
        for i in suburban_idx:
            for j in _real_attr:
                if j == i:
                    continue
                if dist[i][j] > _nearby:
                    for d in range(days_count):
                        model.Add(V[(d, i)] + V[(d, j)] <= 1)

    # Constraint 2g: weekly closing day (周一闭馆). A venue closed on weekday W
    # cannot be visited on any travel day that falls on W. Only active when the
    # trip's real weekdays are known; otherwise we never guess which day is Monday.
    day_weekdays = constraints.day_weekdays or []
    if day_weekdays:
        for i in range(1, n):
            closed = set(pois[i].closed_weekdays or [])
            if not closed:
                continue
            for d in range(days_count):
                if d < len(day_weekdays) and day_weekdays[d] in closed:
                    model.Add(V[(d, i)] == 0)

    # Constraint 3: conditional time windows
    for d in range(days_count):
        for i in range(1, n):
            model.Add(A[(d, i)] >= open_times[i] - M_time * (1 - V[(d, i)]))
            model.Add(A[(d, i)] + durations[i] <= close_times[i] + M_time * (1 - V[(d, i)]))
            model.Add(A[(d, i)] <= M_time * V[(d, i)])

    # Constraint 4: exact commute propagation (bidirectional)
    # NOTE: the arc back to the hotel (j == 0) must NOT propagate time, because
    # A[hotel] is pinned to day_start. Propagating would force
    # day_start >= A[last] + duration[last], which is impossible whenever any POI
    # is visited — silently making EVERY day infeasible and forcing the greedy
    # fallback. (This is why the "CP-SAT" solver never actually ran.)
    for d in range(days_count):
        for i in route_nodes:
            for j in route_nodes:
                if i == j:
                    continue
                travel = dist[i][j]
                if travel > max_transit:
                    model.Add(X[(d, i, j)] == 0)
                elif j != 0:
                    # Lower bound only: you cannot arrive at j before finishing i
                    # and travelling there (+ transition/queue slack). We must NOT
                    # also bound it from above — an upper bound forces arrival ==
                    # prev_end + travel, forbidding any wait, which makes it
                    # impossible to arrive early and wait for a venue (or meal
                    # window) to open. The previous bidirectional equality silently
                    # made every itinerary that needed to wait for opening hours
                    # infeasible.
                    travel_eff = travel + _arc_extra(i, j)
                    model.Add(A[(d, j)] >= A[(d, i)] + durations[i] + travel_eff - M_travel * (1 - X[(d, i, j)]))

    # Constraint 5: daily duration excluding return-to-hotel commute. Commute
    # includes the per-arc transition/queue slack so the day budget reflects
    # realistic time-on-the-ground, not a minute-perfect packing.
    _route_real = [i for i in route_nodes if i != 0]
    for d in range(days_count):
        model.Add(
            sum(V[(d, i)] * durations[i] for i in range(1, n))
            + sum(
                X[(d, i, j)] * (dist[i][j] + _arc_extra(i, j))
                for i in _route_real
                for j in _route_real
                if i != j
            )
            + rest_day
            <= day_available
        )

    # Constraint 5b: one-thing-at-a-time per day. Meals are no longer in the routing
    # circuit, so nothing else stops a meal's time block overlapping a sightseeing
    # block. Optional intervals (present iff V) + AddNoOverlap force the meal into a
    # real gap in the day — between attractions — exactly where you'd stop to eat.
    _big_end = day_end + max(durations) + 1
    for d in range(days_count):
        intervals = []
        for i in range(1, n):
            end_var = model.NewIntVar(0, _big_end, f"end_{d}_{i}")
            intervals.append(
                model.NewOptionalIntervalVar(
                    A[(d, i)], durations[i], end_var, V[(d, i)], f"iv_{d}_{i}"
                )
            )
        model.AddNoOverlap(intervals)

    # Constraint 6: daily walking limit
    for d in range(days_count):
        model.Add(DW[d] == sum(V[(d, i)] * walks[i] for i in range(1, n)))
        model.Add(DW[d] <= walk_limits[d])

    # Constraint 7: budget (daily + total) with MAD
    total_tc_expr = sum(
        X[(d, i, j)] * int(tc[i][j])
        for d in range(days_count)
        for i in route_nodes
        for j in route_nodes
        if i != j
    )
    for d in range(days_count):
        day_tc = sum(
            X[(d, i, j)] * int(tc[i][j])
            for i in route_nodes
            for j in route_nodes
            if i != j
        )
        model.Add(DC[d] == int(food_day) + sum(V[(d, i)] * costs[i] for i in range(1, n)) + day_tc)
        if day_budget != float("inf"):
            model.Add(DC[d] <= int(day_budget))

    if total_budget != float("inf"):
        model.Add(sum(DC[d] for d in range(days_count)) + total_tc_expr <= int(total_budget))

    # Track the daily cost band for budget capping (max) and load balancing.
    # NOTE: the objective must penalise the *spread* (max - min) across days, not
    # the absolute level. DC includes the fixed food_day plus tickets; penalising
    # the level made every extra (ticketed) POI look like a loss, so the solver
    # left days nearly empty. The spread cancels the constant food_day and only
    # discourages lumping all the expensive POIs into one day.
    max_day_cost = model.NewIntVar(0, 100000, "max_day_cost")
    min_day_cost = model.NewIntVar(0, 100000, "min_day_cost")
    for d in range(days_count):
        model.Add(max_day_cost >= DC[d])
        model.Add(min_day_cost <= DC[d])
    day_cost_spread = max_day_cost - min_day_cost

    # Constraint 8: per-day attraction cap. Meal nodes are excluded so the cap
    # bounds sightseeing only — meals must not eat into the day's POI budget.
    attraction_indices = [
        i for i in range(1, n) if not pois[i].id.startswith("__meal_d")
    ]
    for d in range(days_count):
        model.Add(sum(V[(d, i)] for i in attraction_indices) <= MAX_POI_PER_DAY)

    # Preferences
    interests = set(constraints.interests or [])
    prefs = []
    for p in pois:
        s = p.score
        if interests:
            s += len(set(p.tags) & interests) * 0.1
        prefs.append(min(s, 1.0))
    pref_score = sum(V[(d, i)] * int(prefs[i] * 100) for d in range(days_count) for i in range(1, n))

    # Peak score
    peak_score = sum(V[(d, i)] * (100 if pois[i].is_peak else 0) for d in range(days_count) for i in range(1, n))

    # Walk diff
    walk_diffs = []
    for d1 in range(days_count):
        for d2 in range(days_count):
            if d1 < d2:
                diff = model.NewIntVar(0, max(walk_limits) * 2, f"wdiff_{d1}_{d2}")
                model.Add(diff >= DW[d1] - DW[d2])
                model.Add(diff >= DW[d2] - DW[d1])
                walk_diffs.append(diff)
    total_walk_diff = sum(walk_diffs)

    # Epsilon-constraint: primary objective = minimize total travel time
    total_travel_time = sum(
        X[(d, i, j)] * dist[i][j]
        for d in range(days_count)
        for i in route_nodes
        for j in route_nodes
        if i != j
    )

    eps = constraints.epsilon_config
    model.Add(total_walk_diff <= int(eps.max_walk_diff))
    model.Add(max_day_cost <= int(eps.max_budget_mad))
    model.Add(pref_score >= int(eps.min_preference))
    model.Add(peak_score <= int(eps.max_peak_score))

    # Cost balancing only makes sense under a real budget. With an unlimited
    # budget, penalising spend (even as spread) would needlessly drop valuable
    # but pricey attractions (e.g. an aquarium), so disable it.
    spread_weight = 2 if total_budget != float("inf") else 0

    # Soft penalty for too few POIs per day
    total_visits = sum(V[(d, i)] for d in range(days_count) for i in range(1, n))
    # Travel weight is deliberately light. Geographic clustering should come from
    # the per-day TIME budget (constraint 5) + realistic commute times: packing
    # far-apart POIs burns the day budget so fewer fit, which naturally groups
    # nearby POIs per day. A heavy travel penalty (was -10/min) instead made any
    # POI needing a >~15min hop look like a net loss, leaving days near-empty.
    # Pull every visit as early as its window/order allows. Without this the
    # arrival times are free slack, so the solver often leaves whole mornings
    # empty and starts a day at the 11:30 lunch. Unvisited nodes settle at 0 and
    # do not affect the term. Other weights are scaled up 10x so this stays a
    # tie-breaker that compresses the day without dropping visits or reordering
    # preferences.
    total_arrival = sum(A[(d, i)] for d in range(days_count) for i in range(1, n))

    # Redundancy penalty: POIs sharing a redundant tag (观景台 etc.) are
    # functionally interchangeable, so selecting more than one across the whole
    # trip wastes a large duplicate ticket (东方明珠/环球/上海中心). Penalise the
    # overflow (group_visits - 1) hard enough to beat one POI's marginal value
    # (pref ~2000 + visit 500), so the solver keeps the single best of each group.
    # must-visit POIs are exempt — if the user insists on two towers, honour it.
    _must_set = set(must_indices)
    redundant_tags = {t for t in (constraints.redundant_tags or []) if t}
    redundancy_penalty_terms = []
    for tag in redundant_tags:
        group = [
            i
            for i in range(1, n)
            if not pois[i].id.startswith("__meal_d")
            and tag in set(pois[i].tags or [])
            and i not in _must_set
        ]
        if len(group) <= 1:
            continue
        group_visits = sum(V[(d, i)] for d in range(days_count) for i in group)
        overflow = model.NewIntVar(0, len(group), f"redundant_overflow_{tag}")
        model.Add(overflow >= group_visits - 1)
        redundancy_penalty_terms.append(overflow)
    redundancy_penalty = sum(redundancy_penalty_terms) if redundancy_penalty_terms else 0

    # Compactness: discourage one day from sprawling across geographically distant
    # attractions. The travel term alone under-penalises this because AMap driving
    # time across the river is cheap, so a day could span far districts at near-equal
    # cost (the "动线散" complaint). A soft penalty on same-day pairs whose straight-
    # line distance exceeds COMPACT_KM nudges each day to stay within one area. The
    # weight is far below the visit reward, so it only *rearranges* days — it never
    # drops a visit. Only far pairs get a var (suburbs are already hard-excluded), so
    # the model barely grows. Shapes the incumbent even when the solve stays FEASIBLE.
    COMPACT_KM = 8.0
    compact_terms = []
    _real_attr_c = [
        i for i in range(1, n)
        if not pois[i].id.startswith("__meal_d") and (pois[i].lat or pois[i].lng)
    ]
    for _a in range(len(_real_attr_c)):
        for _b in range(_a + 1, len(_real_attr_c)):
            i, j = _real_attr_c[_a], _real_attr_c[_b]
            if _haversine_km(pois[i], pois[j]) <= COMPACT_KM:
                continue
            for d in range(days_count):
                sd = model.NewBoolVar(f"compact_{d}_{i}_{j}")
                model.Add(sd >= V[(d, i)] + V[(d, j)] - 1)
                compact_terms.append(sd)
    compactness_penalty = sum(compact_terms) if compact_terms else 0

    # Weight hierarchy (critical):
    #   • total_visits = 3000 dominates travel/spread so a reachable POI is NEVER
    #     dropped just because it needs a moderate hop. With travel at -80/min, a
    #     POI 30min from its day-mates costs 2400 < 3000 → kept; the solver instead
    #     minimises travel by *grouping* it with near POIs (or giving a far one its
    #     own 0-commute day), which is exactly the clustering we want.
    #   • redundancy = 6000 > 3000 + pref so a 2nd interchangeable tower is still
    #     dropped (3-towers fix intact).
    #   • spread = -1/¥, travel = -80/min, peak/arrival: tie-breakers for ARRANGING
    #     the chosen set, never strong enough to drop a visit.
    model.Maximize(
        3000 * total_visits
        - 6000 * redundancy_penalty
        - 400 * compactness_penalty
        - 80 * total_travel_time
        - 1 * spread_weight * day_cost_spread
        + 20 * pref_score
        - 10 * peak_score
        - 1 * total_arrival
    )

    # Warm start: seed CP-SAT with a geographic day-clustering. On realistic
    # instances (10 POIs + meal nodes + 5 days ≈ 2000+ arcs) single-worker search
    # within the few-second cap cannot, from scratch, find the compact grouping the
    # travel objective wants — it returns a feasible but geographically scattered
    # incumbent (陆家嘴 水族馆/环球金融 split across days, each a separate river
    # crossing). Hinting a proximity clustering lets CP-SAT *refine* a compact plan
    # instead of *discovering* one. The hint is advisory: a stale/imperfect hint
    # cannot make the model infeasible, so it is wrapped defensively.
    try:
        clusters = _geo_day_clusters(pois, days_count, dist, MAX_POI_PER_DAY)
        for d_i, idxs in clusters.items():
            for idx in idxs:
                model.AddHint(V[(d_i, idx)], 1)
    except Exception as exc:  # pragma: no cover - hint is best-effort
        logger.debug("Geo warm-start hint skipped: %s", exc)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params["max_time_in_seconds"]
    solver.parameters.num_search_workers = params["num_search_workers"]

    # Convergence early-stop: don't bail before 45% of the budget (quality floor),
    # then stop once a new incumbent improves the objective by <0.3% — the plan has
    # settled and the rest would just prove optimality. Trims the ~18s tail on large
    # instances; small ones finish well before the floor and are unaffected.
    _tl = params["max_time_in_seconds"]
    callback = TimeoutCallback(
        _tl,
        min_seconds=params.get("min_time_in_seconds", _tl * 0.45),
        rel_improve_stop=params.get("rel_improve_stop", 0.003),
    )
    status = solver.Solve(model, callback)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning(
            "CP-SAT not solved (status=%s, wall=%.1fs), falling back to greedy",
            solver.StatusName(status),
            solver.WallTime(),
        )
        try:
            import json as _json
            import time as _time

            dump = {
                "status": solver.StatusName(status),
                "wall": solver.WallTime(),
                "n_pois": len(pois),
                "constraints": constraints.model_dump(),
                "pois": [p.model_dump() for p in pois],
            }
            path = f"/tmp/vrp_infeasible_{int(_time.time())}.json"
            with open(path, "w") as fh:
                _json.dump(dump, fh, ensure_ascii=False, default=str)
            logger.warning("Dumped infeasible request to %s", path)
        except Exception as _exc:  # pragma: no cover - diagnostic only
            logger.warning("Failed to dump infeasible request: %s", _exc)
        return _greedy_solve(pois, constraints, dist, tc, walk_limits)

    return _extract_schedule(pois, constraints, dist, tc, V, A, X, DC, DW, solver, status)


def _greedy_solve(
    pois: list[POIInput],
    constraints: ConstraintsInput,
    dist: list[list[int]],
    tc: list[list[float]],
    walk_limits: list[int],
) -> list[DayPlanOutput]:
    """Greedy heuristic fallback."""
    n = len(pois)
    days_count = constraints.travel_days
    day_start = constraints.day_start_min or DAY_START_MIN
    day_end = constraints.day_end_min or DAY_END_MIN
    day_available = day_end - day_start
    total_budget = constraints.total_budget or float("inf")
    day_budget = total_budget / days_count if total_budget != float("inf") else float("inf")
    max_transit = constraints.max_transit_minutes
    rest_day = constraints.rest_day
    must_see_set = set(constraints.must_visit or [])
    for idx in range(1, n):
        if pois[idx].name in must_see_set:
            must_see_set.add(pois[idx].id)
        if pois[idx].id in must_see_set:
            must_see_set.add(pois[idx].name)

    # Separate meal dummy nodes from real attractions. Meals are bound to a
    # specific day and must land inside their dining window, so they are placed
    # explicitly rather than competing with attractions in the greedy loop.
    meals_by_day: dict[int, list[int]] = {}
    attraction_pool: set[int] = set()
    for i in range(1, n):
        m = re.match(r"__meal_d(\d+)_", pois[i].id)
        if m:
            meals_by_day.setdefault(int(m.group(1)) - 1, []).append(i)
        else:
            attraction_pool.add(i)

    def _to_min(hhmm: str | None, default: int) -> int:
        if not hhmm:
            return default
        try:
            h, mm = hhmm.split(":")
            return int(h) * 60 + int(mm)
        except (ValueError, AttributeError):
            return default

    remaining = attraction_pool  # exclude hotel and meals
    days: list[DayPlanOutput] = []

    # Mirror the CP-SAT fixes on the greedy path so short trips (the greedy-auto
    # branch: <=15 POIs, <=3 days) get the same quality, not the old behaviour.
    _buf = max(0, constraints.transition_buffer_min or 0)
    _peak_pad = max(0, constraints.peak_queue_pad_min or 0)
    _remote_min = max(0, constraints.remote_pair_min or 0)
    _redundant_tags = {t for t in (constraints.redundant_tags or []) if t}
    _suburban = _suburban_indices(pois, constraints)
    _suburb_nearby = max(0, constraints.suburb_nearby_min or 0)
    _day_weekdays = constraints.day_weekdays or []

    def _g_arc_extra(i: int, j: int) -> int:
        if i == 0 or j == 0:
            return 0
        return _buf + (_peak_pad if pois[j].is_peak else 0)

    def _g_redundant_tags(i: int) -> set[str]:
        return _redundant_tags & set(pois[i].tags or [])

    # Redundant tags already "spent" trip-wide (观景台 etc.) so a second
    # interchangeable landmark is not picked. must-visit POIs bypass this.
    used_redundant: set[str] = set()

    for d in range(days_count):
        activities: list[ActivityOutput] = []
        day_poi_indices: list[int] = []  # real attractions placed today (remote check)
        current_idx = 0  # start from hotel
        current_time = day_start
        day_walk = 0
        day_cost = 0.0
        day_tc = 0.0
        visits_today = 0

        # Meals bound to this day, ordered by window start.
        day_meals = sorted(
            meals_by_day.get(d, []),
            key=lambda mi: _to_min(pois[mi].open_time, 0),
        )

        def _place_due_meals(until_min: int) -> None:
            """Insert any meal whose window has opened by ``until_min``."""
            nonlocal current_time
            while day_meals:
                mi = day_meals[0]
                win_start = _to_min(pois[mi].open_time, 0)
                win_end = _to_min(pois[mi].close_time, day_end)
                if win_start > until_min:
                    break
                day_meals.pop(0)
                m_start = max(current_time, win_start)
                # Skip the meal only if it truly cannot fit before its window
                # closes or the day ends.
                if m_start > win_end or m_start + pois[mi].duration_minutes > day_end:
                    continue
                m_end = m_start + pois[mi].duration_minutes
                activities.append(
                    ActivityOutput(
                        poi_id=pois[mi].id,
                        poi_name=pois[mi].name,
                        category=pois[mi].category,
                        start_time=_fmt_time(m_start),
                        end_time=_fmt_time(m_end),
                        duration_min=pois[mi].duration_minutes,
                        ticket_price=pois[mi].ticket_price,
                        transport_cost=0.0,
                        lat=pois[mi].lat,
                        lng=pois[mi].lng,
                        tags=pois[mi].tags,
                    )
                )
                current_time = m_end

        def _place_meal_if_missed(proj_end: int) -> bool:
            """Eat now (waiting for the window if needed) when running the next
            attraction until ``proj_end`` would close the meal window first."""
            if not day_meals:
                return False
            mi = day_meals[0]
            win_start = _to_min(pois[mi].open_time, 0)
            win_end = _to_min(pois[mi].close_time, day_end)
            # Window opens only after the attraction ends, or stays open past it:
            # in both cases we can safely eat later.
            if win_start > proj_end or win_end >= proj_end:
                return False
            m_start = max(current_time, win_start)
            if m_start > win_end or m_start + pois[mi].duration_minutes > day_end:
                return False
            _place_due_meals(win_start)
            return True

        while remaining and visits_today < MAX_POI_PER_DAY:
            # Drop in any meal whose window has already opened before we add the
            # next attraction.
            _place_due_meals(current_time)
            best = None
            best_score = -1.0
            for i in remaining:
                poi = pois[i]
                travel = dist[current_idx][i]
                if travel > max_transit:
                    continue
                is_must = poi.id in must_see_set or poi.name in must_see_set
                # #1 redundancy: skip a second interchangeable landmark (观景台 …)
                # once its tag is spent, unless the user pinned it.
                if not is_must and _g_redundant_tags(i) & used_redundant:
                    continue
                # #3 remote isolation: do not add a POI that is a long hop from any
                # attraction already scheduled today.
                if _remote_min and any(
                    dist[i][j] > _remote_min for j in day_poi_indices
                ):
                    continue
                # Suburb isolation: a far-suburb attraction shares its day only with
                # genuinely adjacent stops (mirrors CP-SAT constraint 2f).
                if _suburban and any(
                    (i in _suburban or j in _suburban) and dist[i][j] > _suburb_nearby
                    for j in day_poi_indices
                ):
                    continue
                # Weekly closing day (周一闭馆): skip a venue closed on this day's
                # weekday (mirrors CP-SAT constraint 2g).
                if (
                    _day_weekdays
                    and poi.closed_weekdays
                    and d < len(_day_weekdays)
                    and _day_weekdays[d] in poi.closed_weekdays
                ):
                    continue
                duration = poi.duration_minutes
                extra = _g_arc_extra(current_idx, i)
                arr = current_time + travel + extra
                # #5 honour opening hours: an evening-only POI (夜景, open>=16:00)
                # cannot start before it opens — wait, then check it still fits.
                open_min = _to_min(poi.open_time, day_start)
                close_min = _to_min(poi.close_time, day_end)
                start = max(arr, open_min)
                if start + duration > close_min or start + duration > day_end:
                    continue
                # #4 day-budget check counts commute + slack (idle wait is not
                # ground time, so it is excluded from the duration budget).
                total_day_time = (
                    sum(a.duration_min for a in activities) + duration
                    + travel + extra
                    + rest_day
                )
                if total_day_time > day_available:
                    continue
                if day_cost + poi.ticket_price > day_budget:
                    continue
                if day_walk + poi.walk_intensity > walk_limits[d]:
                    continue
                # Deprioritise a candidate that would idle a long time waiting to
                # open, so evening POIs are naturally picked later in the day.
                wait_penalty = (start - arr) / 600.0
                score = poi.score + (0.5 if is_must else 0.0) - wait_penalty
                if score > best_score:
                    best_score = score
                    best = i

            if best is None:
                break

            # If visiting the best attraction now would run past a meal window
            # and drop the meal, eat first (waiting for the window) and re-select.
            _b_extra = _g_arc_extra(current_idx, best)
            _b_arr = current_time + dist[current_idx][best] + _b_extra
            _b_start = max(_b_arr, _to_min(pois[best].open_time, day_start))
            proj_end = _b_start + pois[best].duration_minutes
            if _place_meal_if_missed(proj_end):
                continue

            i = best
            travel = dist[current_idx][i]
            extra = _g_arc_extra(current_idx, i)
            arr = current_time + travel + extra
            start = max(arr, _to_min(pois[i].open_time, day_start))
            end = start + pois[i].duration_minutes
            leg_tc = tc[current_idx][i]

            activities.append(
                ActivityOutput(
                    poi_id=pois[i].id,
                    poi_name=pois[i].name,
                    category=pois[i].category,
                    start_time=_fmt_time(start),
                    end_time=_fmt_time(end),
                    duration_min=pois[i].duration_minutes,
                    ticket_price=pois[i].ticket_price,
                    transport_cost=leg_tc,
                    lat=pois[i].lat,
                    lng=pois[i].lng,
                    tags=pois[i].tags,
                )
            )

            current_time = end
            current_idx = i
            day_walk += pois[i].walk_intensity
            day_cost += pois[i].ticket_price
            day_tc += leg_tc
            day_poi_indices.append(i)
            used_redundant |= _g_redundant_tags(i)
            remaining.discard(i)
            visits_today += 1

        # Place any meals still pending (e.g. dinner after the last attraction).
        _place_due_meals(day_end)
        # Keep the day timeline chronological after meal insertion.
        activities.sort(key=lambda a: a.start_time)

        days.append(
            DayPlanOutput(
                day_number=d + 1,
                activities=activities,
                # Match the CP-SAT daily cost (DC): tickets + per-day food + city
                # transit, so both solver paths report a consistent total_cost.
                total_cost=day_cost + day_tc + constraints.food_day,
                transport_cost=day_tc,
                walk_intensity=day_walk,
            )
        )

    return days


def _extract_schedule(
    pois: list[POIInput],
    constraints: ConstraintsInput,
    dist: list[list[int]],
    tc: list[list[float]],
    V,
    A,
    X,
    DC,
    DW,
    solver,
    status,
) -> list[DayPlanOutput]:
    """Build DayPlanOutput from CP-SAT solution."""
    n = len(pois)
    days_count = constraints.travel_days

    day_assignments: list[list[int]] = [[] for _ in range(days_count)]
    for d in range(days_count):
        for i in range(1, n):
            if solver.Value(V[(d, i)]) == 1:
                day_assignments[d].append(i)

    days: list[DayPlanOutput] = []
    for d, indices in enumerate(day_assignments):
        if not indices:
            days.append(DayPlanOutput(day_number=d + 1, activities=[]))
            continue
        indices.sort(key=lambda i: solver.Value(A[(d, i)]))
        activities: list[ActivityOutput] = []
        prev = 0  # last *routed* node (meals carry no coords / commute)
        for i in indices:
            arr = solver.Value(A[(d, i)])
            start = arr
            end = start + pois[i].duration_minutes
            is_meal = pois[i].id.startswith("__meal_d")
            activities.append(
                ActivityOutput(
                    poi_id=pois[i].id,
                    poi_name=pois[i].name,
                    category=pois[i].category,
                    start_time=_fmt_time(start),
                    end_time=_fmt_time(end),
                    duration_min=pois[i].duration_minutes,
                    ticket_price=pois[i].ticket_price,
                    transport_cost=0.0 if is_meal else tc[prev][i],
                    lat=pois[i].lat,
                    lng=pois[i].lng,
                    tags=pois[i].tags,
                )
            )
            if not is_meal:
                prev = i

        day_transport_cost = sum(a.transport_cost for a in activities)
        days.append(
            DayPlanOutput(
                day_number=d + 1,
                activities=activities,
                total_cost=float(solver.Value(DC[d])) if d in DC else 0.0,
                transport_cost=day_transport_cost,
                walk_intensity=solver.Value(DW[d]) if d in DW else 0,
            )
        )

    logger.info(
        "VRP solved: status=%s, days=%d, pois=%d, time=%.2fs",
        solver.StatusName(status),
        days_count,
        sum(len(day.activities) for day in days),
        solver.WallTime(),
    )
    return days
