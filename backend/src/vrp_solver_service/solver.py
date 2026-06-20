"""Standalone VRP-TW solver aligned with v4.0 production-grade design."""

from __future__ import annotations

import logging
import math
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
            dist, tc = self._transport_selector.build_matrices(pois, constraints)
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
        pois = self._inject_hotel(pois)

        if request.dist_matrix is not None and request.tc_matrix is not None:
            dist = request.dist_matrix
            tc = request.tc_matrix
        else:
            dist, tc = self._transport_selector.build_matrices(pois, constraints)

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

    M_time = max(close_times)
    M_travel = max(max(row) for row in dist) + max(durations)

    for d in range(days_count):
        for i in range(n):
            V[(d, i)] = model.NewBoolVar(f"V_{d}_{i}")
            A[(d, i)] = model.NewIntVar(0, day_end, f"A_{d}_{i}")
            for j in range(n):
                if i != j:
                    X[(d, i, j)] = model.NewBoolVar(f"X_{d}_{i}_{j}")
        DC[d] = model.NewIntVar(0, int(day_budget * 2) if day_budget != float("inf") else 100000, f"DC_{d}")
        DW[d] = model.NewIntVar(0, max(walk_limits) * 2, f"DW_{d}")

    # Constraint 1: AddCircuit for each day (hotel node + real POIs)
    for d in range(days_count):
        arcs = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                arcs.append((i, j, X[(d, i, j)]))
            self_loop = model.NewBoolVar(f"sl_{d}_{i}")
            arcs.append((i, i, self_loop))
        model.AddCircuit(arcs)

    # Constraint 2: visit-edge linkage (excluding hotel)
    for d in range(days_count):
        for i in range(1, n):
            model.Add(sum(X[(d, j, i)] for j in range(n) if j != i) == V[(d, i)])
            model.Add(sum(X[(d, i, j)] for j in range(n) if j != i) == V[(d, i)])

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

    # Constraint 3: conditional time windows
    for d in range(days_count):
        for i in range(1, n):
            model.Add(A[(d, i)] >= open_times[i] - M_time * (1 - V[(d, i)]))
            model.Add(A[(d, i)] + durations[i] <= close_times[i] + M_time * (1 - V[(d, i)]))
            model.Add(A[(d, i)] <= M_time * V[(d, i)])

    # Constraint 4: exact commute propagation (bidirectional)
    for d in range(days_count):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                travel = dist[i][j]
                if travel > max_transit:
                    model.Add(X[(d, i, j)] == 0)
                else:
                    model.Add(A[(d, j)] >= A[(d, i)] + durations[i] + travel - M_travel * (1 - X[(d, i, j)]))
                    model.Add(A[(d, j)] <= A[(d, i)] + durations[i] + travel + M_travel * (1 - X[(d, i, j)]))

    # Constraint 5: daily duration excluding return-to-hotel commute
    for d in range(days_count):
        model.Add(
            sum(V[(d, i)] * durations[i] for i in range(1, n))
            + sum(
                X[(d, i, j)] * dist[i][j]
                for i in range(1, n)
                for j in range(1, n)
                if i != j
            )
            + rest_day
            <= day_available
        )

    # Constraint 6: daily walking limit
    for d in range(days_count):
        model.Add(DW[d] == sum(V[(d, i)] * walks[i] for i in range(1, n)))
        model.Add(DW[d] <= walk_limits[d])

    # Constraint 7: budget (daily + total) with MAD
    total_tc_expr = sum(
        X[(d, i, j)] * int(tc[i][j])
        for d in range(days_count)
        for i in range(n)
        for j in range(n)
        if i != j
    )
    for d in range(days_count):
        day_tc = sum(
            X[(d, i, j)] * int(tc[i][j])
            for i in range(n)
            for j in range(n)
            if i != j
        )
        model.Add(DC[d] == int(food_day) + sum(V[(d, i)] * costs[i] for i in range(1, n)) + day_tc)
        if day_budget != float("inf"):
            model.Add(DC[d] <= int(day_budget))

    if total_budget != float("inf"):
        model.Add(sum(DC[d] for d in range(days_count)) + total_tc_expr <= int(total_budget))

    # Minimise the maximum daily cost (linear, CP-SAT native).
    # Replaces the previous MAD computation which used // on SumArray.
    max_day_cost = model.NewIntVar(0, 100000, "max_day_cost")
    for d in range(days_count):
        model.Add(max_day_cost >= DC[d])

    # Constraint 8: POI count hard upper bound, soft lower bound as penalty
    for d in range(days_count):
        model.Add(sum(V[(d, i)] for i in range(1, n)) <= MAX_POI_PER_DAY)

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
        for i in range(n)
        for j in range(n)
        if i != j
    )

    eps = constraints.epsilon_config
    model.Add(total_walk_diff <= int(eps.max_walk_diff))
    model.Add(max_day_cost <= int(eps.max_budget_mad))
    model.Add(pref_score >= int(eps.min_preference))
    model.Add(peak_score <= int(eps.max_peak_score))

    # Soft penalty for too few POIs per day
    total_visits = sum(V[(d, i)] for d in range(days_count) for i in range(1, n))
    model.Maximize(
        -10 * total_travel_time
        - 5 * max_day_cost
        + 2 * pref_score
        - 1 * peak_score
        + 50 * total_visits
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = params["max_time_in_seconds"]
    solver.parameters.num_search_workers = params["num_search_workers"]

    callback = TimeoutCallback(params["max_time_in_seconds"])
    status = solver.Solve(model, callback)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning("CP-SAT infeasible, falling back to greedy")
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

    remaining = set(range(1, n))  # exclude hotel
    days: list[DayPlanOutput] = []

    for d in range(days_count):
        activities: list[ActivityOutput] = []
        current_idx = 0  # start from hotel
        current_time = day_start
        day_walk = 0
        day_cost = 0.0
        day_tc = 0.0
        visits_today = 0

        while remaining and visits_today < MAX_POI_PER_DAY:
            best = None
            best_score = -1.0
            for i in remaining:
                poi = pois[i]
                travel = dist[current_idx][i]
                if travel > max_transit:
                    continue
                duration = poi.duration_minutes
                if current_time + travel + duration > day_end:
                    continue
                # duration + travel + rest check
                total_day_time = (
                    sum(a.duration_min for a in activities) + duration
                    + sum(dist[current_idx][i] if current_idx != 0 else 0 for _ in [i])
                    + rest_day
                )
                if total_day_time > day_available:
                    continue
                if day_cost + poi.ticket_price > day_budget:
                    continue
                if day_walk + poi.walk_intensity > walk_limits[d]:
                    continue
                is_must = poi.id in must_see_set or poi.name in must_see_set
                score = poi.score + (0.5 if is_must else 0.0)
                if score > best_score:
                    best_score = score
                    best = i

            if best is None:
                break

            i = best
            travel = dist[current_idx][i]
            current_time += travel
            start = current_time
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
            remaining.discard(i)
            visits_today += 1

        days.append(
            DayPlanOutput(
                day_number=d + 1,
                activities=activities,
                total_cost=day_cost,
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
        prev = 0
        for i in indices:
            arr = solver.Value(A[(d, i)])
            start = arr
            end = start + pois[i].duration_minutes
            activities.append(
                ActivityOutput(
                    poi_id=pois[i].id,
                    poi_name=pois[i].name,
                    category=pois[i].category,
                    start_time=_fmt_time(start),
                    end_time=_fmt_time(end),
                    duration_min=pois[i].duration_minutes,
                    ticket_price=pois[i].ticket_price,
                    transport_cost=tc[prev][i],
                    lat=pois[i].lat,
                    lng=pois[i].lng,
                    tags=pois[i].tags,
                )
            )
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
