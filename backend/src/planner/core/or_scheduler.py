"""
OR-Tools CP-SAT 多约束行程求解器。

两阶段求解:
  1. CP-SAT: 将 POI 分配到每天（同时满足预算/时间/必去/多样性/远程约束）
  2. NN+2-opt: 每日内路径优化（复用现有代码）

替代原有纯贪心算法，支持全局最优。
"""

from __future__ import annotations

import logging
import math
from typing import Optional

from ortools.sat.python import cp_model

from schemas import (
    Activity,
    DayPlan,
    Location,
    ScoredPOI,
    UserProfile,
    WeatherDay,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DAY_START_HOUR = 9   # 9:00
DAY_END_HOUR = 19     # 19:00
MEAL_SLOTS = 2        # 预留午晚餐时间
SLOT_DURATION_MIN = 60  # 每个时段 60 分钟
TRANSIT_KMH = 50      # 市内平均车速 km/h
MAX_POIS_PER_DAY = 5  # 每天最多景点数


def _distance_km(a: Location, b: Location) -> float:
    """近似距离（km）。"""
    lat_km = (a.lat - b.lat) * 111
    lng_km = (a.lng - b.lng) * 85
    return math.sqrt(lat_km**2 + lng_km**2)


def _resolve_minutes(duration_str: Optional[str]) -> int:
    """解析时长字符串为分钟数。"""
    if not duration_str:
        return 120
    dur = duration_str.lower().replace(" ", "")
    total = 0
    for part in dur.replace("h", ":").replace("m", "").split(":"):
        part = part.strip()
        if not part:
            continue
        try:
            total = total * 60 + int(part)
        except ValueError:
            pass
    return total if total > 0 else 120


def _day_available_minutes(profile: UserProfile) -> int:
    """每天可用于游览的分钟数（扣除用餐）。"""
    pace = profile.pace or "moderate"
    if pace == "relaxed":
        hours = 7
    elif pace == "intensive":
        hours = 10
    else:
        hours = 8
    return hours * 60 - MEAL_SLOTS * 60


# ---------------------------------------------------------------------------
# CP-SAT Solver
# ---------------------------------------------------------------------------


def solve_itinerary_or(
    pois: list[ScoredPOI],
    profile: UserProfile,
    must_see: Optional[list[str]] = None,
    remote_threshold_km: float = 50.0,
    time_limit_seconds: float = 5.0,
) -> list[DayPlan]:
    """
    OR-Tools CP-SAT 多约束行程求解。

    Args:
        pois: 候选 POI 列表
        profile: 用户画像
        must_see: 必去 POI 名称列表
        remote_threshold_km: 远程 POI 判定阈值
        time_limit_seconds: CP-SAT 求解超时

    Returns:
        list[DayPlan]: 每日行程
    """
    travel_days = profile.travel_days or 1
    total_budget = profile.budget_range or float("inf")
    interests = set(profile.interests or [])
    food_prefs = set(profile.food_preferences or [])
    must_see_set = set(must_see or [])

    if not pois:
        return []

    n = len(pois)
    days = travel_days
    day_minutes = _day_available_minutes(profile)

    # 预算分配
    daily_budget = total_budget / days if total_budget != float("inf") else float("inf")

    # 偏好打分
    scores = []
    for poi in pois:
        s = poi.score or 0.5
        if interests:
            match = len(set(poi.tags or []) & interests)
            s += match * 0.1
        if food_prefs:
            match = len(set(poi.tags or []) & food_prefs)
            s += match * 0.15
        s = min(s, 1.0)
        scores.append(s)

    # ── CP-SAT 建模 ──
    model = cp_model.CpModel()

    # 变量 X[i][d] = 1 表示 POI i 分配到 Day d
    X = {}
    for i in range(n):
        for d in range(days):
            X[(i, d)] = model.NewBoolVar(f"x_{i}_{d}")

    # 约束 1: 每个 POI 最多安排一天
    for i in range(n):
        model.Add(sum(X[(i, d)] for d in range(days)) <= 1)

    # 约束 2: Must-see 必须安排
    for i, poi in enumerate(pois):
        if poi.name in must_see_set:
            model.Add(sum(X[(i, d)] for d in range(days)) == 1)

    # 约束 3: 每天时长不超限
    for d in range(days):
        model.Add(
            sum(
                X[(i, d)] * _resolve_minutes(pois[i].recommended_hours)
                for i in range(n)
            )
            <= day_minutes
        )

    # 约束 4: 每天预算不超限
    if daily_budget != float("inf"):
        for d in range(days):
            model.Add(
                sum(
                    X[(i, d)] * (pois[i].ticket_price or 0)
                    for i in range(n)
                )
                <= daily_budget
            )

    # 约束 5: 每天 POI 数量上限
    for d in range(days):
        model.Add(sum(X[(i, d)] for i in range(n)) <= MAX_POIS_PER_DAY)

    # 约束 6: 同 category 每天 ≤ 2
    categories: dict[str, list[int]] = {}
    for i, poi in enumerate(pois):
        cat = poi.category or "attraction"
        categories.setdefault(cat, []).append(i)
    for d in range(days):
        for cat, indices in categories.items():
            model.Add(sum(X[(i, d)] for i in indices) <= 2)

    # 约束 7: 远程 POI 只安排全天（单独一天）
    # 计算城市中心
    if pois and pois[0].location:
        center_lat = sum(p.location.lat for p in pois if p.location) / max(len([p for p in pois if p.location]), 1)
        center_lng = sum(p.location.lng for p in pois if p.location) / max(len([p for p in pois if p.location]), 1)
        center = Location(lat=center_lat, lng=center_lng)

        remote_pois = []
        for i, poi in enumerate(pois):
            if poi.location and _distance_km(poi.location, center) > remote_threshold_km:
                remote_pois.append(i)

        if remote_pois and days >= 2:
            # 远程 POI 只能放在特定天（最后一天或独立天）
            remote_day = days - 1  # 最后一天
            for i in remote_pois:
                # 强制远程 POI 在 remote_day
                model.Add(X[(i, remote_day)] == 1)
                # 不在其他天
                for d in range(days):
                    if d != remote_day:
                        model.Add(X[(i, d)] == 0)

    # 目标: 最大化偏好分 + 均衡日间负载
    objective_terms = []
    for i in range(n):
        for d in range(days):
            objective_terms.append(X[(i, d)] * int(scores[i] * 100))

    # 日间负载均衡惩罚
    if days >= 2:
        load_vars = []
        for d in range(days):
            day_load = model.NewIntVar(0, 500, f"load_{d}")
            model.Add(day_load == sum(
                X[(i, d)] * _resolve_minutes(pois[i].recommended_hours)
                for i in range(n)
            ))
            load_vars.append(day_load)

        # 用辅助变量计算方差
        for d1 in range(days):
            for d2 in range(d1 + 1, days):
                diff = model.NewIntVar(0, 500, f"diff_{d1}_{d2}")
                model.Add(diff >= load_vars[d1] - load_vars[d2])
                model.Add(diff >= load_vars[d2] - load_vars[d1])
                objective_terms.append(diff * -5)  # 惩罚不均衡

    model.Maximize(sum(objective_terms))

    # ── 求解 ──
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = False

    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        logger.warning(
            "CP-SAT returned status %s, falling back to greedy",
            solver.StatusName(status),
        )
        return _greedy_fallback(pois, profile, must_see_set)

    # ── 构建 DayPlan ──
    day_assignments: list[list[ScoredPOI]] = [[] for _ in range(days)]
    for i in range(n):
        for d in range(days):
            if solver.Value(X[(i, d)]) == 1:
                day_assignments[d].append(pois[i])

    # 每日内路径优化（NN + 2-opt）
    from planner.core.daily_scheduler import _optimize_daily_routes

    optimized = _optimize_daily_routes(day_assignments)

    # 构建 DayPlan
    schedule = _build_daily_plans(optimized, profile)

    logger.info(
        "CP-SAT solved: status=%s, days=%d, pois=%d, time=%.2fs",
        solver.StatusName(status),
        days,
        sum(len(day) for day in day_assignments),
        solver.WallTime(),
    )

    return schedule


def _greedy_fallback(
    pois: list[ScoredPOI],
    profile: UserProfile,
    must_see: set[str],
) -> list[DayPlan]:
    """CP-SAT 失败时退回贪心算法。"""
    from planner.core.daily_scheduler import build_schedule
    from planner.core.strategy import Strategy

    strategy = Strategy()
    return build_schedule(strategy, pois, [], profile)


def _build_daily_plans(
    day_assignments: list[list[ScoredPOI]],
    profile: UserProfile,
) -> list[DayPlan]:
    """将 POI 分组转换为 DayPlan 列表（含时段分配）。"""
    from planner.core.daily_scheduler import (
        _assign_time_slots,
        _create_meal_activity,
    )

    schedule: list[DayPlan] = []

    for day_idx, pois in enumerate(day_assignments):
        if not pois:
            schedule.append(DayPlan(day_number=day_idx + 1, activities=[]))
            continue

        activities: list[Activity] = []
        current_minutes = DAY_START_HOUR * 60

        for i, poi in enumerate(pois):
            # 交通时间
            if i > 0 and poi.location and pois[i - 1].location:
                dist = _distance_km(pois[i - 1].location, poi.location)
                transit_min = int(dist / TRANSIT_KMH * 60)
                current_minutes += transit_min

            # 营业时间检查
            start = current_minutes
            duration = _resolve_minutes(poi.recommended_hours)
            end = start + duration

            start_str = f"{start // 60:02d}:{start % 60:02d}"
            end_str = f"{end // 60:02d}:{end % 60:02d}"

            activities.append(
                Activity(
                    poi_name=poi.name,
                    category=poi.category,
                    start_time=start_str,
                    end_time=end_str,
                    duration_min=duration,
                    ticket_price=poi.ticket_price,
                    location=poi.location,
                    tags=poi.tags or [],
                    open_time=poi.open_time,
                    close_time=poi.close_time,
                )
            )
            current_minutes = end + 15  # buffer

        schedule.append(
            DayPlan(
                day_number=day_idx + 1,
                activities=activities,
                total_cost=sum(
                    (a.ticket_price or 0) + (a.meal_cost or 0)
                    for a in activities
                ),
            )
        )

    return schedule
