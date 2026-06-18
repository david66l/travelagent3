"""
规划引擎增强 — 多约束权重 + 人群模板 + 可行性校验 + 错峰调度。

对应蓝图 5.5 - 5.8
"""

from __future__ import annotations

from schemas import DayPlan, ScoredPOI, UserProfile


# ---------------------------------------------------------------------------
# 5.5 多约束权重自定义
# ---------------------------------------------------------------------------


class OptimizationWeights:
    """用户可配权重 —— 省钱/省力/打卡/美食优先。"""

    DEFAULT = {
        "cost_save": 0.25,
        "effort_save": 0.25,
        "poi_count": 0.25,
        "food_quality": 0.25,
    }

    @classmethod
    def apply(cls, profile: UserProfile) -> dict[str, float]:
        w = dict(cls.DEFAULT)
        if profile.has_elderly or profile.has_children:
            w["effort_save"] = 0.5
            w["poi_count"] = 0.1
        if profile.interests and "美食" in profile.interests:
            w["food_quality"] = 0.4
            w["cost_save"] = 0.1
        if profile.pace == "intensive":
            w["poi_count"] = 0.4
            w["effort_save"] = 0.1
        return w


# ---------------------------------------------------------------------------
# 5.6 人群专属模板引擎
# ---------------------------------------------------------------------------


class PersonaRules:
    """人群约束规则库。"""

    RULES = {
        "elderly": {
            "max_walk_minutes": 120,
            "max_transit_minutes": 60,
            "avoid_morning_rush": True,
            "require_rest_after_3h": True,
            "prefer_elevator": True,
            "prefer_flat_terrain": True,
        },
        "children": {
            "max_walk_minutes": 90,
            "require_playground_nearby": True,
            "include_kids_activities": True,
            "avoid_night_activities": True,
            "require_frequent_breaks": True,
        },
        "couple": {
            "prefer_scenic_spots": True,
            "prefer_night_view": True,
            "prefer_atmosphere_dining": True,
            "avoid_rush": True,
        },
        "hiking": {
            "match_hiking_duration": True,
            "recommend_gear": True,
            "prefer_nature_lodging": True,
        },
    }

    @classmethod
    def apply(cls, profile: UserProfile) -> dict:
        rules: dict = {}
        if profile.has_elderly:
            rules.update(cls.RULES["elderly"])
        if profile.has_children:
            rules.update(cls.RULES["children"])
        if profile.travelers_type == "情侣":
            rules.update(cls.RULES["couple"])
        return rules

    @classmethod
    def adjust_profile(cls, profile: UserProfile) -> UserProfile:
        """根据人群规则自动调整画像约束。"""
        rules = cls.apply(profile)
        if "max_walk_minutes" in rules:
            profile.max_walk_minutes = min(profile.max_walk_minutes, rules["max_walk_minutes"])
        if "max_transit_minutes" in rules:
            profile.max_transit_minutes = min(profile.max_transit_minutes, rules["max_transit_minutes"])
        return profile


# ---------------------------------------------------------------------------
# 5.7 可行性校验 + 折中方案
# ---------------------------------------------------------------------------


def feasibility_check(profile: UserProfile) -> list[str]:
    """前置校验：检测不可行约束组合，给出折中方案。"""
    conflicts: list[str] = []

    if profile.budget_range and profile.travel_days:
        daily = profile.budget_range / max(profile.travel_days, 1)
        if daily < 200:
            conflicts.append(
                f"预算 {profile.budget_range:.0f} 元 / {profile.travel_days} 天 = 日均 {daily:.0f} 元，"
                f"建议: 1) 缩短天数 2) 增加预算 3) 选择经济型住宿"
            )

    # 老年人 + 高强度 = 冲突
    if profile.has_elderly and profile.pace == "intensive":
        conflicts.append(
            "有老人同行但选择了紧凑节奏，建议调整为 'moderate' 或 'relaxed'"
        )

    # 儿童 + 深夜活动 = 冲突
    if profile.has_children and profile.pace == "intensive":
        conflicts.append(
            "有儿童同行但选择了紧凑节奏，建议增加休息时间"
        )

    return conflicts


# ---------------------------------------------------------------------------
# 5.8 热门景点错峰安排
# ---------------------------------------------------------------------------


def avoid_peak_hours(schedule: list[DayPlan], pois: list[ScoredPOI]) -> list[DayPlan]:
    """将热门景点安排在非高峰时段（早 8 点前或下午 4 点后）。"""
    for day in schedule:
        for act in day.activities:
            poi = next((p for p in pois if p.name == act.poi_name), None)
            if not poi or not getattr(poi, "peak_hours", None):
                continue
            # 已经是错峰时段
            if act.start_time and act.start_time < "10:00":
                continue
            if act.start_time and act.start_time >= "16:00":
                continue
            # 调整到早 8 点
            act.start_time = "08:00"
            dur = getattr(act, "duration_min", 120)
            end_h = 8 + dur // 60
            end_m = dur % 60
            act.end_time = f"{end_h:02d}:{end_m:02d}"
    return schedule
