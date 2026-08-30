"""Deterministic, diverse user scenarios for native ReAct teacher rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from evaluation.full_agent_loop_benchmark import FullAgentLoopCase


@dataclass(frozen=True)
class CityProfile:
    city: str
    interest: str
    must_visit: str
    alternative: str
    food: str


CITY_PROFILES = (
    CityProfile("上海", "城市历史与建筑", "上海博物馆", "外滩", "本帮菜"),
    CityProfile("北京", "历史文化", "故宫", "天坛", "北京菜"),
    CityProfile("杭州", "湖景与人文", "西湖", "灵隐寺", "杭帮菜"),
    CityProfile("苏州", "园林与古城", "拙政园", "虎丘", "苏帮菜"),
    CityProfile("成都", "巴蜀文化与美食", "武侯祠", "熊猫基地", "川菜"),
    CityProfile("南京", "近现代历史", "南京博物院", "中山陵", "金陵菜"),
    CityProfile("西安", "古代历史", "兵马俑", "陕西历史博物馆", "陕西菜"),
    CityProfile("广州", "岭南文化", "陈家祠", "广州塔", "粤菜"),
    CityProfile("重庆", "山城夜景", "洪崖洞", "三峡博物馆", "重庆菜"),
    CityProfile("武汉", "江城文化", "黄鹤楼", "湖北省博物馆", "湖北菜"),
    CityProfile("厦门", "海滨与近代建筑", "鼓浪屿", "厦门园林植物园", "闽南菜"),
    CityProfile("深圳", "现代城市与自然", "深圳博物馆", "莲花山公园", "粤菜"),
)


def _common_slots(profile: CityProfile, days: int, budget: int) -> dict[str, object]:
    return {
        "destination": profile.city,
        "travel_days": days,
        "total_budget": budget,
    }


def build_native_react_training_case(index: int) -> FullAgentLoopCase:
    """Build one stable scenario; index controls city, intent family and values."""
    if index < 0:
        raise ValueError("index must be non-negative")
    profile = CITY_PROFILES[index % len(CITY_PROFILES)]
    family = (index // len(CITY_PROFILES)) % 20
    cycle = index // (len(CITY_PROFILES) * 20)
    start = date(2026, 9, 3) + timedelta(days=(index * 3 + cycle) % 75)
    days = 2 + index % 3
    budget = 2800 + (index % 6) * 700
    date_text = start.isoformat()
    case_id = f"native-react-train-{index:04d}"
    slots = _common_slots(profile, days, budget)
    expected_outcome = "draft"
    required_actions: list[str] = []
    safe_required_actions: list[str] = []

    if family == 0:
        slice_name = "ordinary_interest"
        text = (
            f"{date_text}去{profile.city}玩{days}天，总预算{budget}元，"
            f"喜欢{profile.interest}，节奏正常。"
        )
    elif family == 1:
        slice_name = "elderly_relaxed"
        text = (
            f"{date_text}带两位65岁父母去{profile.city}{days}天，共3人，预算{budget}元，"
            "少走路、不要赶行程。"
        )
        slots.update({"travelers_count": 3, "has_elderly": True, "pace": "relaxed"})
    elif family == 2:
        slice_name = "child_friendly"
        text = (
            f"{date_text}一家三口带8岁孩子去{profile.city}玩{days}天，预算{budget}元，"
            f"想去{profile.must_visit}，安排得轻松些。"
        )
        slots.update({"travelers_count": 3, "has_children": True, "pace": "relaxed"})
    elif family == 3:
        slice_name = "wheelchair_accessibility"
        text = (
            f"{date_text}带轮椅使用者去{profile.city}{days}天，预算{budget}元，"
            "优先无障碍景点，每天步行不超过40分钟。"
        )
        slots.update({"has_wheelchair": True, "max_walk_minutes": 40})
    elif family == 4:
        slice_name = "tight_budget"
        tight_budget = 1600 + (index % 3) * 300
        slots["total_budget"] = tight_budget
        text = (
            f"{date_text}一个人去{profile.city}玩{days}天，总预算只有{tight_budget}元，"
            "住宿交通和门票都算进去，优先免费景点。"
        )
    elif family == 5:
        slice_name = "multiple_must_visit"
        text = (
            f"{date_text}去{profile.city}玩{days}天，预算{budget}元，"
            f"{profile.must_visit}和{profile.alternative}都必须安排。"
        )
        slots["must_visit"] = [profile.must_visit, profile.alternative]
    elif family == 6:
        slice_name = "must_not_visit"
        text = (
            f"{date_text}去{profile.city}玩{days}天，预算{budget}元，"
            f"{profile.must_visit}必须去，但不要安排{profile.alternative}。"
        )
        slots.update(
            {
                "must_visit": [profile.must_visit],
                "must_not_visit": [profile.alternative],
            }
        )
    elif family == 7:
        slice_name = "max_transit"
        text = (
            f"{date_text}去{profile.city}玩{days}天，预算{budget}元，喜欢{profile.interest}，"
            "任意两个景点之间通勤不要超过35分钟。"
        )
        slots["max_transit_minutes"] = 35
    elif family == 8:
        slice_name = "weather_adaptation"
        text = (
            f"{date_text}去{profile.city}玩{days}天，预算{budget}元，先查天气，"
            "如果下雨就多安排室内景点。"
        )
        required_actions = ["get_weather"]
    elif family == 9:
        slice_name = "opening_hours"
        text = (
            f"{date_text}去{profile.city}玩{days}天，预算{budget}元，想去{profile.must_visit}，"
            "请先核实最新营业时间再排行程。"
        )
        required_actions = ["search_current_info"]
        safe_required_actions = ["search_current_info", "propose_tradeoff"]
        expected_outcome = "draft_or_safe_termination"
    elif family == 10:
        slice_name = "late_restaurant"
        text = (
            f"{date_text}去{profile.city}玩{days}天，预算{budget}元，想吃{profile.food}，"
            "帮我查晚上22点以后还营业的店。"
        )
        required_actions = ["search_current_info"]
        safe_required_actions = ["search_current_info", "propose_tradeoff"]
        expected_outcome = "draft_or_safe_termination"
    elif family == 11:
        slice_name = "seasonal_activity"
        text = (
            f"{date_text}去{profile.city}玩{days}天，预算{budget}元，"
            "先查当季活动和最新开放情况，再安排景点。"
        )
        required_actions = ["search_current_info"]
        safe_required_actions = ["search_current_info", "propose_tradeoff"]
        expected_outcome = "draft_or_safe_termination"
    elif family == 12:
        slice_name = "event_trip"
        text = (
            f"{date_text}去{profile.city}玩{days}天，预算{budget}元，想看当周末的音乐节，"
            "具体活动日期和场馆请先查清楚。"
        )
        required_actions = ["search_current_info"]
        safe_required_actions = ["search_current_info", "propose_tradeoff"]
        expected_outcome = "draft_or_safe_termination"
    elif family == 13:
        slice_name = "food_allergy"
        text = (
            f"{date_text}去{profile.city}玩{days}天，预算{budget}元，想吃{profile.food}，"
            "但对花生严重过敏，餐饮推荐必须避开花生。"
        )
        slots["food_taboos"] = ["花生"]
    elif family == 14:
        slice_name = "pregnant_low_fatigue"
        text = (
            f"{date_text}夫妻两人去{profile.city}玩{days}天，其中有孕妇，预算{budget}元，"
            "疲劳度要低，行程宽松。"
        )
        slots.update(
            {
                "travelers_count": 2,
                "has_pregnant": True,
                "fatigue_preference": "low",
                "pace": "relaxed",
            }
        )
    elif family == 15:
        slice_name = "solo_relaxed"
        text = (
            f"{date_text}我一个人去{profile.city}{days}天，预算{budget}元，"
            f"喜欢{profile.interest}，每天不要安排太满。"
        )
        slots.update({"travelers_count": 1, "pace": "relaxed"})
    elif family == 16:
        slice_name = "friends_food"
        text = (
            f"{date_text}和3个朋友去{profile.city}玩{days}天，预算{budget}元，"
            f"重点体验{profile.food}和当地夜生活。"
        )
        slots["travelers_count"] = 4
    elif family == 17:
        slice_name = "museum_focus"
        text = (
            f"{date_text}去{profile.city}{days}天，预算{budget}元，"
            f"主要想看博物馆和{profile.interest}相关景点，节奏轻松。"
        )
        slots["pace"] = "relaxed"
    elif family == 18:
        slice_name = "first_time_landmarks"
        text = (
            f"{date_text}第一次去{profile.city}，玩{days}天，预算{budget}元，"
            "经典地标要覆盖，但同一区域尽量放在同一天。"
        )
    else:
        slice_name = "low_crowd"
        text = (
            f"{date_text}去{profile.city}玩{days}天，预算{budget}元，"
            f"想看{profile.interest}，尽量避开人挤人的地方。"
        )
        slots["avoid_crowds"] = True

    return FullAgentLoopCase(
        case_id=case_id,
        suite="expanded",
        slice=slice_name,
        user_input=text,
        expected_outcome=expected_outcome,
        required_actions=required_actions,
        safe_required_actions=safe_required_actions,
        expected_slots=slots,
    )


def build_native_react_training_cases(start_index: int, count: int) -> list[FullAgentLoopCase]:
    if start_index < 0 or count < 1:
        raise ValueError("start_index must be non-negative and count must be positive")
    return [build_native_react_training_case(index) for index in range(start_index, start_index + count)]


__all__ = [
    "CITY_PROFILES",
    "build_native_react_training_case",
    "build_native_react_training_cases",
]
