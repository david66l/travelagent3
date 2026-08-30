"""Frozen end-to-end cases for the production ReAct travel-planning loop."""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field


SCHEMA_VERSION = "full-agent-loop.v3"


class FullAgentLoopCase(BaseModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case_id: str
    suite: Literal["core", "expanded"] = "core"
    slice: str
    user_input: str
    expected_outcome: Literal["draft", "clarification", "draft_or_safe_termination", "revision"] = (
        "draft"
    )
    expected_missing: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
    required_actions: list[str] = Field(default_factory=list)
    safe_required_actions: list[str] = Field(default_factory=list)
    expected_slots: dict[str, object] = Field(default_factory=dict)
    revision_input: str | None = None
    expected_revision_hard: dict[str, object] = Field(default_factory=dict)
    expected_revision_soft: dict[str, object] = Field(default_factory=dict)
    expected_revision_exclusions: list[str] = Field(default_factory=list)


def build_frozen_cases() -> list[FullAgentLoopCase]:
    core_artifacts = [
        "city_knowledge",
        "poi_candidate_set",
        "poi_detail_set",
        "route_matrix",
        "solver_result",
        "validation_report",
        "itinerary_draft",
    ]
    core_actions = [
        "retrieve_city_knowledge",
        "search_pois",
        "get_poi_detail",
        "get_route_matrix",
        "finalize_research",
        "solve_itinerary",
        "validate_itinerary",
        "compose_draft",
    ]

    def draft(
        case_id: str,
        slice_name: str,
        user_input: str,
        *,
        suite: Literal["core", "expanded"] = "core",
        extra_artifacts: list[str] | None = None,
        extra_actions: list[str] | None = None,
        expected_slots: dict[str, object] | None = None,
    ) -> FullAgentLoopCase:
        return FullAgentLoopCase(
            case_id=case_id,
            suite=suite,
            slice=slice_name,
            user_input=user_input,
            required_artifacts=[*core_artifacts, *(extra_artifacts or [])],
            required_actions=[*core_actions, *(extra_actions or [])],
            expected_slots=expected_slots or {},
        )

    def externally_bounded(
        case_id: str,
        slice_name: str,
        user_input: str,
        *,
        suite: Literal["core", "expanded"] = "core",
        evidence_artifact: str,
        evidence_action: str,
        expected_slots: dict[str, object] | None = None,
    ) -> FullAgentLoopCase:
        case = draft(
            case_id,
            slice_name,
            user_input,
            suite=suite,
            extra_artifacts=[evidence_artifact],
            extra_actions=[evidence_action],
            expected_slots=expected_slots,
        )
        return case.model_copy(
            update={
                "expected_outcome": "draft_or_safe_termination",
                "safe_required_actions": [evidence_action, "propose_tradeoff"],
            }
        )

    cases = [
        draft(
            "fal-v1-ordinary-shanghai",
            "ordinary",
            "2026年9月12日去上海玩两天，预算4000元，喜欢历史文化和美食，节奏轻松一点。",
        ),
        draft(
            "fal-v1-family-beijing",
            "family_constraints",
            "2026年10月3日带父母去北京玩三天，两个人预算6000元，少走路，想看历史景点。",
        ),
        draft(
            "fal-v1-weather-hangzhou",
            "weather",
            "2026年9月1日去杭州玩两天，预算3000元，想游西湖，出发前先查天气。",
            extra_artifacts=["weather_snapshot"],
            extra_actions=["get_weather"],
        ),
        draft(
            "fal-v1-opening-suzhou",
            "opening_hours",
            "2026年9月12日去苏州玩两天，预算3000元，想去拙政园，先核实最新营业时间。",
            extra_artifacts=["current_info_search"],
            extra_actions=["search_current_info"],
        ),
        externally_bounded(
            "fal-v1-restaurant-chengdu",
            "restaurant",
            "2026年9月5日去成都玩三天，预算4500元，想吃火锅，帮我查晚上还营业的店。",
            evidence_artifact="current_info_search",
            evidence_action="search_current_info",
        ),
        draft(
            "fal-v1-seasonal-beijing",
            "seasonal_activity",
            "2026年10月20日去北京玩三天，预算5000元，想看红叶，先查今年红叶季的最新情况。",
            extra_artifacts=["current_info_search"],
            extra_actions=["search_current_info"],
        ),
        externally_bounded(
            "fal-v1-train-jinan-shanghai",
            "intercity_transport",
            "2026年9月20日从济南坐高铁去上海玩三天，预算5000元，查一个中午前到的车次。",
            evidence_artifact="transport_search_result",
            evidence_action="search_transport",
        ),
        externally_bounded(
            "fal-v1-event-shanghai",
            "event_trip",
            "2026年9月12日去上海玩两天，预算4000元，想看当周末的音乐节，日期和场馆你去查。",
            evidence_artifact="event_search_result",
            evidence_action="search_current_info",
        ),
        FullAgentLoopCase(
            case_id="fal-v1-clarify-chengdu",
            slice="clarification",
            user_input="我想坐高铁去成都。",
            expected_outcome="clarification",
            expected_missing=["origin", "travel_days"],
        ),
        draft(
            "fal-v2-revise-shanghai",
            "user_revision",
            "2026年9月12日去上海玩两天，预算4000元，喜欢历史文化，正常节奏。",
        ).model_copy(
            update={
                "expected_outcome": "revision",
                "revision_input": "上一版太赶了，改成3天，节奏改为轻松一点。",
                "expected_revision_hard": {"travel_days": 3},
                "expected_revision_soft": {"pace": "relaxed"},
            }
        ),
        draft(
            "fal-v3-xian-must-visit",
            "multiple_must_visit",
            "2026年10月12日去西安玩5天，预算7000元，兵马俑和陕西历史博物馆必须去，偏爱历史文化。",
            suite="expanded",
            expected_slots={
                "destination": "西安",
                "travel_days": 5,
                "total_budget": 7000,
                "must_visit": ["兵马俑", "陕西历史博物馆"],
            },
        ),
        draft(
            "fal-v3-child-shanghai",
            "child_friendly",
            "2026年10月17日一家三口带8岁孩子去上海玩3天，预算6000元，想去科技馆，节奏轻松。",
            suite="expanded",
            expected_slots={
                "destination": "上海",
                "travel_days": 3,
                "travelers_count": 3,
                "has_children": True,
                "pace": "relaxed",
            },
        ),
        draft(
            "fal-v3-elderly-nanjing",
            "elderly_low_fatigue",
            "2026年9月18日带两位65岁父母去南京玩3天，共3人，预算5000元，少走路、不要太赶。",
            suite="expanded",
            expected_slots={
                "destination": "南京",
                "travel_days": 3,
                "travelers_count": 3,
                "has_elderly": True,
                "pace": "relaxed",
            },
        ),
        draft(
            "fal-v3-wheelchair-beijing",
            "wheelchair_accessibility",
            "2026年10月10日带轮椅使用者去北京玩3天，预算7000元，每天步行尽量不超过40分钟，优先无障碍景点。",
            suite="expanded",
            expected_slots={
                "destination": "北京",
                "travel_days": 3,
                "has_wheelchair": True,
                "max_walk_minutes": 40,
            },
        ),
        draft(
            "fal-v3-pregnant-hangzhou",
            "pregnant_low_fatigue",
            "2026年9月15日夫妻两人去杭州玩2天，其中有孕妇，预算4000元，疲劳度要低，行程宽松。",
            suite="expanded",
            expected_slots={
                "destination": "杭州",
                "travel_days": 2,
                "travelers_count": 2,
                "has_pregnant": True,
                "fatigue_preference": "low",
                "pace": "relaxed",
            },
        ),
        draft(
            "fal-v3-food-taboo-chengdu",
            "food_taboo",
            "2026年9月22日去成都玩3天，预算4500元，想吃川菜但对花生严重过敏，推荐饮食时必须避开花生。",
            suite="expanded",
            expected_slots={
                "destination": "成都",
                "travel_days": 3,
                "food_prefs": ["川菜"],
                "food_taboos": ["花生"],
            },
        ),
        draft(
            "fal-v3-exclude-shanghai",
            "must_not_visit",
            "2026年9月26日去上海玩2天，预算3500元，上海博物馆必须去，但不要安排外滩。",
            suite="expanded",
            expected_slots={
                "destination": "上海",
                "travel_days": 2,
                "must_visit": ["上海博物馆"],
                "must_not_visit": ["外滩"],
            },
        ),
        draft(
            "fal-v3-transit-cap-guangzhou",
            "max_transit",
            "2026年10月6日去广州玩3天，预算5000元，喜欢美食和岭南文化，任意两站之间通勤不要超过30分钟。",
            suite="expanded",
            expected_slots={
                "destination": "广州",
                "travel_days": 3,
                "max_transit_minutes": 30,
            },
        ),
        draft(
            "fal-v3-low-budget-suzhou",
            "tight_budget",
            "2026年9月14日一个人去苏州玩3天，总预算只有900元，优先免费或低价景点，正常节奏。",
            suite="expanded",
            expected_slots={
                "destination": "苏州",
                "travel_days": 3,
                "travelers_count": 1,
                "total_budget": 900,
            },
        ).model_copy(
            update={
                "expected_outcome": "draft_or_safe_termination",
                "safe_required_actions": ["propose_tradeoff"],
            }
        ),
        externally_bounded(
            "fal-v3-closure-nanjing",
            "temporary_closure",
            "2026年9月19日去南京玩2天，预算3000元，想去南京博物院，先核实当天是否临时闭馆。",
            suite="expanded",
            evidence_artifact="current_info_search",
            evidence_action="search_current_info",
            expected_slots={
                "destination": "南京",
                "travel_days": 2,
                "information_needs": ["closure"],
            },
        ),
        externally_bounded(
            "fal-v3-flight-sanya",
            "flight_schedule",
            "2026年10月15日从北京坐飞机去三亚玩4天，预算8000元，查一个中午12点前到达的航班。",
            suite="expanded",
            evidence_artifact="transport_search_result",
            evidence_action="search_transport",
            expected_slots={
                "origin": "北京",
                "destination": "三亚",
                "travel_days": 4,
                "transport_modes_requested": ["flight"],
            },
        ),
        externally_bounded(
            "fal-v3-exhibition-shanghai",
            "current_exhibition",
            "2026年9月12日去上海玩2天，预算4000元，想看当周正在举办的艺术展，展览和场馆请先搜索核实。",
            suite="expanded",
            evidence_artifact="event_search_result",
            evidence_action="search_current_info",
            expected_slots={
                "destination": "上海",
                "travel_days": 2,
                "intent_kind": "event_trip",
                "information_needs": ["event"],
            },
        ),
        externally_bounded(
            "fal-v3-late-restaurant-guangzhou",
            "late_restaurant",
            "2026年9月12日去广州玩2天，预算3500元，第一晚22点后才有空，查一家那时仍营业的粤菜馆。",
            suite="expanded",
            evidence_artifact="current_info_search",
            evidence_action="search_current_info",
            expected_slots={
                "destination": "广州",
                "travel_days": 2,
                "information_needs": ["restaurant"],
            },
        ),
        draft(
            "fal-v3-weather-chongqing",
            "weather_adaptation",
            "2026年9月3日去重庆玩3天，预算4500元，先查天气，如果下雨就多安排室内景点。",
            suite="expanded",
            extra_artifacts=["weather_snapshot"],
            extra_actions=["get_weather"],
            expected_slots={
                "destination": "重庆",
                "travel_days": 3,
                "information_needs": ["weather"],
            },
        ),
        FullAgentLoopCase(
            case_id="fal-v3-clarify-destination",
            suite="expanded",
            slice="missing_destination",
            user_input="2026年10月1日想出去玩3天，预算3000元，喜欢自然风光。",
            expected_outcome="clarification",
            expected_missing=["destination"],
            expected_slots={"travel_days": 3, "total_budget": 3000},
        ),
        FullAgentLoopCase(
            case_id="fal-v3-clarify-days",
            suite="expanded",
            slice="missing_travel_days",
            user_input="2026年国庆想去西安看兵马俑，预算5000元。",
            expected_outcome="clarification",
            expected_missing=["travel_days"],
            expected_slots={"destination": "西安", "must_visit": ["兵马俑"]},
        ),
        FullAgentLoopCase(
            case_id="fal-v3-clarify-flight-origin",
            suite="expanded",
            slice="missing_transport_origin",
            user_input="2026年10月18日查航班去三亚玩4天，预算7000元。",
            expected_outcome="clarification",
            expected_missing=["origin"],
            expected_slots={
                "destination": "三亚",
                "travel_days": 4,
                "transport_modes_requested": ["flight"],
            },
        ),
        draft(
            "fal-v3-revise-add-poi",
            "revision_add_must_visit",
            "2026年10月9日去北京玩3天，预算5500元，喜欢历史文化，正常节奏。",
            suite="expanded",
        ).model_copy(
            update={
                "expected_outcome": "revision",
                "revision_input": "上一版漏了故宫，请把故宫设为必去并重新安排。",
                "expected_revision_hard": {"travel_days": 3, "must_visit": ["故宫"]},
            }
        ),
        draft(
            "fal-v3-revise-budget",
            "revision_lower_budget",
            "2026年9月12日去上海玩2天，预算4000元，喜欢城市文化，正常节奏。",
            suite="expanded",
        ).model_copy(
            update={
                "expected_outcome": "revision",
                "revision_input": "预算超了，把总预算降到2500元，其他要求不变。",
                "expected_revision_hard": {"travel_days": 2, "budget_range": 2500},
            }
        ),
        draft(
            "fal-v3-revise-exclude-poi",
            "revision_exclude_poi",
            "2026年9月12日去上海玩2天，预算4000元，第一次来，正常节奏。",
            suite="expanded",
        ).model_copy(
            update={
                "expected_outcome": "revision",
                "revision_input": "我不喜欢人挤人，下一版不要安排外滩。",
                "expected_revision_hard": {"travel_days": 2},
                "expected_revision_exclusions": ["外滩"],
            }
        ),
    ]
    if len({case.case_id for case in cases}) != len(cases):
        raise AssertionError("case ids must be unique")
    return cases


def benchmark_hash(cases: list[FullAgentLoopCase] | None = None) -> str:
    payload = [case.model_dump(mode="json") for case in (cases or build_frozen_cases())]
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "FullAgentLoopCase",
    "benchmark_hash",
    "build_frozen_cases",
]
