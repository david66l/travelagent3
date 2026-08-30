"""Frozen semantic benchmark for demand and itinerary-revision understanding."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from models.travel_slots import RevisionParseOutput, SlotParseOutput


SCHEMA_VERSION = "intent-revision-semantic.v1"


class InitialExpectation(BaseModel):
    slot_values: dict[str, Any] = Field(default_factory=dict)
    required_needs: list[str] = Field(default_factory=list)
    any_of_needs: list[str] = Field(default_factory=list)
    forbidden_needs: list[str] = Field(default_factory=list)
    required_modes: list[str] = Field(default_factory=list)
    forbidden_modes: list[str] = Field(default_factory=list)
    event_query_required: bool = False


class ExpectedRevisionOperation(BaseModel):
    fields: list[str] = Field(min_length=1)
    operation: Literal["set", "add", "remove", "clear"] | None = None
    value: Any = None


class RevisionExpectation(BaseModel):
    operations: list[ExpectedRevisionOperation] = Field(default_factory=list)
    forbidden_fields: list[str] = Field(default_factory=list)
    needs_clarification: bool = False


class InitialSemanticCase(BaseModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case_id: str
    slice: str
    text: str
    expected: InitialExpectation
    kind: Literal["initial"] = "initial"


class RevisionSemanticCase(BaseModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case_id: str
    slice: str
    text: str
    current_goal: dict[str, Any]
    expected: RevisionExpectation
    kind: Literal["revision"] = "revision"


SemanticCase = InitialSemanticCase | RevisionSemanticCase


def _initial(
    case_id: str,
    slice_name: str,
    text: str,
    *,
    slot_values: dict[str, Any] | None = None,
    needs: list[str] | None = None,
    any_of_needs: list[str] | None = None,
    forbidden_needs: list[str] | None = None,
    modes: list[str] | None = None,
    forbidden_modes: list[str] | None = None,
    event_query: bool = False,
) -> InitialSemanticCase:
    return InitialSemanticCase(
        case_id=case_id,
        slice=slice_name,
        text=text,
        expected=InitialExpectation(
            slot_values=slot_values or {},
            required_needs=needs or [],
            any_of_needs=any_of_needs or [],
            forbidden_needs=forbidden_needs or [],
            required_modes=modes or [],
            forbidden_modes=forbidden_modes or [],
            event_query_required=event_query,
        ),
    )


def _op(
    field: str | list[str],
    value: Any = None,
    operation: Literal["set", "add", "remove", "clear"] | None = None,
) -> ExpectedRevisionOperation:
    return ExpectedRevisionOperation(
        fields=[field] if isinstance(field, str) else field,
        operation=operation,
        value=value,
    )


_BASE_GOAL = {
    "hard_constraints": {
        "origin": "南京",
        "destination": "上海",
        "travel_days": 3,
        "budget_range": 6000,
        "intent_kind": "itinerary",
    },
    "soft_preferences": {"pace": "moderate", "hotel_preference": "comfortable"},
}


def _revision(
    case_id: str,
    slice_name: str,
    text: str,
    *,
    operations: list[ExpectedRevisionOperation] | None = None,
    forbidden_fields: list[str] | None = None,
    clarify: bool = False,
) -> RevisionSemanticCase:
    return RevisionSemanticCase(
        case_id=case_id,
        slice=slice_name,
        text=text,
        current_goal=_BASE_GOAL,
        expected=RevisionExpectation(
            operations=operations or [],
            forbidden_fields=forbidden_fields or [],
            needs_clarification=clarify,
        ),
    )


def build_frozen_cases() -> list[SemanticCase]:
    """Return 100 human-authored, deterministic cases (50 initial + 50 revision)."""
    cases: list[SemanticCase] = []

    ordinary = [
        ("杭州", 3, "帮我安排杭州三天的历史文化行程"),
        ("成都", 4, "去成都玩四天，重点吃当地小吃"),
        ("北京", 2, "北京两日游，想多看博物馆"),
        ("厦门", 5, "我想在厦门慢慢玩五天"),
        ("南京", 1, "只有一天时间，帮我逛逛南京"),
    ]
    for index, (city, days, text) in enumerate(ordinary, 1):
        cases.append(
            _initial(
                f"ir-v1-initial-ordinary-{index:02d}",
                "ordinary",
                text,
                slot_values={"destination": city, "travel_days": days, "intent_kind": "itinerary"},
                forbidden_needs=["event", "transport", "weather"],
            )
        )

    implicit_events = [
        ("上海", 2, "我抢到了9月12日周杰伦上海站的票，顺便安排两天行程"),
        ("北京", 3, "周六晚去工体看国安主场，前后在北京待三天"),
        ("广州", 2, "已经买好广州国际车展周日上午的票，安排个周末行程"),
        ("成都", 3, "草莓音乐节那天我一定要去，顺便玩三天成都"),
        ("杭州", 2, "周五晚有一场话剧票在杭州，帮我排两天"),
    ]
    for index, (city, days, text) in enumerate(implicit_events, 1):
        cases.append(
            _initial(
                f"ir-v1-initial-implicit-event-{index:02d}",
                "implicit_event",
                text,
                slot_values={"destination": city, "travel_days": days, "intent_kind": "event_trip"},
                needs=["event"],
                event_query=True,
            )
        )

    explicit_events = [
        ("深圳", 2, "去深圳看演唱会，场馆和开场时间你帮我查，再排两天"),
        ("上海", 3, "围绕上海网球大师赛安排三天旅行"),
        ("北京", 2, "想去北京看那个埃及文物展，具体馆和日期需要核实，玩两天"),
        ("青岛", 4, "啤酒节期间去青岛四天，活动时间地点要查清楚"),
        ("广州", 2, "广州漫展周末游，两天，先查展馆和开放时间"),
    ]
    for index, (city, days, text) in enumerate(explicit_events, 1):
        cases.append(
            _initial(
                f"ir-v1-initial-explicit-event-{index:02d}",
                "explicit_event",
                text,
                slot_values={"destination": city, "travel_days": days, "intent_kind": "event_trip"},
                needs=["event"],
                event_query=True,
            )
        )

    schedules = [
        ("上海", 3, "从济南去上海玩三天，坐高铁，查一个上午能到的车次", "train"),
        ("成都", 4, "北京出发去成都四天，帮我找晚饭前落地的航班", "flight"),
        ("珠海", 2, "从广州去珠海两天，想查直达大巴班次", "bus"),
        ("舟山", 3, "从上海去舟山玩三天，需要查轮渡时间", "ferry"),
        ("杭州", 2, "南京到杭州周末两日游，去程火车别太早，帮我查车次", "train"),
    ]
    for index, (city, days, text, mode) in enumerate(schedules, 1):
        cases.append(
            _initial(
                f"ir-v1-initial-schedule-{index:02d}",
                "intercity_schedule",
                text,
                slot_values={"destination": city, "travel_days": days, "intent_kind": "itinerary"},
                needs=["transport"],
                modes=[mode],
            )
        )

    city_preferences = [
        ("上海玩三天，市内尽量坐地铁，不用查进出城班次", "public"),
        ("北京两天，景点之间优先打车", "taxi"),
        ("苏州三日游，城区尽量步行", "walk"),
        ("三亚四天，落地后想租车自驾", "rental_car"),
        ("成都三天，市内地铁和打车混着来", "mixed"),
    ]
    for index, (text, preference) in enumerate(city_preferences, 1):
        cases.append(
            _initial(
                f"ir-v1-initial-city-transit-{index:02d}",
                "city_transport_preference",
                text,
                slot_values={"intent_kind": "itinerary", "transport_preference": preference},
                forbidden_needs=["transport"],
                forbidden_modes=["flight", "train", "bus", "ferry"],
            )
        )

    opening = [
        ("苏州", "去苏州两天，拙政园要安排进去，并核实当天是否开放"),
        ("北京", "北京三天，故宫放第二天，先确认最新开放安排"),
        ("上海", "上海两日游要去天文馆，查一下当天营业时间"),
        ("西安", "西安三天必须去陕历博，确认预约日几点开门"),
        ("成都", "成都两天想逛三星堆，核实那周的开放时间"),
    ]
    for index, (city, text) in enumerate(opening, 1):
        cases.append(
            _initial(
                f"ir-v1-initial-opening-{index:02d}",
                "opening_hours",
                text,
                slot_values={"destination": city, "intent_kind": "itinerary"},
                needs=["opening_hours"],
                forbidden_needs=["event"],
            )
        )

    closures = [
        ("北京", "北京玩三天，听说首博可能临时闭馆，出发前核实"),
        ("杭州", "杭州两天要去灵隐寺，查一下近期有没有关闭通知"),
        ("上海", "上海三日游，确认美术馆那天没有停业"),
        ("南京", "南京两天，看看总统府最近是否临时关闭"),
        ("武汉", "武汉三天，黄鹤楼如遇维护关闭就换景点，先查公告"),
    ]
    for index, (city, text) in enumerate(closures, 1):
        cases.append(
            _initial(
                f"ir-v1-initial-closure-{index:02d}",
                "closure",
                text,
                slot_values={"destination": city, "intent_kind": "itinerary"},
                needs=["closure"],
                forbidden_needs=["event"],
            )
        )

    seasonal = [
        ("武汉", "三月底去武汉玩三天，想看当季花期，合适就安排赏花"),
        ("北京", "十月底北京三天，想确认红叶现在到没到最佳观赏期"),
        ("洛阳", "四月去洛阳两天，按当时牡丹花期安排"),
        ("哈尔滨", "十二月哈尔滨四天，先查冰雪景观是否已经开放"),
        ("婺源", "去婺源三天看油菜花，按今年实际花期排日期"),
    ]
    for index, (city, text) in enumerate(seasonal, 1):
        cases.append(
            _initial(
                f"ir-v1-initial-seasonal-{index:02d}",
                "seasonal_activity",
                text,
                slot_values={"destination": city, "intent_kind": "itinerary"},
                needs=["seasonal_activity"],
                forbidden_needs=["event"],
            )
        )

    weather = [
        ("黄山", "下周去黄山两天，户外为主，先确认天气是否适合"),
        ("三亚", "三亚四天，出海前需要看台风和降雨预报"),
        ("张家界", "张家界三日徒步，按出发那几天的天气安排"),
        ("北京", "北京两天带孩子，若下雨就多排室内，先查天气"),
        ("厦门", "厦门三天想骑行环岛路，需要确认风雨情况"),
    ]
    for index, (city, text) in enumerate(weather, 1):
        cases.append(
            _initial(
                f"ir-v1-initial-weather-{index:02d}",
                "weather",
                text,
                slot_values={"destination": city, "intent_kind": "itinerary"},
                needs=["weather"],
            )
        )

    restaurants = [
        ("上海", "上海两天，想去一家深夜还能吃饭的本帮菜馆，核实营业时间"),
        ("成都", "成都三天，帮我找当天仍营业而且不排太久的火锅店"),
        ("广州", "广州两日美食游，餐厅是否正常营业要查最新信息"),
        ("北京", "北京三天，晚场结束后安排烤鸭，确认餐厅最晚接待时间"),
        ("长沙", "长沙两天想吃夜宵，查一查目标店近期营业安排"),
    ]
    for index, (city, text) in enumerate(restaurants, 1):
        cases.append(
            _initial(
                f"ir-v1-initial-restaurant-{index:02d}",
                "restaurant",
                text,
                slot_values={"destination": city, "intent_kind": "itinerary"},
                any_of_needs=["restaurant", "opening_hours"],
                forbidden_needs=["event"],
            )
        )

    pace_cases = [
        ("第二天太赶了，整体轻松一点", "relaxed"),
        ("老人走不动，把节奏调慢", "relaxed"),
        ("景点太少，我能走很多路，安排紧凑些", "intensive"),
        ("不要那么松散，尽量多看几个地方", "intensive"),
        ("现在太极端，改成正常节奏", "moderate"),
    ]
    for index, (text, value) in enumerate(pace_cases, 1):
        cases.append(
            _revision(
                f"ir-v1-revision-pace-{index:02d}",
                "pace_revision",
                text,
                operations=[_op("pace", value, "set")],
            )
        )

    budget_cases = [
        ("总预算控制在4500以内，但酒店品质别降低", 4500),
        ("预算降到5000，同时住宿还要舒适", 5000),
        ("最多花3800，酒店不要换成青旅", 3800),
        ("整体不超过5200，住的地方保持现在档次", 5200),
        ("把预算上限改成4000，不过住宿质量不能降", 4000),
    ]
    for index, (text, value) in enumerate(budget_cases, 1):
        cases.append(
            _revision(
                f"ir-v1-revision-budget-{index:02d}",
                "multi_clause_budget",
                text,
                operations=[_op("budget_range", value, "set"), _op("hotel_preference")],
            )
        )

    poi_cases = [
        ("删掉博物馆，换点自然风景", "博物馆"),
        ("不要再安排外滩", "外滩"),
        ("把迪士尼排除掉", "迪士尼"),
        ("第二天的古镇不想去了", "古镇"),
        ("取消科技馆，孩子不感兴趣", "科技馆"),
    ]
    for index, (text, value) in enumerate(poi_cases, 1):
        cases.append(
            _revision(
                f"ir-v1-revision-remove-poi-{index:02d}",
                "poi_removal",
                text,
                operations=[_op(["must_not_visit", "avoid_pois"], value)],
            )
        )

    revision_events = [
        "把周六晚的周杰伦上海站加进去，场馆和时间你去查",
        "新加一个申花主场，具体开球和球场先搜索",
        "行程里加入那个埃及文物展，日期地点需要核实",
        "第二晚要看话剧，票面只写了剧名，帮我查场馆时间",
        "加上周末音乐节，先确认是哪天和在哪里",
    ]
    for index, text in enumerate(revision_events, 1):
        cases.append(
            _revision(
                f"ir-v1-revision-event-{index:02d}",
                "event_revision",
                text,
                operations=[
                    _op("intent_kind", "event_trip", "set"),
                    _op("event_query"),
                    _op("information_needs", "event"),
                ],
            )
        )

    revision_schedules = [
        ("改成坐高铁去，查一个中午前到上海的班次", "train"),
        ("还是坐飞机，帮我找晚饭前落地的航班", "flight"),
        ("去程改成长途大巴，需要查发车时间", "bus"),
        ("加一段轮渡，班次时间先核实", "ferry"),
        ("返程坐火车，别晚于晚上十点到南京", "train"),
    ]
    for index, (text, mode) in enumerate(revision_schedules, 1):
        cases.append(
            _revision(
                f"ir-v1-revision-schedule-{index:02d}",
                "intercity_schedule_revision",
                text,
                operations=[
                    _op("transport_modes_requested", mode),
                    _op("information_needs", "transport"),
                ],
            )
        )

    revision_city_transport = [
        ("还是尽量坐公共交通吧，不用查车次", "public"),
        ("市内景点之间都打车", "taxi"),
        ("老城区只安排步行", "walk"),
        ("落地以后租车自驾，不是要查航班", "rental_car"),
        ("地铁和出租车结合，哪个方便用哪个", "mixed"),
    ]
    for index, (text, preference) in enumerate(revision_city_transport, 1):
        cases.append(
            _revision(
                f"ir-v1-revision-city-transit-{index:02d}",
                "city_transport_revision",
                text,
                operations=[_op("transport_preference", preference, "set")],
                forbidden_fields=["transport_modes_requested", "information_needs"],
            )
        )

    day_cases = [
        ("改成五天", 5),
        ("在原来基础上多玩两天", 5),
        ("少玩一天就好", 2),
        ("延长到七天", 7),
        ("时间不够，压缩成两天", 2),
    ]
    for index, (text, days) in enumerate(day_cases, 1):
        cases.append(
            _revision(
                f"ir-v1-revision-days-{index:02d}",
                "travel_days_revision",
                text,
                operations=[_op("travel_days", days, "set")],
            )
        )

    vague = ["不好，改一下", "感觉不太对", "换个方案吧", "我不喜欢这版", "再优化一下"]
    for index, text in enumerate(vague, 1):
        cases.append(
            _revision(
                f"ir-v1-revision-vague-{index:02d}",
                "ambiguous_revision",
                text,
                clarify=True,
            )
        )

    preference_cases = [
        ("多安排一些历史类景点", _op("interests", "历史")),
        ("我不吃辣，把饮食偏好改清淡", _op("food_preferences", "清淡")),
        ("酒店最好靠近市中心", _op("hotel_preference")),
        ("想多看自然风光，少逛商场", _op("interests", "自然")),
        ("同行有孩子，安排亲子友好一点", _op("has_children", True, "set")),
    ]
    for index, (text, operation) in enumerate(preference_cases, 1):
        cases.append(
            _revision(
                f"ir-v1-revision-preference-{index:02d}",
                "preference_revision",
                text,
                operations=[operation],
            )
        )

    constraint_cases = [
        ("目的地改成杭州", _op("destination", "杭州", "set")),
        ("出发地改为苏州", _op("origin", "苏州", "set")),
        ("改到2026年9月10日出发，12日结束", _op("start_date", "2026-09-10", "set")),
        ("这次有老人同行", _op("has_elderly", True, "set")),
        ("每段交通最多45分钟", _op("max_transit_minutes", 45, "set")),
    ]
    for index, (text, operation) in enumerate(constraint_cases, 1):
        cases.append(
            _revision(
                f"ir-v1-revision-constraint-{index:02d}",
                "constraint_revision",
                text,
                operations=[operation],
            )
        )

    if len(cases) != 100:
        raise AssertionError(f"expected 100 frozen cases, got {len(cases)}")
    if len({case.case_id for case in cases}) != len(cases):
        raise AssertionError("case ids must be unique")
    return cases


def benchmark_hash(cases: list[SemanticCase] | None = None) -> str:
    selected = cases or build_frozen_cases()
    payload = [case.model_dump(mode="json") for case in selected]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _matches(expected: Any, actual: Any) -> bool:
    if isinstance(actual, dict) and expected is not None:
        candidates = [actual.get(key) for key in ("max", "value", "amount", "total")]
        if expected in candidates:
            return True
    if isinstance(actual, list) and not isinstance(expected, list):
        return expected in actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float, str)):
        try:
            return float(expected) == float(actual)
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str) and isinstance(actual, str):
        expected_text = expected.strip()
        actual_text = actual.strip()
        if expected_text and actual_text:
            return expected_text in actual_text or actual_text in expected_text
    return expected == actual


def score_initial(case: InitialSemanticCase, result: SlotParseOutput) -> tuple[bool, list[str]]:
    failures: list[str] = []
    slots = result.slots.model_dump()
    if result.parse_source != "llm":
        failures.append("MODEL_FALLBACK_USED")
    for field, expected in case.expected.slot_values.items():
        if not _matches(expected, slots.get(field)):
            failures.append(f"SLOT_MISMATCH:{field}:{slots.get(field)!r}")
    needs = set(result.slots.information_needs)
    for item in case.expected.required_needs:
        if item not in needs:
            failures.append(f"MISSING_NEED:{item}")
    if case.expected.any_of_needs and not needs.intersection(case.expected.any_of_needs):
        failures.append(f"MISSING_ANY_NEED:{'|'.join(case.expected.any_of_needs)}")
    for item in case.expected.forbidden_needs:
        if item in needs:
            failures.append(f"FORBIDDEN_NEED:{item}")
    modes = set(result.slots.transport_modes_requested)
    for item in case.expected.required_modes:
        if item not in modes:
            failures.append(f"MISSING_MODE:{item}")
    for item in case.expected.forbidden_modes:
        if item in modes:
            failures.append(f"FORBIDDEN_MODE:{item}")
    if case.expected.event_query_required and not result.slots.event_query:
        failures.append("EVENT_QUERY_MISSING")
    return not failures, failures


def score_revision(
    case: RevisionSemanticCase,
    result: RevisionParseOutput,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if result.needs_clarification != case.expected.needs_clarification:
        failures.append(f"CLARIFICATION_MISMATCH:{result.needs_clarification}")
    for expected in case.expected.operations:
        matched = False
        for operation in result.operations:
            if operation.field not in expected.fields:
                continue
            if expected.operation and operation.operation != expected.operation:
                continue
            if expected.value is not None and not _matches(expected.value, operation.value):
                continue
            matched = True
            break
        if not matched:
            failures.append(f"MISSING_OPERATION:{'|'.join(expected.fields)}:{expected.value!r}")
    for field in case.expected.forbidden_fields:
        if any(
            operation.field == field and operation.operation in {"set", "add"}
            for operation in result.operations
        ):
            failures.append(f"FORBIDDEN_OPERATION:{field}")
    return not failures, failures


__all__ = [
    "SCHEMA_VERSION",
    "InitialSemanticCase",
    "RevisionSemanticCase",
    "SemanticCase",
    "benchmark_hash",
    "build_frozen_cases",
    "score_initial",
    "score_revision",
]
