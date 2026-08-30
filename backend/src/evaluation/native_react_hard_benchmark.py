"""Frozen, cluster-aware hard benchmark for the production Native ReAct loop."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field, model_validator

from evaluation.external_benchmark import character_ngrams, jaccard_similarity, normalize_text
from evaluation.full_agent_loop_benchmark import FullAgentLoopCase, build_frozen_cases


SCHEMA_VERSION = "native-react-independent-hard.v1"
FAMILIES = (
    "clarification",
    "poi_grounding",
    "current_information",
    "transport_grounding",
    "hard_constraint_conflict",
    "accessibility",
    "tool_recovery",
    "revision",
    "safe_termination",
    "noisy_multilingual",
)
DEV_CITIES = {"北京", "上海", "广州", "成都"}

CORE_ARTIFACTS = [
    "city_knowledge",
    "poi_candidate_set",
    "poi_detail_set",
    "route_matrix",
    "solver_result",
    "validation_report",
    "itinerary_draft",
]
CORE_ACTIONS = [
    "retrieve_city_knowledge",
    "search_pois",
    "get_poi_detail",
    "get_route_matrix",
    "finalize_research",
    "solve_itinerary",
    "validate_itinerary",
    "compose_draft",
]


class CityProfile(BaseModel):
    slug: str
    city: str
    cluster: Literal["north", "east", "south", "southwest", "central", "northwest"]
    must_visit: str
    alternative: str
    cuisine: str
    origin: str


CITY_PROFILES = [
    CityProfile(
        slug="beijing",
        city="北京",
        cluster="north",
        must_visit="故宫",
        alternative="颐和园",
        cuisine="京味菜",
        origin="天津",
    ),
    CityProfile(
        slug="shanghai",
        city="上海",
        cluster="east",
        must_visit="上海博物馆",
        alternative="外滩",
        cuisine="本帮菜",
        origin="济南",
    ),
    CityProfile(
        slug="guangzhou",
        city="广州",
        cluster="south",
        must_visit="陈家祠",
        alternative="广东省博物馆",
        cuisine="粤菜",
        origin="长沙",
    ),
    CityProfile(
        slug="chengdu",
        city="成都",
        cluster="southwest",
        must_visit="金沙遗址博物馆",
        alternative="武侯祠",
        cuisine="川菜",
        origin="重庆",
    ),
    CityProfile(
        slug="hangzhou",
        city="杭州",
        cluster="east",
        must_visit="西湖",
        alternative="良渚博物院",
        cuisine="杭帮菜",
        origin="南京",
    ),
    CityProfile(
        slug="xian",
        city="西安",
        cluster="northwest",
        must_visit="兵马俑",
        alternative="陕西历史博物馆",
        cuisine="西安小吃",
        origin="郑州",
    ),
    CityProfile(
        slug="chongqing",
        city="重庆",
        cluster="southwest",
        must_visit="三峡博物馆",
        alternative="李子坝",
        cuisine="重庆火锅",
        origin="成都",
    ),
    CityProfile(
        slug="suzhou",
        city="苏州",
        cluster="east",
        must_visit="拙政园",
        alternative="苏州博物馆",
        cuisine="苏帮菜",
        origin="上海",
    ),
    CityProfile(
        slug="nanjing",
        city="南京",
        cluster="east",
        must_visit="南京博物院",
        alternative="明孝陵",
        cuisine="金陵菜",
        origin="合肥",
    ),
    CityProfile(
        slug="xiamen",
        city="厦门",
        cluster="south",
        must_visit="鼓浪屿",
        alternative="厦门园林植物园",
        cuisine="闽南菜",
        origin="福州",
    ),
    CityProfile(
        slug="qingdao",
        city="青岛",
        cluster="north",
        must_visit="青岛啤酒博物馆",
        alternative="八大关",
        cuisine="海鲜",
        origin="济南",
    ),
    CityProfile(
        slug="dali",
        city="大理",
        cluster="southwest",
        must_visit="崇圣寺三塔",
        alternative="洱海生态廊道",
        cuisine="白族菜",
        origin="昆明",
    ),
    CityProfile(
        slug="lijiang",
        city="丽江",
        cluster="southwest",
        must_visit="丽江古城",
        alternative="黑龙潭",
        cuisine="纳西菜",
        origin="昆明",
    ),
    CityProfile(
        slug="sanya",
        city="三亚",
        cluster="south",
        must_visit="南山文化旅游区",
        alternative="亚龙湾",
        cuisine="海南菜",
        origin="北京",
    ),
    CityProfile(
        slug="changsha",
        city="长沙",
        cluster="central",
        must_visit="湖南博物院",
        alternative="岳麓书院",
        cuisine="湘菜",
        origin="武汉",
    ),
    CityProfile(
        slug="wuhan",
        city="武汉",
        cluster="central",
        must_visit="湖北省博物馆",
        alternative="东湖",
        cuisine="湖北菜",
        origin="长沙",
    ),
    CityProfile(
        slug="kunming",
        city="昆明",
        cluster="southwest",
        must_visit="云南省博物馆",
        alternative="翠湖公园",
        cuisine="云南菜",
        origin="贵阳",
    ),
    CityProfile(
        slug="guilin",
        city="桂林",
        cluster="south",
        must_visit="漓江",
        alternative="象鼻山",
        cuisine="桂林米粉",
        origin="广州",
    ),
    CityProfile(
        slug="lhasa",
        city="拉萨",
        cluster="northwest",
        must_visit="布达拉宫",
        alternative="罗布林卡",
        cuisine="藏餐",
        origin="成都",
    ),
    CityProfile(
        slug="shenzhen",
        city="深圳",
        cluster="south",
        must_visit="深圳博物馆",
        alternative="莲花山公园",
        cuisine="潮汕菜",
        origin="广州",
    ),
]


class FaultSpec(BaseModel):
    action: str
    occurrence: int = Field(default=1, ge=1)
    fault_type: Literal["timeout", "rate_limit", "empty_result", "stale_data"]
    recoverable: bool
    expected_behavior: str


class BenchmarkMetadata(BaseModel):
    benchmark_schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    statistical_unit_id: str = Field(pattern=r"^nrhb-v1-[a-z0-9-]+$")
    split: Literal["dev", "test"]
    family: Literal[
        "clarification",
        "poi_grounding",
        "current_information",
        "transport_grounding",
        "hard_constraint_conflict",
        "accessibility",
        "tool_recovery",
        "revision",
        "safe_termination",
        "noisy_multilingual",
    ]
    difficulty: Literal["L2", "L3", "L4"]
    city: str
    city_cluster: str
    challenge_variant: str
    cluster_id: str
    prompt_pattern_id: str
    authoring_method: Literal["deterministic_composition"] = "deterministic_composition"
    human_review_status: Literal["pending"] = "pending"
    eligible_for_independent_human_benchmark_claim: Literal[False] = False
    frozen_for_training: Literal[True] = True
    fault_spec: FaultSpec | None = None


class NativeReactHardCase(BaseModel):
    metadata: BenchmarkMetadata
    case: FullAgentLoopCase

    @model_validator(mode="after")
    def validate_identity_and_fault(self) -> "NativeReactHardCase":
        if self.case.case_id != self.metadata.statistical_unit_id:
            raise ValueError("case_id must equal statistical_unit_id")
        if (self.metadata.family == "tool_recovery") != (self.metadata.fault_spec is not None):
            raise ValueError("only tool_recovery cases may define fault_spec")
        return self


def _draft(
    case_id: str,
    family: str,
    user_input: str,
    *,
    expected_slots: dict[str, object],
    extra_artifacts: list[str] | None = None,
    extra_actions: list[str] | None = None,
) -> FullAgentLoopCase:
    return FullAgentLoopCase(
        case_id=case_id,
        suite="expanded",
        slice=family,
        user_input=user_input,
        required_artifacts=[*CORE_ARTIFACTS, *(extra_artifacts or [])],
        required_actions=[*CORE_ACTIONS, *(extra_actions or [])],
        expected_slots=expected_slots,
    )


def _external(
    case_id: str,
    family: str,
    user_input: str,
    *,
    expected_slots: dict[str, object],
    action: str,
    artifact: str,
) -> FullAgentLoopCase:
    case = _draft(
        case_id,
        family,
        user_input,
        expected_slots=expected_slots,
        extra_actions=[action],
        extra_artifacts=[artifact],
    )
    return case.model_copy(
        update={
            "expected_outcome": "draft_or_safe_termination",
            "safe_required_actions": [action, "propose_tradeoff"],
        }
    )


def _base_slots(profile: CityProfile, index: int, *, days: int = 3) -> dict[str, object]:
    return {
        "destination": profile.city,
        "travel_days": days,
        "travelers_count": 1 + index % 3,
        "total_budget": 2600 + (index % 6) * 700,
    }


def _build_family_case(
    family: str,
    profile: CityProfile,
    index: int,
    case_id: str,
) -> tuple[FullAgentLoopCase, str, str, FaultSpec | None]:
    days = 2 + index % 4
    travelers = 1 + index % 3
    budget = 2600 + (index % 6) * 700
    slots = _base_slots(profile, index, days=days)
    variant = index % 4
    pattern_id = f"{family}-pattern-{variant + 1}"

    if family == "clarification":
        prompts = [
            f"9月中旬想出去玩{days}天，{travelers}个人，预算{budget}元，偏爱{profile.cuisine}，帮我规划。",
            f"准备去{profile.city}，总预算{budget}元，想看{profile.must_visit}，但是还没决定玩几天。",
            f"想坐高铁去{profile.city}玩{days}天，最好中午前到，出发城市我还没告诉你。",
            f"国庆想带家里人出去走走，想吃{profile.cuisine}，具体去哪和玩多久都还没定。",
        ]
        missing = [["destination"], ["travel_days"], ["origin"], ["destination", "travel_days"]][
            variant
        ]
        expected = {
            0: {"travel_days": days, "travelers_count": travelers, "total_budget": budget},
            1: {"destination": profile.city, "total_budget": budget},
            2: {"destination": profile.city, "travel_days": days},
            3: {},
        }[variant]
        return (
            FullAgentLoopCase(
                case_id=case_id,
                suite="expanded",
                slice=family,
                user_input=prompts[variant],
                expected_outcome="clarification",
                expected_missing=missing,
                expected_slots=expected,
            ),
            f"missing-{'-'.join(missing)}",
            pattern_id,
            None,
        )

    if family == "poi_grounding":
        prompts = [
            f"{days}天去{profile.city}，{profile.must_visit}必须安排，但不要去{profile.alternative}，预算{budget}元。",
            f"帮{travelers}个人做一份{profile.city}{days}日行程，核心是{profile.must_visit}和{profile.cuisine}，节奏别太赶。",
            f"第一次去{profile.city}玩{days}天，预算{budget}，优先历史文化，{profile.must_visit}一定要有，景点信息先核实。",
            f"在{profile.city}待{days}天，想围绕{profile.must_visit}规划，排除{profile.alternative}，相邻地点通勤别超过45分钟。",
        ]
        slots = {
            "destination": profile.city,
            "travel_days": days,
            "must_visit": [profile.must_visit],
        }
        if variant in {0, 2}:
            slots["total_budget"] = budget
        if variant == 1:
            slots["travelers_count"] = travelers
        if variant in {0, 3}:
            slots["must_not_visit"] = [profile.alternative]
        variants = [
            "must-visit-and-exclusion",
            "preference-and-pace",
            "fact-grounding",
            "transit-cap-and-exclusion",
        ]
        return (
            _draft(case_id, family, prompts[variant], expected_slots=slots),
            variants[variant],
            pattern_id,
            None,
        )

    if family == "current_information":
        prompts = [
            f"{profile.city}{days}日游，出发前先查天气；如果下雨就多排室内场馆，预算{budget}元。",
            f"计划去{profile.city}的{profile.must_visit}，请先核实我出发那天是否开放和最新营业时间，再排{days}天行程。",
            f"去{profile.city}玩{days}天，想看当周正在举办的演出或展览，活动、日期和场馆都先搜索确认。",
            f"到{profile.city}第一晚22点后才有空，查一家那时仍营业的{profile.cuisine}餐厅，再安排{days}天行程。",
        ]
        slots = {"destination": profile.city, "travel_days": days}
        if variant == 0:
            slots["total_budget"] = budget
        actions = [
            "get_weather",
            "search_current_info",
            "search_current_info",
            "search_current_info",
        ]
        artifacts = [
            "weather_snapshot",
            "current_info_search",
            "event_search_result",
            "current_info_search",
        ]
        variants = ["weather", "opening-hours", "event", "late-restaurant"]
        return (
            _external(
                case_id,
                family,
                prompts[variant],
                expected_slots=slots,
                action=actions[variant],
                artifact=artifacts[variant],
            ),
            variants[variant],
            pattern_id,
            None,
        )

    if family == "transport_grounding":
        mode = "飞机" if variant % 2 else "高铁"
        arrival = ["11点前", "中午12点前", "下午3点前", "晚上8点前"][variant]
        prompt = f"从{profile.origin}坐{mode}去{profile.city}玩{days}天，{travelers}个人预算{budget}元，查一个{arrival}到达的真实班次再规划。"
        slots.update(
            {
                "origin": profile.origin,
                "transport_modes_requested": ["flight" if mode == "飞机" else "train"],
            }
        )
        return (
            _external(
                case_id,
                family,
                prompt,
                expected_slots=slots,
                action="search_transport",
                artifact="transport_search_result",
            ),
            f"{mode}-{arrival}",
            pattern_id,
            None,
        )

    if family == "hard_constraint_conflict":
        prompts = [
            f"一个人去{profile.city}玩{days}天，总预算只有{500 + index * 15}元，{profile.must_visit}必须去，住宿交通吃饭都算在内。",
            f"只在{profile.city}停留1天，但{profile.must_visit}、{profile.alternative}和另外6个热门景点全都必须去，不能删。",
            f"去{profile.city}{days}天，每天只接受10点到15点活动，还要求安排至少6个景点且相邻通勤不超过15分钟。",
            f"{travelers}个人去{profile.city}{days}天，预算{900 + index * 20}元，必须住五星酒店并安排{profile.must_visit}，不能超预算。",
        ]
        conflict_slots = {"destination": profile.city, "travel_days": 1 if variant == 1 else days}
        case = _draft(case_id, family, prompts[variant], expected_slots=conflict_slots)
        return (
            case.model_copy(
                update={
                    "expected_outcome": "draft_or_safe_termination",
                    "safe_required_actions": ["propose_tradeoff"],
                }
            ),
            [
                "tight-budget",
                "too-many-must-visits",
                "narrow-time-window",
                "luxury-budget-conflict",
            ][variant],
            pattern_id,
            None,
        )

    if family == "accessibility":
        prompts = [
            f"带轮椅使用者去{profile.city}玩{days}天，预算{budget}元，优先无障碍景点，每天步行不超过40分钟。",
            f"带两位65岁父母去{profile.city}{days}天，共3人，少走路不要赶，{profile.must_visit}想去。",
            f"夫妻去{profile.city}玩{days}天，其中有孕妇，预算{budget}元，疲劳度要低，行程宽松。",
            f"一家三口带8岁孩子去{profile.city}{days}天，预算{budget}元，希望亲子友好并安排{profile.must_visit}。",
        ]
        variants = ["wheelchair", "elderly", "pregnant", "child"]
        extras = [
            {
                "destination": profile.city,
                "travel_days": days,
                "total_budget": budget,
                "has_wheelchair": True,
                "max_walk_minutes": 40,
            },
            {
                "destination": profile.city,
                "travel_days": days,
                "travelers_count": 3,
                "has_elderly": True,
                "pace": "relaxed",
            },
            {
                "destination": profile.city,
                "travel_days": days,
                "travelers_count": 2,
                "total_budget": budget,
                "has_pregnant": True,
                "fatigue_preference": "low",
                "pace": "relaxed",
            },
            {
                "destination": profile.city,
                "travel_days": days,
                "travelers_count": 3,
                "total_budget": budget,
                "has_children": True,
            },
        ]
        slots = extras[variant]
        return (
            _draft(case_id, family, prompts[variant], expected_slots=slots),
            variants[variant],
            pattern_id,
            None,
        )

    if family == "tool_recovery":
        fault_actions = ["search_pois", "get_poi_detail", "get_route_matrix", "search_current_info"]
        fault_types = ["timeout", "empty_result", "rate_limit", "stale_data"]
        action = fault_actions[variant]
        prompt = f"去{profile.city}玩{days}天，预算{budget}元，想去{profile.must_visit}并体验{profile.cuisine}；即使查询暂时失败也请给出可验证的处理结果。"
        extra_actions = ["search_current_info"] if action == "search_current_info" else []
        extra_artifacts = ["current_info_search"] if action == "search_current_info" else []
        recovery_slots = {
            "destination": profile.city,
            "travel_days": days,
            "total_budget": budget,
            "must_visit": [profile.must_visit],
        }
        case = _draft(
            case_id,
            family,
            prompt,
            expected_slots=recovery_slots,
            extra_actions=extra_actions,
            extra_artifacts=extra_artifacts,
        )
        case = case.model_copy(
            update={
                "expected_outcome": "draft_or_safe_termination",
                "safe_required_actions": [action, "propose_tradeoff"],
            }
        )
        fault = FaultSpec(
            action=action,
            fault_type=fault_types[variant],
            recoverable=variant != 3,
            expected_behavior="change arguments or use a grounded fallback; never fabricate success",
        )
        return case, f"{action}-{fault_types[variant]}", pattern_id, fault

    if family == "revision":
        initial = f"去{profile.city}玩{days}天，预算{budget}元，喜欢文化和{profile.cuisine}，{profile.must_visit}想去，先给我一版正常节奏的行程。"
        revisions = [
            f"上一版太赶了，增加到{days + 1}天并改成轻松节奏。",
            f"预算要降到{max(1200, budget - 1000)}元，保留{profile.must_visit}，其他可以调整。",
            f"把{profile.must_visit}设为必去，同时不要安排{profile.alternative}。",
            "临时多来一个人，改成共4个人，预算不变，重新排程。",
        ]
        revision_slots = {
            "destination": profile.city,
            "travel_days": days,
            "total_budget": budget,
            "must_visit": [profile.must_visit],
        }
        case = _draft(case_id, family, initial, expected_slots=revision_slots)
        updates: dict[str, Any] = {
            "expected_outcome": "revision",
            "revision_input": revisions[variant],
        }
        if variant == 0:
            updates.update(
                expected_revision_hard={"travel_days": days + 1},
                expected_revision_soft={"pace": "relaxed"},
            )
        elif variant == 1:
            updates.update(
                expected_revision_hard={
                    "total_budget": max(1200, budget - 1000),
                    "must_visit": [profile.must_visit],
                }
            )
        elif variant == 2:
            updates.update(
                expected_revision_hard={"must_visit": [profile.must_visit]},
                expected_revision_exclusions=[profile.alternative],
            )
        else:
            updates.update(expected_revision_hard={"travelers_count": 4})
        return (
            case.model_copy(update=updates),
            ["extend-and-relax", "lower-budget", "must-and-exclude", "traveler-count-change"][
                variant
            ],
            pattern_id,
            None,
        )

    if family == "safe_termination":
        prompts = [
            f"明天去{profile.city}，{profile.must_visit}必须参观；如果临时闭馆也不能换景点，请核实后安排。",
            f"从{profile.origin}出发，要求一小时内到{profile.city}并马上参观{profile.must_visit}，任何条件都不能改。",
            f"{300 + index * 10}元要包下{profile.city}{days}天的五星酒店、往返交通和全部门票，不接受加预算。",
            f"去{profile.city}只停留半天，必须完成{profile.must_visit}和{profile.alternative}等8个地点，不允许删减。",
        ]
        action = (
            "search_current_info"
            if variant == 0
            else "search_transport"
            if variant == 1
            else "search_pois"
        )
        artifact = (
            "current_info_search"
            if variant == 0
            else "transport_search_result"
            if variant == 1
            else "poi_candidate_set"
        )
        case = _external(
            case_id,
            family,
            prompts[variant],
            expected_slots={"destination": profile.city},
            action=action,
            artifact=artifact,
        )
        return (
            case,
            [
                "closure-no-alternative",
                "impossible-arrival",
                "impossible-budget",
                "impossible-capacity",
            ][variant],
            pattern_id,
            None,
        )

    if family == "noisy_multilingual":
        prompts = [
            f"plz plan {profile.city} {days}天, budget {budget} RMB, MUST={profile.must_visit}, no {profile.alternative}，别太赶哈",
            f"{profile.city}自由行 {days}d / {travelers}ppl / ￥{budget}，想吃{profile.cuisine}，{profile.must_visit}必去，帮我排下",
            f"下个月粗发{profile.city}，玩{days}天，预算大概{budget}，想去{profile.must_visit}，不要那种特种兵行程",
            f"Need a {days}-day trip to {profile.city} for {travelers}; total budget is CNY {budget}. Please verify {profile.must_visit} before scheduling.",
        ]
        slots = {
            "destination": profile.city,
            "travel_days": days,
            "total_budget": budget,
            "must_visit": [profile.must_visit],
        }
        if variant in {1, 3}:
            slots["travelers_count"] = travelers
        return (
            _draft(case_id, family, prompts[variant], expected_slots=slots),
            ["code-switch", "abbreviation", "typo-colloquial", "english-request"][variant],
            pattern_id,
            None,
        )

    raise ValueError(f"unsupported family: {family}")


def build_cases() -> list[NativeReactHardCase]:
    rows: list[NativeReactHardCase] = []
    for family in FAMILIES:
        for index, profile in enumerate(CITY_PROFILES):
            case_id = f"nrhb-v1-{family.replace('_', '-')}-{index + 1:03d}-{profile.slug}"
            case, variant, pattern_id, fault = _build_family_case(family, profile, index, case_id)
            difficulty: Literal["L2", "L3", "L4"] = (
                "L2"
                if family in {"clarification", "poi_grounding"}
                else "L4"
                if family in {"tool_recovery", "hard_constraint_conflict", "safe_termination"}
                else "L3"
            )
            metadata = BenchmarkMetadata(
                statistical_unit_id=case_id,
                split="dev" if profile.city in DEV_CITIES else "test",
                family=family,
                difficulty=difficulty,
                city=profile.city,
                city_cluster=profile.cluster,
                challenge_variant=variant,
                cluster_id=f"{family}:{variant}",
                prompt_pattern_id=pattern_id,
                fault_spec=fault,
            )
            rows.append(NativeReactHardCase(metadata=metadata, case=case))
    return rows


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def benchmark_hash(cases: Iterable[NativeReactHardCase]) -> str:
    return canonical_hash([case.model_dump(mode="json") for case in cases])


def _prompt(case: NativeReactHardCase) -> str:
    values = [case.case.user_input]
    if case.case.revision_input:
        values.append(case.case.revision_input)
    return "\n".join(values)


def _near_duplicate_pairs(texts: list[str], *, threshold: float) -> list[dict[str, Any]]:
    grams = [character_ngrams(text) for text in texts]
    findings = []
    for left in range(len(texts)):
        for right in range(left + 1, len(texts)):
            score = jaccard_similarity(grams[left], grams[right])
            if score >= threshold:
                findings.append({"left": left, "right": right, "similarity": round(score, 6)})
    return findings


def audit_cases(
    cases: list[NativeReactHardCase],
    *,
    forbidden_prompts: Iterable[str] = (),
    near_duplicate_threshold: float = 0.90,
) -> dict[str, Any]:
    prompts = [_prompt(case) for case in cases]
    normalized = [normalize_text(prompt) for prompt in prompts]
    fingerprints = [
        canonical_hash(
            {
                "prompt": normalized[index],
                "expected": case.case.model_dump(mode="json", exclude={"case_id"}),
                "fault": case.metadata.fault_spec.model_dump(mode="json")
                if case.metadata.fault_spec
                else None,
            }
        )
        for index, case in enumerate(cases)
    ]
    forbidden = sorted(
        {normalize_text(prompt) for prompt in forbidden_prompts if normalize_text(prompt)}
    )
    forbidden_exact = set(normalized) & set(forbidden)
    forbidden_near: list[dict[str, Any]] = []
    if forbidden:
        forbidden_grams = [character_ngrams(text) for text in forbidden]
        for case_index, prompt in enumerate(normalized):
            prompt_grams = character_ngrams(prompt)
            best = max(
                (jaccard_similarity(prompt_grams, item) for item in forbidden_grams),
                default=0.0,
            )
            if best >= near_duplicate_threshold and prompt not in forbidden_exact:
                forbidden_near.append(
                    {
                        "case_hash": canonical_hash(cases[case_index].case.case_id),
                        "similarity": round(best, 6),
                    }
                )
    internal_near = _near_duplicate_pairs(normalized, threshold=near_duplicate_threshold)
    family_counts = Counter(case.metadata.family for case in cases)
    split_counts = Counter(case.metadata.split for case in cases)
    difficulty_counts = Counter(case.metadata.difficulty for case in cases)
    outcome_counts = Counter(case.case.expected_outcome for case in cases)
    dev_cities = {case.metadata.city for case in cases if case.metadata.split == "dev"}
    test_cities = {case.metadata.city for case in cases if case.metadata.split == "test"}
    gates = {
        "total_200": len(cases) == 200,
        "dev_40_test_160": split_counts == {"dev": 40, "test": 160},
        "ten_families_twenty_each": family_counts == {family: 20 for family in FAMILIES},
        "unique_statistical_units": len({case.metadata.statistical_unit_id for case in cases})
        == len(cases),
        "unique_normalized_prompts": len(set(normalized)) == len(cases),
        "unique_case_fingerprints": len(set(fingerprints)) == len(cases),
        "dev_test_city_disjoint": not (dev_cities & test_cities),
        "forty_semantic_clusters": len({case.metadata.cluster_id for case in cases}) == 40,
        "no_internal_near_duplicates": not internal_near,
        "no_forbidden_exact_overlap": not forbidden_exact,
        "no_forbidden_near_overlap": not forbidden_near,
        "all_cases_excluded_from_training": all(
            case.metadata.frozen_for_training for case in cases
        ),
        "human_claim_disabled": all(
            not case.metadata.eligible_for_independent_human_benchmark_claim for case in cases
        ),
    }
    return {
        "schema_version": "native-react-independent-hard-audit.v1",
        "passed": all(gates.values()),
        "gates": gates,
        "counts": {
            "total": len(cases),
            "split": dict(sorted(split_counts.items())),
            "family": dict(sorted(family_counts.items())),
            "difficulty": dict(sorted(difficulty_counts.items())),
            "outcome": dict(sorted(outcome_counts.items())),
            "cities": len({case.metadata.city for case in cases}),
            "semantic_clusters": len({case.metadata.cluster_id for case in cases}),
            "forbidden_prompts_scanned": len(forbidden),
        },
        "split_isolation": {
            "dev_cities": sorted(dev_cities),
            "test_city_count": len(test_cities),
            "city_overlap": sorted(dev_cities & test_cities),
        },
        "duplicate_audit": {
            "threshold": near_duplicate_threshold,
            "internal_near_duplicate_pairs": internal_near,
            "forbidden_exact_overlap_count": len(forbidden_exact),
            "forbidden_near_overlap": forbidden_near,
        },
        "dataset_sha256": benchmark_hash(cases),
        "claim_boundary": (
            "The 200 rows are distinct frozen statistical units generated by deterministic "
            "composition across 20 cities and 40 semantic clusters. They are not independently "
            "human-authored or double-annotated, so task-cluster statistics are mandatory."
        ),
    }


def existing_full_loop_prompts() -> list[str]:
    return [
        "\n".join(filter(None, [case.user_input, case.revision_input]))
        for case in build_frozen_cases()
    ]


def write_benchmark(
    output_dir: Path,
    *,
    forbidden_prompts: Iterable[str] = (),
    git_commit: str = "unknown",
) -> dict[str, Any]:
    cases = build_cases()
    audit = audit_cases(
        cases,
        forbidden_prompts=[*existing_full_loop_prompts(), *forbidden_prompts],
    )
    if not audit["passed"]:
        raise ValueError(f"native ReAct hard benchmark audit failed: {audit}")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_hashes = {}
    for split in ("dev", "test"):
        path = output_dir / f"{split}.jsonl"
        selected = [case for case in cases if case.metadata.split == split]
        path.write_text(
            "".join(case.model_dump_json() + "\n" for case in selected),
            encoding="utf-8",
        )
        file_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "frozen Native ReAct hard benchmark; excluded from all training and checkpoint selection",
        "git_commit": git_commit,
        "files": file_hashes,
        "audit": audit,
        "evaluation_contract": {
            "primary_unit": "unique statistical_unit_id",
            "repeated_rollouts": "diagnostic stability only",
            "required_statistics": "task-level paired delta plus cluster bootstrap confidence interval",
            "checkpoint_selection_split": "dev",
            "final_claim_split": "test",
            "test_reuse_for_tuning": False,
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "BenchmarkMetadata",
    "CITY_PROFILES",
    "DEV_CITIES",
    "FAMILIES",
    "FaultSpec",
    "NativeReactHardCase",
    "SCHEMA_VERSION",
    "audit_cases",
    "benchmark_hash",
    "build_cases",
    "existing_full_loop_prompts",
    "write_benchmark",
]
