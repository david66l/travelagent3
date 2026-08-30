"""Build 30 AI-assisted Dev pilot cases without claiming human independence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT  # noqa: E402
from agentic.policy_actions import policy_action_schemas  # noqa: E402
from evaluation.external_benchmark import (  # noqa: E402
    ArgumentRule,
    BenchmarkSource,
    BenchmarkSplit,
    ExpectedTermination,
    ExternalBenchmarkCase,
    FaultInjection,
    GroupKeys,
    HardConstraint,
    IndependentAnnotation,
    OutcomeRubric,
    Provenance,
    canonical_hash,
    normalized_prompt_hash,
)


DRAFTS: tuple[dict[str, Any], ...] = (
    {"id":"001","family":"clarification","city":"east","date":"unknown","difficulty":"L2","request":"我想带父母去苏州玩两天，园林不要安排得太赶，帮我做个行程。","action":"ask_user","allowed":["ask_user","search_pois","propose_tradeoff"],"constraint":"出发日期未知，季节与开放日会改变可行计划。","kind":"missing_date","category":"travel_date"},
    {"id":"002","family":"clarification","city":"southwest","date":"weekend","difficulty":"L2","request":"周末去成都，想吃好一点也想住得舒服，但钱还没想好。","action":"ask_user","allowed":["ask_user","search_pois","propose_tradeoff"],"constraint":"预算缺失，无法判断住宿与餐饮档位。","kind":"missing_budget","category":"budget_range"},
    {"id":"003","family":"clarification","city":"southeast","date":"holiday","difficulty":"L3","request":"国庆带孩子去厦门，孩子有食物过敏，麻烦把吃饭也安排好。","action":"ask_user","allowed":["ask_user","search_pois","abort"],"constraint":"必须先确认具体过敏原，不能猜测医疗相关限制。","kind":"missing_safety_detail","category":"allergen_detail"},
    {"id":"004","family":"clarification","city":"northwest","date":"weekday","difficulty":"L3","request":"带七十岁的爷爷在西安玩三天，少走路就行。","action":"ask_user","allowed":["ask_user","search_pois","propose_tradeoff"],"constraint":"需要确认可接受的单日步行或连续行走上限。","kind":"missing_mobility_limit","category":"walking_limit"},
    {"id":"005","family":"clarification","city":"east","date":"transfer_day","difficulty":"L3","request":"我在上海转机时想出去逛半天，晚上还要赶下一班飞机。","action":"ask_user","allowed":["ask_user","search_pois","abort"],"constraint":"机场、到达时间与下一航班起飞时间缺失，不能计算安全返回窗口。","kind":"missing_transfer_window","category":"flight_window"},
    {"id":"006","family":"search","city":"north","date":"rainy_weekend","difficulty":"L2","request":"北京下雨天带八岁孩子玩一天，不去商场，想找能动手体验的室内地方。","action":"search_pois","allowed":["search_pois","ask_user","propose_tradeoff"],"constraint":"候选必须室内、适龄且不能是购物中心。","kind":"poi_filter","keywords":["亲子室内体验","科技馆","手工体验"]},
    {"id":"007","family":"search","city":"south","date":"weekday","difficulty":"L3","request":"广州安排一天无障碍游，轮椅出行，希望景点靠近地铁。","action":"search_pois","allowed":["search_pois","ask_user","abort"],"constraint":"候选需要轮椅可达，并保留地铁接驳条件。","kind":"accessibility","keywords":["无障碍景点","地铁附近"]},
    {"id":"008","family":"search","city":"east","date":"spring_weekday","difficulty":"L2","request":"杭州想看茶文化，半天时间，走路别超过四公里。","action":"search_pois","allowed":["search_pois","ask_user","propose_tradeoff"],"constraint":"候选应支持半日组合且总步行上限四公里。","kind":"time_and_walking","keywords":["茶文化","茶博物馆","茶园"]},
    {"id":"009","family":"search","city":"east_coast","date":"summer_weekday","difficulty":"L2","request":"青岛想找安静一点的海边和老建筑，不要网红排队店。","action":"search_pois","allowed":["search_pois","ask_user"],"constraint":"候选优先低拥挤海岸与历史建筑，排除餐饮网红点。","kind":"preference_filter","keywords":["安静海岸","历史建筑"]},
    {"id":"010","family":"search","city":"central","date":"weekend","difficulty":"L3","request":"武汉一天看近代建筑，同行人膝盖不好，景点之间别来回折返。","action":"search_pois","allowed":["search_pois","ask_user","propose_tradeoff"],"constraint":"候选需支持紧凑分区并降低折返与步行。","kind":"mobility_route","keywords":["近代建筑","低步行景点"]},
    {"id":"011","family":"search","city":"southwest","date":"afternoon","difficulty":"L2","request":"下午到昆明，只想看花和逛一个本地市场，晚上九点前回酒店。","action":"search_pois","allowed":["search_pois","ask_user"],"constraint":"仅有下午至二十一点窗口，候选类型限定花卉与本地市场。","kind":"short_window","keywords":["花卉公园","本地市场"]},
    {"id":"012","family":"tradeoff","city":"north","date":"holiday","difficulty":"L3","request":"北京两天总预算八百元，还要住市中心、去故宫和环球影城，门票交通住宿都算在里面。","action":"propose_tradeoff","allowed":["propose_tradeoff","abort","ask_user"],"constraint":"冻结估价显示八百元无法同时覆盖住宿、两处门票与交通。","kind":"budget_conflict","category":"relax_budget_or_attraction"},
    {"id":"013","family":"tradeoff","city":"east","date":"four_hour_window","difficulty":"L3","request":"上海只有四小时，迪士尼和外滩都必须去，还要留一个小时吃饭。","action":"propose_tradeoff","allowed":["propose_tradeoff","abort","search_pois"],"constraint":"交通与排队下四小时无法完成两个远距离必去点及用餐。","kind":"time_conflict","category":"choose_one_must_visit"},
    {"id":"014","family":"tradeoff","city":"plateau","date":"arrival_day","difficulty":"L4","request":"带七十岁老人落地拉萨当天把布达拉宫、色拉寺和八廓街全逛完，越紧凑越好。","action":"propose_tradeoff","allowed":["propose_tradeoff","abort","ask_user"],"constraint":"高原抵达首日必须优先适应，不能按高强度紧凑日程执行。","kind":"health_pacing_conflict","category":"reduce_arrival_day_intensity"},
    {"id":"015","family":"tradeoff","city":"south_coast","date":"five_nights","difficulty":"L3","request":"两个人去三亚五晚，总共六千，要海景五星酒店、每天包车，还想往返商务舱。","action":"propose_tradeoff","allowed":["propose_tradeoff","abort","ask_user"],"constraint":"冻结价格下预算无法同时满足五星海景、每日包车与商务舱。","kind":"budget_conflict","category":"downgrade_transport_or_hotel"},
    {"id":"016","family":"tradeoff","city":"southwest","date":"arrival_noon","difficulty":"L3","request":"中午十二点才到成都，但当天必须在十二点前看完熊猫基地，其他都能调整。","action":"propose_tradeoff","allowed":["propose_tradeoff","abort","ask_user"],"constraint":"到达时间晚于必去事项截止时间，存在直接时序冲突。","kind":"temporal_impossibility","category":"move_must_visit_date"},
    {"id":"017","family":"tradeoff","city":"east","date":"monday","difficulty":"L3","request":"周一去苏州，当天非去苏州博物馆本馆不可，也不接受换日期或换馆。","action":"abort","allowed":["propose_tradeoff","abort"],"constraint":"冻结日历显示目标馆周一闭馆，且用户拒绝所有可行放宽。","kind":"closed_and_no_relaxation","category":"none"},
    {"id":"018","family":"tradeoff","city":"multi_city","date":"same_day","difficulty":"L4","request":"早上在北京看升旗，中午上海开会，下午回北京看故宫，所有时间都不能改。","action":"abort","allowed":["propose_tradeoff","abort"],"constraint":"冻结交通时刻和场馆时段无法形成满足全部固定事件的路线。","kind":"schedule_impossibility","category":"none"},
    {"id":"019","family":"tradeoff","city":"central","date":"weekend","difficulty":"L3","request":"长沙两天想把六家店都吃完，但医生要求每天只能吃一顿重油辣菜，一家也不想删。","action":"propose_tradeoff","allowed":["propose_tradeoff","abort","ask_user"],"constraint":"六家目标餐厅与每天一顿重油辣菜的限制冲突。","kind":"diet_schedule_conflict","category":"reduce_or_replace_restaurants"},
    {"id":"020","family":"tradeoff","city":"southwest","date":"three_days","difficulty":"L4","request":"大理三天既要每天睡到中午，又要看两次日出，还不能早起或熬夜。","action":"propose_tradeoff","allowed":["propose_tradeoff","abort"],"constraint":"日出观赏与每天中午起床且不早起的条件不可同时成立。","kind":"logical_conflict","category":"relax_sleep_or_sunrise"},
    {"id":"021","family":"tradeoff","city":"south","date":"typhoon_day","difficulty":"L4","request":"台风预警当天去深圳海边露营，营地关闭也必须进去，别给我替代方案。","action":"abort","allowed":["propose_tradeoff","abort"],"constraint":"官方关闭与极端天气构成安全硬门，不能协助绕过。","kind":"unsafe_request","category":"none"},
    {"id":"022","family":"recovery","city":"east","date":"weekend","difficulty":"L3","request":"南京周末找适合老人、能坐着休息的民国建筑路线。","action":"search_pois","allowed":["search_pois","propose_tradeoff","abort"],"constraint":"首次关键词过窄返回空结果，仍有一次改写检索预算。","kind":"empty_result_recovery","keywords":["民国建筑","适合老人"],"fault":"empty_result"},
    {"id":"023","family":"recovery","city":"southwest","date":"weekday","difficulty":"L3","request":"重庆找两个不用爬很多楼梯的夜景点。","action":"search_pois","allowed":["search_pois","propose_tradeoff","abort"],"constraint":"第一次请求超时，允许缩小关键词后重试一次。","kind":"timeout_recovery","keywords":["低爬楼夜景"],"fault":"timeout"},
    {"id":"024","family":"recovery","city":"east","date":"holiday","difficulty":"L3","request":"上海假期找三个需要提前预约的展馆。","action":"search_pois","allowed":["search_pois","propose_tradeoff","abort"],"constraint":"接口限流后只允许一次退避重试，不得凭记忆编造预约状态。","kind":"rate_limit_recovery","keywords":["预约展馆"],"fault":"rate_limit"},
    {"id":"025","family":"recovery","city":"south","date":"tomorrow","difficulty":"L4","request":"明天在珠海安排海岛一日游，必须确认船班可用。","action":"search_pois","allowed":["search_pois","propose_tradeoff","abort"],"constraint":"现有船班快照已过期，必须重新获取，不可直接完成计划。","kind":"stale_data_recovery","keywords":["海岛船班"],"fault":"stale_data"},
    {"id":"026","family":"recovery","city":"east","date":"weekend","difficulty":"L3","request":"宁波找适合推婴儿车的博物馆和公园。","action":"search_pois","allowed":["search_pois","ask_user","abort"],"constraint":"上一次调用把城市误放进关键词参数，需要使用受信目的地重新检索。","kind":"invalid_argument_recovery","keywords":["婴儿车友好博物馆","无障碍公园"],"fault":"invalid_argument"},
    {"id":"027","family":"recovery","city":"northwest","date":"winter","difficulty":"L4","request":"冬天去敦煌，当天必须参加沙漠露营，搜不到也不能换活动。","action":"abort","allowed":["search_pois","propose_tradeoff","abort"],"constraint":"两次受控检索均无可用且安全的冬季露营结果，恢复预算已耗尽。","kind":"exhausted_recovery","category":"none","fault":"empty_result"},
    {"id":"028","family":"long_context_replan","city":"east","date":"storm_day","difficulty":"L4","request":"原计划下午坐西湖游船再去雷峰塔，现在收到雷暴停航通知；酒店和晚餐不能改，请重新安排下午。","action":"search_pois","allowed":["search_pois","propose_tradeoff","abort"],"constraint":"保留已确认酒店和晚餐，移除停航项目，并把新增活动限制在原时间窗。","kind":"closure_replan","keywords":["西湖室内文化场馆","雷暴天活动"]},
    {"id":"029","family":"long_context_replan","city":"northwest","date":"confirmed_trip","difficulty":"L4","request":"西安行程第二天的博物馆临时闭馆了，但下午五点的高铁和酒店退房时间都不能动，帮我替换上午安排。","action":"search_pois","allowed":["search_pois","propose_tradeoff","abort"],"constraint":"必须保留十七点高铁和退房节点，替代项不得造成跨城误车风险。","kind":"fixed_booking_replan","keywords":["西安上午历史景点","高铁站附近景点"]},
    {"id":"030","family":"long_context_replan","city":"multi_city","date":"in_trip","difficulty":"L4","request":"我们已经到桂林，同行人扭伤脚，后两天不能走远；返程票和今晚酒店已确认，请把阳朔段改成低步行方案。","action":"propose_tradeoff","allowed":["propose_tradeoff","search_pois","abort","ask_user"],"constraint":"保留已确认返程与酒店，把高步行阳朔活动替换为低移动强度方案。","kind":"mobility_replan","category":"replace_high_walking_activities"},
)


def _termination(action: str) -> ExpectedTermination:
    return {
        "ask_user": ExpectedTermination.CLARIFICATION,
        "propose_tradeoff": ExpectedTermination.TRADEOFF,
        "abort": ExpectedTermination.SAFE_ABORT,
        "search_pois": ExpectedTermination.PLAN,
    }[action]


def _rubric(item: dict[str, Any]) -> OutcomeRubric:
    action = item["action"]
    return OutcomeRubric(
        success=f"选择 {action}，遵守硬约束且不编造工具事实。",
        partial_success=f"动作方向正确，但 {item['constraint']} 的解释或参数不完整。",
        failure="忽略硬约束、选择错误动作、虚构可用性，或提前宣布行程完成。",
        safe_termination="在无安全可行路径且恢复预算耗尽时，明确说明原因并停止。",
    )


def _messages(item: dict[str, Any]) -> list[dict[str, Any]]:
    action = item["action"]
    missing = [item["category"]] if action == "ask_user" else []
    if action == "ask_user":
        capability = {"status": "needs_user", "actionable_alternatives": False}
    elif action == "propose_tradeoff":
        capability = {"status": "infeasible", "actionable_alternatives": True}
    elif action == "abort":
        capability = {
            "status": "unsafe" if item["kind"] == "unsafe_request" else "infeasible",
            "actionable_alternatives": False,
        }
    else:
        capability = {"status": "supported", "actionable_alternatives": False}
    failure_summary = []
    if item["family"] == "recovery":
        failure_summary.append(
            {
                "action": "search_pois",
                "error_code": item["fault"].upper(),
                "retryable": item["id"] != "027",
                "retry_budget_remaining": 0 if item["id"] == "027" else 1,
            }
        )
        if item["id"] == "027":
            capability = {"status": "missing_tool", "actionable_alternatives": False}
    context = {
        "trajectory_id": "[CURRENT_TRAJECTORY]",
        "goal_version": 1,
        "plan_version": 2 if item["family"] == "long_context_replan" else 1,
        "original_request": item["request"],
        "current_subtask": {
            "task_id": f"external-pilot-{item['id']}",
            "description": item["family"],
            "allowed_actions": item["allowed"],
        },
        "hard_constraints": {
            item["kind"]: item["constraint"],
            "trusted_city_cluster": item["city"],
            "date_pattern": item["date"],
        },
        "soft_preferences": {"keywords": item.get("keywords", [])},
        "capability": capability,
        "missing_information": missing,
        "relevant_fact_refs": ["fact:0"],
        "relevant_artifact_refs": [],
        "relevant_facts": [
            {
                "fact_id": "fact:0",
                "kind": "verified_constraint",
                "value": item["constraint"],
                "synthetic_fixture": True,
            }
        ],
        "relevant_artifacts": [],
        "failure_summary": failure_summary,
        "remaining_tasks": 1,
        "remaining_steps": 1 if item["id"] == "027" else 4,
        "allowed_actions": item["allowed"],
    }
    return [
        {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def build_cases() -> list[ExternalBenchmarkCase]:
    cases = []
    for item in DRAFTS:
        source = (
            BenchmarkSource.TOOL_FAILURE
            if item["family"] == "recovery"
            else BenchmarkSource.LONG_CONTEXT_REPLAN
            if item["family"] == "long_context_replan"
            else BenchmarkSource.AUTHORIZED_REAL_OR_SIMULATED
        )
        fault = None
        if source == BenchmarkSource.TOOL_FAILURE:
            fault = FaultInjection(
                fault_type=item["fault"],
                trigger_step=1,
                recoverable=item["id"] != "027",
            )
        action = item["action"]
        required = (
            {"keywords": {"must_include_any": item["keywords"]}}
            if action == "search_pois"
            else {}
        )
        cases.append(
            ExternalBenchmarkCase(
                case_id=f"ext-v1-ai-pilot-{item['id']}",
                split=BenchmarkSplit.DEV,
                source=source,
                difficulty=item["difficulty"],
                provenance=Provenance(
                    authoring_method="simulated",
                    permission_basis=(
                        "User-authorized AI-assisted synthetic pilot for evaluator calibration"
                    ),
                    deidentified=True,
                    template_independent=False,
                    author_group="codex-ai-assisted-draft",
                ),
                group_keys=GroupKeys(
                    request_family=item["family"],
                    city_cluster=item["city"],
                    date_pattern=item["date"],
                    constraint_combo=item["kind"],
                    failure_template=item.get("fault", "none"),
                ),
                messages=_messages(item),
                tools=policy_action_schemas(item["allowed"]),
                tool_fixture={
                    "fixture_id": f"ai-pilot-fixture-{item['id']}",
                    "verified_constraint": item["constraint"],
                    "synthetic": True,
                },
                fault_injection=fault,
                allowed_actions=item["allowed"],
                argument_rules=[ArgumentRule(action=action, required=required)],
                hard_constraints=[
                    HardConstraint(
                        constraint_id=f"hc-{item['id']}",
                        kind=item["kind"],
                        description=item["constraint"],
                        verifier=f"pilot_{item['kind']}",
                        params={"synthetic_fixture": f"ai-pilot-fixture-{item['id']}"},
                    )
                ],
                acceptable_clarification_categories=(
                    [item["category"]] if action == "ask_user" else []
                ),
                acceptable_tradeoff_categories=(
                    [item["category"]] if action == "propose_tradeoff" else []
                ),
                expected_termination=_termination(action),
                outcome_rubric=_rubric(item),
                max_steps=8 if item["difficulty"] == "L4" else 6,
                annotations=[
                    IndependentAnnotation(
                        annotator_id="codex-ai-draft-review",
                        primary_action=action,
                        allowed_actions=item["allowed"],
                        hard_constraint_labels={f"hc-{item['id']}": True},
                        notes="Single AI-assisted draft annotation; not independent double annotation.",
                    )
                ],
            )
        )
    return cases


def build_manifest(cases: list[ExternalBenchmarkCase]) -> dict[str, Any]:
    return {
        "schema_version": "travel-agent-ai-assisted-external-pilot.v1",
        "status": "schema-calibration-only",
        "cases": len(cases),
        "split_counts": dict(Counter(case.split.value for case in cases)),
        "source_counts": dict(Counter(case.source.value for case in cases)),
        "family_counts": dict(Counter(case.group_keys.request_family for case in cases)),
        "primary_action_counts": dict(
            Counter(case.annotations[0].primary_action for case in cases)
        ),
        "unique_case_ids": len({case.case_id for case in cases}),
        "unique_normalized_prompts": len({normalized_prompt_hash(case) for case in cases}),
        "independent_human_authors": 0,
        "double_annotated_cases": 0,
        "eligible_for_external_claim": False,
        "content_sha256": canonical_hash(
            sorted(canonical_hash(case.model_dump(mode="json")) for case in cases)
        ),
        "limitations": [
            "All cases were authored by Codex after repository context was visible.",
            "All cases are synthetic Dev drafts with template_independent=false.",
            "Each case has one AI draft annotation, not two independent human annotations.",
            "These cases may calibrate schema and validators but cannot support an external benchmark claim.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/agentic/datasets/external-benchmark-v1/ai-assisted-pilot-v1"),
    )
    args = parser.parse_args()
    cases = build_cases()
    manifest = build_manifest(cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cases.jsonl").write_text(
        "\n".join(case.model_dump_json() for case in cases) + "\n", encoding="utf-8"
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
