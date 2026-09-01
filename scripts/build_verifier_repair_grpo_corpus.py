"""Build a split-safe GRPO corpus for ReAct verifier-repair decisions.

The corpus is derived only from previously accepted Native ReAct train and
validation snapshots. It never reads the independent hard benchmark. Each row
replays a verified research/solver prefix and exposes one production
``review_itinerary`` decision: retry a repairable solve, propose an actionable
trade-off, or abort when the user forbids every safe alternative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import (  # noqa: E402
    GRPOCorpusRow,
    VERIFIER_REPAIR_DECISION_SCHEMA_VERSION,
    load_grpo_corpus,
    preflight_grpo_corpus,
)
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT  # noqa: E402
from agentic.policy_actions import controller_tradeoff_options  # noqa: E402
from agentic.trl_environment import TRLReactEnvironment  # noqa: E402
from agentic.verifier_repair import (  # noqa: E402
    VERIFIER_REPAIR_CONSTRAINT_BY_CODE,
    constraint_handling_for_violation,
    relaxation_options_for_violation,
)
from contracts.constraint_flexibility import ConstraintFlexibilityContract  # noqa: E402


SCHEMA_VERSION = "react-verifier-repair-corpus.v5"
DECISION_SCHEMA_VERSION = VERIFIER_REPAIR_DECISION_SCHEMA_VERSION
PREFIX_ACTIONS = (
    ("retrieve_city_knowledge", {}),
    ("search_pois", None),
    ("get_poi_detail", {}),
)

_TEMPLATES: dict[str, tuple[dict[str, Any], ...]] = {
    "train": (
        {
            "family": "repairable_solver_failure",
            "target_action": "retry_solve",
            "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
            "request_suffix": " 若校验只发现求解策略问题，请先更换策略重算一次，不要直接放弃。",
            "violation": "当前排程超出每日活动结束边界；允许使用贪心策略进行一次有界重算",
            "grounding_phrases": ["每日活动结束边界", "有界重算"],
            "expected_arguments": {"strategy": "greedy"},
        },
        {
            "family": "repairable_solver_failure",
            "target_action": "retry_solve",
            "violation_code": "FIXED_EVENT_TIME_MISMATCH",
            "request_suffix": " 如果只是排程器卡在当前策略，可切换一次算法重新排程。",
            "violation": "当前CP-SAT排程与预约时间冲突；替换为greedy策略仍可能修复",
            "grounding_phrases": ["预约时间冲突", "greedy"],
            "expected_arguments": {"strategy": "greedy"},
        },
        {
            "family": "actionable_tradeoff",
            "target_action": "propose_tradeoff",
            "violation_code": "TOTAL_BUDGET_EXCEEDED",
            "request_suffix": " 若预算与景点数量冲突，我接受提高预算或减少一个景点，请先让我选择。",
            "violation": "当前行程超出总预算300元，不能直接通过硬约束校验",
            "grounding_phrases": ["超出总预算300元", "硬约束"],
            "require_options": True,
            "supervised_options": ["提高总预算", "减少一个非必去景点"],
        },
        {
            "family": "actionable_tradeoff",
            "target_action": "propose_tradeoff",
            "violation_code": "ACTIVITY_TIME_OVERLAP",
            "request_suffix": " 若固定活动和交通时间无法同时满足，可以给我改时间或删活动的选项。",
            "violation": "固定活动与锁定交通时段重叠40分钟，现有安排不可执行",
            "grounding_phrases": ["时段重叠40分钟", "不可执行"],
            "require_options": True,
            "supervised_options": ["调整固定活动时间", "删除一个非必要活动"],
        },
        {
            "family": "necessary_abort",
            "target_action": "abort",
            "violation_code": "ACTIVITY_TIME_OVERLAP",
            "request_suffix": " 两场已付款活动的时间都不能改，也不接受取消；若冲突无解就停止。",
            "violation": "两场固定活动时段重叠，且用户拒绝改期或取消任一活动",
            "grounding_phrases": ["固定活动时段重叠", "拒绝改期"],
        },
        {
            "family": "necessary_abort",
            "target_action": "abort",
            "violation_code": "TOTAL_BUDGET_EXCEEDED",
            "request_suffix": " 预算和全部必去项都已锁定，不能加钱或删减；若超支无解就停止。",
            "violation": "全部必去项的最低费用仍超过固定总预算，用户拒绝加预算或删减",
            "grounding_phrases": ["超过固定总预算", "拒绝加预算"],
        },
    ),
    "validation": (
        {
            "family": "repairable_solver_failure",
            "target_action": "retry_solve",
            "violation_code": "ACTIVITY_TIME_OVERLAP",
            "request_suffix": " 若验证表明只是排程策略不合适，请允许一次替代策略重算。",
            "violation": "当前求解策略造成预订时段重叠；一次greedy重排可能消除冲突",
            "grounding_phrases": ["预订时段重叠", "greedy重排"],
            "expected_arguments": {"strategy": "greedy"},
        },
        {
            "family": "actionable_tradeoff",
            "target_action": "propose_tradeoff",
            "violation_code": "TOTAL_BUDGET_EXCEEDED",
            "request_suffix": " 若费用与必去项冲突，可以让我在加预算和删除非必去项之间选择。",
            "violation": "保留全部必去项后预算缺口为260元，当前计划无法硬通过",
            "grounding_phrases": ["预算缺口为260元", "无法硬通过"],
            "require_options": True,
            "supervised_options": ["增加260元预算", "删除一个非必去景点"],
        },
        {
            "family": "necessary_abort",
            "target_action": "abort",
            "violation_code": "MAX_TRANSIT_EXCEEDED",
            "request_suffix": " 目的地与每日交通上限都不能改；没有更短路线时直接结束。",
            "violation": "唯一可行路线仍超过每日交通上限，且用户拒绝删除或替换目的地",
            "grounding_phrases": ["超过每日交通上限", "拒绝删除"],
        },
    ),
    "test": (
        {
            "family": "repairable_solver_failure",
            "target_action": "retry_solve",
            "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
            "request_suffix": " 如果硬校验只卡在调度方式，可以换一种方式再算一遍。",
            "violation": "原调度使最后一站超出每日活动结束边界；授权一次greedy策略重排",
            "grounding_phrases": ["超出每日活动结束边界", "greedy策略重排"],
            "expected_arguments": {"strategy": "greedy"},
        },
        {
            "family": "actionable_tradeoff",
            "target_action": "propose_tradeoff",
            "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
            "request_suffix": " 若时长不够，我可以选择延长一天或取消一个普通景点。",
            "violation": "当前可用时长少90分钟，无法同时保留全部普通景点",
            "grounding_phrases": ["少90分钟", "无法同时保留"],
            "require_options": True,
            "supervised_options": ["延长一天", "取消一个普通景点"],
        },
        {
            "family": "necessary_abort",
            "target_action": "abort",
            "violation_code": "FIXED_EVENT_TIME_MISMATCH",
            "request_suffix": " 两个固定活动的日期和时间都不能改；无法同时满足时直接停止。",
            "violation": "固定活动被排到错误时刻，且所有相关时间约束均已锁定",
            "grounding_phrases": ["排到错误时刻", "时间约束均已锁定"],
        },
    ),
}

# The base corpus deliberately keeps a small template surface for smoke tests.
# Formal post-training needs paraphrase diversity, otherwise the policy can
# memorize one verifier sentence instead of learning the semantic boundary
# between retry, trade-off, and safe termination.  These additions use only
# training semantics and keep validation/test wording sealed.
_SEMANTIC_DIVERSE_TRAIN_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " 若约束仍然可满足而失败来自排序方式，请换一种求解策略再算一次。",
        "violation": "当前排序使活动超出每日结束边界，可用贪心算法调整顺序回到边界内",
        "grounding_phrases": ["超出每日结束边界", "调整顺序回到边界内"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "MAX_TRANSIT_EXCEEDED",
        "request_suffix": " 如果只是当前算法没有找到可行顺序，允许切换算法重试一轮。",
        "violation": "交通衔接被当前访问顺序拉长；更换greedy策略后仍有机会找到可行排程",
        "grounding_phrases": ["当前访问顺序", "可行排程"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "FIXED_EVENT_TIME_MISMATCH",
        "request_suffix": " 验证器若只否定本轮调度结果，不要改需求，先进行一次有界重算。",
        "violation": "本轮求解策略未找到满足入场时刻的顺序；授权greedy进行一次有限重算",
        "grounding_phrases": ["本轮求解策略", "有限重算"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " 若游玩时间不够，可以让我选择早出发或删除一个普通景点。",
        "violation": "按当前出发时间计算，总游览时长超出可用时间75分钟，需要用户选择调整项",
        "grounding_phrases": ["超出可用时间75分钟", "用户选择"],
        "require_options": True,
        "supervised_options": ["提前出发", "删除一个普通景点"],
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "MAX_TRANSIT_EXCEEDED",
        "request_suffix": " 若步行强度和必去项冲突，请给我打车或减少远距离景点的选择。",
        "violation": "保留全部远距离景点会使每日步行超过用户上限，需要确认交通或景点取舍",
        "grounding_phrases": ["超过用户上限", "景点取舍"],
        "require_options": True,
        "supervised_options": ["增加打车行程", "减少一个远距离景点"],
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "TOTAL_BUDGET_EXCEEDED",
        "request_suffix": " 若住宿位置和房费不能兼得，请让我选择提高住宿预算或换区域。",
        "violation": "指定区域的可用住宿均超过房费上限，必须由用户确认预算或区域调整",
        "grounding_phrases": ["超过房费上限", "用户确认"],
        "require_options": True,
        "supervised_options": ["提高住宿预算", "更换住宿区域"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "MAX_TRANSIT_EXCEEDED",
        "request_suffix": " 每日通勤上限和全部必去项都不能改；无法满足时直接停止。",
        "violation": "保留全部锁定地点后通勤必然超过上限，且用户拒绝删减或更换",
        "grounding_phrases": ["通勤必然超过上限", "拒绝删减"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "FIXED_EVENT_TIME_MISMATCH",
        "request_suffix": " 活动和返程时间都不能改变；两者无法衔接时不要编造方案。",
        "violation": "唯一固定活动与锁定返程时刻冲突，用户拒绝调整任一时间",
        "grounding_phrases": ["锁定返程时刻冲突", "拒绝调整"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " 每日结束时间和所有固定活动都不能改；排不下时结束规划。",
        "violation": "全部固定活动必然超出每日时间边界，且用户拒绝调整活动或时段",
        "grounding_phrases": ["超出每日时间边界", "拒绝调整活动"],
    },
)

# These train-source challenges are intentionally excluded from SFT.  They are
# used only for on-policy exploration so GRPO sees unseen language with real
# within-group variance instead of replaying already memorized demonstrations.
_RL_CHALLENGE_TRAIN_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "ACTIVITY_TIME_OVERLAP",
        "request_suffix": " 若硬约束没有冲突，只是路线搜索失败，请换策略再尝试一次。",
        "violation": "当前路线顺序造成活动时间重叠，但候选仍可重排；可切换greedy继续求解",
        "grounding_phrases": ["活动时间重叠", "候选仍可重排"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " 候选地点和时窗都有效时，算法超时可以用备用策略重算。",
        "violation": "当前行程超出每日活动时间边界，但候选时窗可调整；允许greedy执行备用重算",
        "grounding_phrases": ["超出每日活动时间边界", "候选时窗可调整"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "FIXED_EVENT_TIME_MISMATCH",
        "request_suffix": " 如果验证器标记为可恢复的求解失败，不要询问我，直接有界重试。",
        "violation": "固定活动开始时间被当前排程改变，但重试预算充足，greedy可执行有界重算",
        "grounding_phrases": ["固定活动开始时间", "重试预算充足"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "TOTAL_BUDGET_EXCEEDED",
        "request_suffix": " 若饮食要求使餐费超支，请让我决定加餐费或取消高价餐厅。",
        "violation": "满足全部饮食要求后餐饮费用超过上限180元，需要用户确认费用取舍",
        "grounding_phrases": ["超过上限180元", "费用取舍"],
        "require_options": True,
        "supervised_options": ["增加餐饮预算", "取消一家高价餐厅"],
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "FIXED_EVENT_END_EXCEEDED",
        "request_suffix": " 若返程时间和最后一个景点冲突，请给我删景点或改车次的选项。",
        "violation": "固定活动的计划结束时间超过已知结束时刻，并与返程发车时间冲突",
        "grounding_phrases": ["超过已知结束时刻", "返程发车时间冲突"],
        "require_options": True,
        "supervised_options": ["删除最后一个景点", "改乘更晚车次"],
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " 若无障碍路线会增加时间，可以让我选延长行程或减少活动。",
        "violation": "采用无障碍路线后每日时长增加70分钟，现有活动数量无法全部保留",
        "grounding_phrases": ["每日时长增加70分钟", "无法全部保留"],
        "require_options": True,
        "supervised_options": ["延长每日行程时间", "减少一个普通活动"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "MAX_TRANSIT_EXCEEDED",
        "request_suffix": " 不接受减少远距离必去项或放宽通勤上限；无法满足时终止计划。",
        "violation": "访问全部锁定地点必然超过通勤上限，用户拒绝任何删减或替换",
        "grounding_phrases": ["必然超过通勤上限", "拒绝任何删减"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "FIXED_EVENT_END_EXCEEDED",
        "request_suffix": " 固定活动和返程车次都不能改；活动无法按时结束就终止。",
        "violation": "固定活动结束时间晚于锁定返程时间，用户拒绝改活动或车次",
        "grounding_phrases": ["晚于锁定返程时间", "拒绝改活动"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "TOTAL_BUDGET_EXCEEDED",
        "request_suffix": " 总预算和所有固定支出都不能变；最低成本仍超支时不要继续。",
        "violation": "全部固定支出的最低总额仍超过预算，且用户拒绝提高预算或删项",
        "grounding_phrases": ["最低总额仍超过预算", "拒绝提高预算"],
    },
)


# This is the only training profile that is eligible for a fresh post-training
# run.  It deliberately describes observable constraint facts rather than
# telling the policy to retry, ask, choose, stop, or use a particular solver.
# Each route has six independent wording/violation combinations, and the
# source-specific fields below make the teacher reason vary with the visible
# travel request instead of repeating one verifier sentence for every row.
_NEUTRAL_DIVERSE_TRAIN_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "ACTIVITY_TIME_OVERLAP",
        "request_suffix": " {destination}{travel_days}日行程的活动集合和交通边界保持不变，活动先后仍可在既有条件内调整。",
        "violation": "{destination}{travel_days}日行程中，相邻活动的时段相差45分钟；地点与可用时段仍然有效。",
        "grounding_phrases": ["时段相差45分钟", "可用时段仍然有效"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " {destination}行程的活动集合、预算和固定活动安排保持不变，每日时间窗存在既有范围内的调整空间。",
        "violation": "{destination}{travel_days}日行程的最后一项晚于当日时间边界52分钟；其余约束保持有效。",
        "grounding_phrases": ["晚于当日时间边界52分钟", "其余约束保持有效"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "FIXED_EVENT_TIME_MISMATCH",
        "request_suffix": " {destination}的固定活动仍处于已授权入场范围内，活动集合、预算和每日时间窗保持不变。",
        "violation": "{destination}{travel_days}日行程中，固定活动的开始时刻偏差30分钟；票面允许范围仍有效。",
        "grounding_phrases": ["开始时刻偏差30分钟", "允许范围仍有效"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "TOTAL_BUDGET_EXCEEDED",
        "request_suffix": " {destination}{travel_days}日行程的每日时间窗和通勤上限保持不变；活动安排、活动集合和总预算存在可调整空间。",
        "violation": "{destination}{travel_days}日行程在保留{focus}偏好后，预算缺口为175元。",
        "grounding_phrases": ["预算缺口为175元", "{focus}偏好"],
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "FIXED_EVENT_END_EXCEEDED",
        "request_suffix": " {destination}行程的活动顺序、活动集合和总预算保持不变；固定活动时间与通勤上限存在可调整空间。",
        "violation": "{destination}{travel_days}日行程中，返程前的固定活动晚于已知结束时刻35分钟。",
        "grounding_phrases": ["晚于已知结束时刻35分钟"],
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " {destination}行程的总预算和固定活动时间保持不变；活动集合与每日时间安排存在可调整空间。",
        "violation": "{destination}{travel_days}日行程的当日可用时长不足68分钟，现有活动无法全部保留。",
        "grounding_phrases": ["当日可用时长不足68分钟", "无法全部保留"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "MAX_TRANSIT_EXCEEDED",
        "request_suffix": " {destination}行程的每日时间窗与通勤上限保持不变；固定活动可在既有范围内安排，活动集合和总预算仍有调整空间。",
        "violation": "{destination}{travel_days}日行程保留锁定地点后，通勤超过上限48分钟。",
        "grounding_phrases": ["通勤超过上限48分钟", "锁定地点"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "FIXED_EVENT_TIME_MISMATCH",
        "request_suffix": " {destination}行程的总预算和固定活动时间保持不变；活动顺序与通勤路线可在既有上限内安排，活动集合仍有调整空间。",
        "violation": "{destination}{travel_days}日行程的固定活动与锁定返程时刻相差50分钟。",
        "grounding_phrases": ["锁定返程时刻相差50分钟"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "TOTAL_BUDGET_EXCEEDED",
        "request_suffix": " {destination}行程的总预算和固定活动时间保持不变；活动顺序与通勤路线可在既有上限内安排，活动集合仍有调整空间。",
        "violation": "{destination}{travel_days}日行程的最低费用超出预算220元。",
        "grounding_phrases": ["最低费用超出预算220元"],
    },
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "ACTIVITY_TIME_OVERLAP",
        "request_suffix": " {destination}行程的地点、预算和活动集合保持不变，已知活动时段尚未固定先后关系。",
        "violation": "{destination}{travel_days}日行程中，两项安排相互占用38分钟；场馆开放时段未发生变化。",
        "grounding_phrases": ["相互占用38分钟", "开放时段未发生变化"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " {destination}行程的活动集合、预算和固定活动安排保持不变，每日时间窗可在既有边界内微调。",
        "violation": "{destination}{travel_days}日行程的末项超出每日结束边界47分钟；地点和开放时段仍有效。",
        "grounding_phrases": ["超出每日结束边界47分钟", "开放时段仍有效"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "FIXED_EVENT_TIME_MISMATCH",
        "request_suffix": " {destination}的固定活动处于允许入场区间内，活动集合、总预算和每日时间窗保持不变。",
        "violation": "{destination}{travel_days}日行程中，固定活动与登记时刻相差28分钟；允许入场区间仍可使用。",
        "grounding_phrases": ["与登记时刻相差28分钟", "允许入场区间仍可使用"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "TOTAL_BUDGET_EXCEEDED",
        "request_suffix": " {destination}行程的每日时间窗和通勤上限保持不变；活动顺序、活动集合和总预算具有可调整空间。",
        "violation": "{destination}{travel_days}日行程围绕{focus}偏好安排后，费用高于当前边界160元。",
        "grounding_phrases": ["高于当前边界160元", "{focus}偏好"],
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "FIXED_EVENT_END_EXCEEDED",
        "request_suffix": " {destination}行程的活动顺序、活动集合和总预算保持不变；固定活动时间与通勤上限具有可调整空间。",
        "violation": "{destination}{travel_days}日行程的固定活动比已知结束时刻晚32分钟，并占用了返程前时段。",
        "grounding_phrases": ["比已知结束时刻晚32分钟", "占用了返程前时段"],
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " {destination}行程的总预算和固定活动时间保持不变；活动集合与每日时间安排具有可调整空间。",
        "violation": "{destination}{travel_days}日行程在保留{focus}偏好时，当日时长多出72分钟。",
        "grounding_phrases": ["当日时长多出72分钟", "{focus}偏好"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "MAX_TRANSIT_EXCEEDED",
        "request_suffix": " {destination}行程的每日时间窗与通勤上限保持不变；固定活动可在既有范围内安排，活动集合和总预算具有可调整空间。",
        "violation": "{destination}{travel_days}日行程保留全部锁定地点后，日间通勤高出上限55分钟。",
        "grounding_phrases": ["日间通勤高出上限55分钟", "锁定地点"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "FIXED_EVENT_TIME_MISMATCH",
        "request_suffix": " {destination}行程的总预算和固定活动时间保持不变；活动顺序与通勤路线可在既有上限内安排，活动集合具有可调整空间。",
        "violation": "{destination}{travel_days}日行程的固定活动与锁定返程时间间隔46分钟。",
        "grounding_phrases": ["锁定返程时间间隔46分钟"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "TOTAL_BUDGET_EXCEEDED",
        "request_suffix": " {destination}行程的总预算和固定活动时间保持不变；活动顺序与通勤路线可在既有上限内安排，活动集合具有可调整空间。",
        "violation": "{destination}{travel_days}日行程在所有锁定支出下仍高出预算195元。",
        "grounding_phrases": ["仍高出预算195元", "锁定支出"],
    },
)


# Strict counterfactual triples for the promotion-eligible profile.  Within
# one case and one source state, request wording, failed verifier fact and all
# non-flexibility task fields are identical.  Only the typed handling of the
# failed constraint changes: solver-adjustable, relaxable or locked.
_COUNTERFACTUAL_CASES: tuple[dict[str, Any], ...] = (
    {
        "case_id": "overlap-45m",
        "violation_code": "ACTIVITY_TIME_OVERLAP",
        "violation": (
            "{destination}{travel_days}日行程第{activity_day}天，{activity_a}尚未结束就与"
            "{activity_b}重叠45分钟。"
        ),
        "grounding_phrases": ["{activity_a}尚未结束", "{activity_b}重叠45分钟"],
    },
    {
        "case_id": "day-boundary-52m",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "violation": (
            "{destination}{travel_days}日行程第{activity_day}天的{activity_b}在当日活动时间"
            "边界后结束52分钟。"
        ),
        "grounding_phrases": ["{activity_b}", "时间边界后结束52分钟"],
    },
    {
        "case_id": "fixed-start-30m",
        "violation_code": "FIXED_EVENT_TIME_MISMATCH",
        "violation": (
            "{destination}{travel_days}日行程第{activity_day}天，固定活动{activity_a}的开始时间"
            "比预约时间晚30分钟。"
        ),
        "grounding_phrases": ["{activity_a}", "比预约时间晚30分钟"],
    },
    {
        "case_id": "fixed-end-35m",
        "violation_code": "FIXED_EVENT_END_EXCEEDED",
        "violation": (
            "{destination}{travel_days}日行程第{activity_day}天，固定活动{activity_b}的结束时间"
            "比已知结束时间晚35分钟。"
        ),
        "grounding_phrases": ["{activity_b}", "比已知结束时间晚35分钟"],
    },
    {
        "case_id": "transit-48m",
        "violation_code": "MAX_TRANSIT_EXCEEDED",
        "violation": (
            "{destination}{travel_days}日行程第{activity_day}天，从{activity_a}前往{activity_b}的"
            "通勤时间比上限多48分钟。"
        ),
        "grounding_phrases": ["{activity_a}前往{activity_b}", "比上限多48分钟"],
    },
    {
        "case_id": "budget-175y",
        "violation_code": "TOTAL_BUDGET_EXCEEDED",
        "violation": (
            "{destination}{travel_days}日行程保留{activity_a}和{activity_b}后，总预算比用户设定"
            "上限高175元。"
        ),
        "grounding_phrases": ["{activity_a}和{activity_b}", "上限高175元"],
    },
)
_COUNTERFACTUAL_OPTIONS: dict[str, list[str]] = {
    "activity_schedule": ["调整活动顺序"],
    "daily_time_window": ["延长每日活动时间"],
    "fixed_event_time": ["调整固定活动时间"],
    "max_transit_minutes": ["放宽通勤时间上限"],
    "total_budget": ["提高总预算"],
}
_CONSTRAINT_DIMENSIONS = tuple(_COUNTERFACTUAL_OPTIONS)


def _counterfactual_templates() -> tuple[dict[str, Any], ...]:
    variants: list[dict[str, Any]] = []
    for case in _COUNTERFACTUAL_CASES:
        for handling, target, family in (
            ("solver_adjustable", "retry_solve", "repairable_solver_failure"),
            ("relaxable", "propose_tradeoff", "actionable_tradeoff"),
            ("locked", "abort", "necessary_abort"),
        ):
            variants.append(
                {
                    **case,
                    "family": family,
                    "target_action": target,
                    "counterfactual_handling": handling,
                    # This suffix is deliberately identical for all three
                    # members of a case and carries no action hint.
                    "request_suffix": " 请在不改变原始需求表述的前提下核对当前安排。",
                    "expected_arguments": (
                        {"strategy": "greedy"} if target == "retry_solve" else {}
                    ),
                }
            )
    return tuple(variants)


_STRICT_COUNTERFACTUAL_TRAIN_TEMPLATES = _counterfactual_templates()


# Fixed external evaluation uses counterfactual triples: the verifier code and
# message stay constant while only neutral user-constraint facts change.  The
# requests avoid direct instructions such as “retry”, “offer options” or “stop”,
# so a lexical action cue or code lookup cannot solve the suite.
_IMPLICIT_COUNTERFACTUAL_VALIDATION_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "ACTIVITY_TIME_OVERLAP",
        "request_suffix": " 场馆时段、活动集合和交通边界均有效，活动先后没有锁定。",
        "violation": "行程中活动A结束时间与活动B开始时间重叠40分钟",
        "grounding_phrases": ["重叠40分钟"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "ACTIVITY_TIME_OVERLAP",
        "request_suffix": " 每日时窗和通勤上限是硬要求，活动顺序的任何变化都需征得我的确认。",
        "violation": "行程中活动A结束时间与活动B开始时间重叠40分钟",
        "grounding_phrases": ["重叠40分钟"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "ACTIVITY_TIME_OVERLAP",
        "request_suffix": " 两场活动均已付款，活动先后与活动集合都不可改变。",
        "violation": "行程中活动A结束时间与活动B开始时间重叠40分钟",
        "grounding_phrases": ["重叠40分钟"],
    },
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " 活动集合、顺序和预算保持不变，每日时间窗有一段预先授权的弹性范围。",
        "violation": "当前排程的最后一项超出每日结束边界35分钟",
        "grounding_phrases": ["超出每日结束边界35分钟"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " 预算与固定活动时间锁定，每日活动时长可以协商调整。",
        "violation": "当前排程的最后一项超出每日结束边界35分钟",
        "grounding_phrases": ["超出每日结束边界35分钟"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "DAY_TIME_BOUNDARY_EXCEEDED",
        "request_suffix": " 每日结束时间是健康硬限制，通勤上限也不能增加。",
        "violation": "当前排程的最后一项超出每日结束边界35分钟",
        "grounding_phrases": ["超出每日结束边界35分钟"],
    },
    {
        "family": "repairable_solver_failure",
        "target_action": "retry_solve",
        "violation_code": "FIXED_EVENT_TIME_MISMATCH",
        "request_suffix": " 固定活动允许在票面标注的入场时段内由系统安排具体开始时间。",
        "violation": "当前排程的活动开始时间与固定活动时刻相差25分钟",
        "grounding_phrases": ["相差25分钟"],
        "expected_arguments": {"strategy": "greedy"},
    },
    {
        "family": "actionable_tradeoff",
        "target_action": "propose_tradeoff",
        "violation_code": "FIXED_EVENT_TIME_MISMATCH",
        "request_suffix": " 活动集合、顺序和预算不变，固定活动时间如需变化必须先征得我的确认。",
        "violation": "当前排程的活动开始时间与固定活动时刻相差25分钟",
        "grounding_phrases": ["相差25分钟"],
    },
    {
        "family": "necessary_abort",
        "target_action": "abort",
        "violation_code": "FIXED_EVENT_TIME_MISMATCH",
        "request_suffix": " 固定活动时间和预算均已锁定，不能改期也不能增加费用。",
        "violation": "当前排程的活动开始时间与固定活动时刻相差25分钟",
        "grounding_phrases": ["相差25分钟"],
    },
)

_TEMPLATES["validation"] = _IMPLICIT_COUNTERFACTUAL_VALIDATION_TEMPLATES


# These are neutral, production-schema intent annotations.  A pattern states
# which user constraints are locked, solver-adjustable inside existing bounds,
# or negotiable with explicit options.  Every pattern appears with all three
# target actions across different verifier codes, so the contract alone cannot
# reveal the hidden reward label.
_FLEXIBILITY_PATTERNS: dict[str, dict[str, Any]] = {
    "P0": {
        "schema_version": "constraint-flexibility.v1",
        "locked_constraints": ["total_budget", "fixed_event_time"],
        "solver_adjustable_constraints": [
            "activity_schedule",
            "max_transit_minutes",
        ],
        "relaxable_constraints": ["activity_set", "daily_time_window"],
        "relaxation_options": {
            "activity_set": ["减少一个非必去活动"],
            "daily_time_window": ["延长每日活动时间"],
        },
        "request_statement": (
            " 总预算和固定活动时间已经锁定；活动顺序与通勤路线可在现有上限内重排；"
            "活动集合和每日活动时间存在可调整空间。"
        ),
    },
    "P1": {
        "schema_version": "constraint-flexibility.v1",
        "locked_constraints": ["daily_time_window", "max_transit_minutes"],
        "solver_adjustable_constraints": ["fixed_event_time"],
        "relaxable_constraints": [
            "activity_schedule",
            "activity_set",
            "total_budget",
        ],
        "relaxation_options": {
            "activity_schedule": ["调整活动顺序"],
            "activity_set": ["减少一个非必去活动"],
            "total_budget": ["提高总预算"],
        },
        "request_statement": (
            " 每日活动时间窗和通勤上限不能改；固定活动可在已授权时段内重排；"
            "活动顺序、活动集合和总预算存在可调整空间。"
        ),
    },
    "P2": {
        "schema_version": "constraint-flexibility.v1",
        "locked_constraints": [
            "activity_schedule",
            "activity_set",
            "total_budget",
        ],
        "solver_adjustable_constraints": ["daily_time_window"],
        "relaxable_constraints": ["fixed_event_time", "max_transit_minutes"],
        "relaxation_options": {
            "fixed_event_time": ["调整固定活动时间"],
            "max_transit_minutes": ["放宽通勤时间上限"],
        },
        "request_statement": (
            " 活动顺序、活动集合和总预算不能改；每日时间窗可在已授权范围内微调；"
            "固定活动时间和通勤上限存在可调整空间。"
        ),
    },
}

_FLEXIBILITY_PATTERN_IDS: dict[str, tuple[str, ...]] = {
    "train": ("P2", "P1", "P1", "P1", "P2", "P0"),
    "validation": ("P0", "P1", "P2", "P2", "P0", "P1", "P1", "P2", "P0"),
    "test": ("P2", "P0", "P0"),
    "semantic": ("P2", "P0", "P1", "P0", "P2", "P1", "P1", "P0", "P1"),
    "rl": ("P0", "P2", "P1", "P1", "P2", "P0", "P1", "P0", "P0"),
    "neutral": (
        "P0", "P2", "P1", "P1", "P2", "P0", "P1", "P0", "P0",
        "P0", "P2", "P1", "P1", "P2", "P0", "P1", "P0", "P0",
    ),
}


def _bind_flexibility_patterns() -> None:
    groups = {
        "train": _TEMPLATES["train"],
        "validation": _TEMPLATES["validation"],
        "test": _TEMPLATES["test"],
        "semantic": _SEMANTIC_DIVERSE_TRAIN_TEMPLATES,
        "rl": _RL_CHALLENGE_TRAIN_TEMPLATES,
        "neutral": _NEUTRAL_DIVERSE_TRAIN_TEMPLATES,
    }
    for group, templates in groups.items():
        pattern_ids = _FLEXIBILITY_PATTERN_IDS[group]
        if len(templates) != len(pattern_ids):
            raise RuntimeError(f"{group}: flexibility pattern count mismatch")
        for template, pattern_id in zip(templates, pattern_ids, strict=True):
            template["flexibility_pattern"] = pattern_id


_bind_flexibility_patterns()


# These lexical fragments encode the desired action rather than the travel
# state.  They are forbidden in the fresh training profile, while the older
# profiles stay readable only for reproducibility of historical artifacts.
_ACTION_LEXICAL_CUES = (
    "让我选",
    "让我决定",
    "请给我",
    "重试",
    "重算",
    "算法",
    "策略",
    "greedy",
    "终止",
    "停止",
    "结束规划",
    "结束计划",
    "直接放弃",
    "不要继续",
    "无解",
    "确认",
)


def _template_context(source: GRPOCorpusRow) -> dict[str, str]:
    """Return only prompt-visible, request-derived fields for wording variety."""
    slots = dict(source.task.slots or {})
    profile = dict(source.task.profile or {})
    interests = [str(item).strip() for item in profile.get("interests") or [] if str(item).strip()]
    activity_context = _verifier_activity_context(source)
    activity_day, activity_a, activity_b = activity_context or (
        "某",
        "前一项活动",
        "后一项活动",
    )
    return {
        "destination": str(slots.get("destination") or "本次目的地").strip(),
        "travel_days": str(slots.get("travel_days") or "本次").strip(),
        "start_date": str(slots.get("start_date") or "行程日期").strip(),
        "budget": str(slots.get("budget_range") or "既定").strip(),
        # Templates add the suffix “偏好”, so the fallback is deliberately
        # “既定” rather than “既定偏好”; otherwise rows without an interest
        # produce the low-quality phrase “既定偏好偏好”.
        "focus": interests[0] if interests else "既定",
        # These labels are taken from the verified solver artifact and then
        # repeated in the user-visible verifier observation.  They make the
        # failure explanation self-contained without padding it with unrelated
        # dates or budget values merely to increase lexical diversity.
        "activity_day": activity_day,
        "activity_a": activity_a,
        "activity_b": activity_b,
    }


def _verifier_activity_context(source: GRPOCorpusRow) -> tuple[str, str, str] | None:
    """Select two concrete, planned activities for a synthetic verifier fact."""
    responses = list(source.snapshot.tool_responses.get("solve_itinerary") or [])
    payload = dict(responses[0].data or {}) if responses else {}
    for day in payload.get("days") or []:
        if not isinstance(day, dict):
            continue
        labels = [
            str(item.get("poi_name") or item.get("name") or "").strip()
            for item in day.get("activities") or []
            if isinstance(item, dict)
            and str(item.get("category") or "").casefold() not in {"restaurant", "meal"}
            and str(item.get("poi_name") or item.get("name") or "").strip()
        ]
        if len(labels) >= 2:
            return str(day.get("day_number") or "某").strip(), labels[0], labels[1]
    return None


def _strict_counterfactual_sources(rows: list[GRPOCorpusRow]) -> list[GRPOCorpusRow]:
    """Keep only sources whose verifier fact can name concrete planned activities.

    Strict counterfactual rows must never fall back to a synthetic placeholder
    such as “前一项活动”.  It would make a formally grounded reason appear
    complete while asking the policy to infer facts absent from its observation.
    Other corpus profiles do not use those activity placeholders and retain the
    broader compatibility pool.
    """
    return [row for row in rows if _verifier_activity_context(row) is not None]


def _render_template_text(value: str, context: dict[str, str]) -> str:
    try:
        return str(value).format_map(context)
    except KeyError as exc:
        raise ValueError(f"unknown verifier-repair template placeholder: {exc.args[0]}") from exc


def _assert_neutral_template_surface(templates: tuple[dict[str, Any], ...]) -> None:
    surface = "\n".join(
        "\n".join(
            [
                str(template.get("request_suffix") or ""),
                str(template.get("violation") or ""),
                *[str(item) for item in template.get("grounding_phrases") or []],
            ]
        )
        for template in templates
    ).casefold()
    leaked = [cue for cue in _ACTION_LEXICAL_CUES if cue.casefold() in surface]
    if leaked:
        raise ValueError("action-lexical leakage in neutral train templates: " + ",".join(leaked))


def _counterfactual_contract(
    violation_code: str,
    handling: str,
) -> ConstraintFlexibilityContract:
    dimension = VERIFIER_REPAIR_CONSTRAINT_BY_CODE.get(violation_code)
    if dimension not in _COUNTERFACTUAL_OPTIONS:
        raise ValueError(f"counterfactual case has unsupported violation: {violation_code}")
    locked = list(_CONSTRAINT_DIMENSIONS)
    relaxed: list[str] = []
    adjustable: list[str] = []
    options: dict[str, list[str]] = {}
    if handling == "solver_adjustable":
        locked.remove(dimension)
        adjustable.append(dimension)
    elif handling == "relaxable":
        locked.remove(dimension)
        relaxed.append(dimension)
        options[dimension] = list(_COUNTERFACTUAL_OPTIONS[dimension])
    elif handling != "locked":
        raise ValueError(f"unknown counterfactual handling: {handling}")
    return ConstraintFlexibilityContract.model_validate(
        {
            "schema_version": "constraint-flexibility.v1",
            "locked_constraints": locked,
            "solver_adjustable_constraints": adjustable,
            "relaxable_constraints": relaxed,
            "relaxation_options": options,
        }
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable(rows: list[GRPOCorpusRow]) -> list[GRPOCorpusRow]:
    return sorted(rows, key=lambda row: hashlib.sha256(row.task.task_id.encode()).hexdigest())


def _prefix_actions(row: GRPOCorpusRow) -> list[dict[str, Any]]:
    interests = list(row.task.profile.get("interests") or [])
    return [
        {
            "action": action,
            "arguments": ({"keywords": interests} if arguments is None else arguments),
        }
        for action, arguments in PREFIX_ACTIONS
    ]


def _prepare_variant(
    source: GRPOCorpusRow,
    *,
    split: str,
    template: dict[str, Any],
    ordinal: int,
) -> GRPOCorpusRow:
    row = source.model_copy(deep=True)
    target = str(template["target_action"])
    signature = hashlib.sha256(
        f"{split}:{source.task.task_id}:{target}:{ordinal}".encode()
    ).hexdigest()[:16]
    row.task.task_id = f"verifier-repair-{split}-{target}-{signature}"
    row.task.template_family = f"verifier-repair:{template['family']}:{split}"
    row.task.difficulty = "L4"
    template_context = _template_context(source)
    violation_code = str(template["violation_code"])
    counterfactual_handling = template.get("counterfactual_handling")
    if counterfactual_handling is not None:
        pattern_id = f"counterfactual:{template['case_id']}"
        request_statement = ""
        contract = _counterfactual_contract(violation_code, str(counterfactual_handling))
    else:
        pattern_id = str(template["flexibility_pattern"])
        pattern = dict(_FLEXIBILITY_PATTERNS[pattern_id])
        request_statement = _render_template_text(
            str(pattern.pop("request_statement")), template_context
        )
        contract = ConstraintFlexibilityContract.model_validate(pattern)
    handling = constraint_handling_for_violation(violation_code, contract)
    expected_target = {
        "solver_adjustable": "retry_solve",
        "relaxable": "propose_tradeoff",
        "locked": "abort",
    }.get(handling)
    if expected_target != target:
        raise ValueError(
            f"{split}:{ordinal}: neutral contract implies {expected_target}, not {target}"
        )
    supervised_options = relaxation_options_for_violation(violation_code, contract)
    row.task.user_request = (
        source.task.user_request
        + _render_template_text(str(template["request_suffix"]), template_context)
        + request_statement
    )
    row.task.slots = {
        **dict(row.task.slots),
        # This is the same typed field produced by the production demand
        # parser. It contains no action/family label and is assigned from a
        # reusable pattern rather than from ``target_action``.
        "constraint_flexibility": contract.model_dump(mode="json"),
    }
    row.snapshot.environment_version = SCHEMA_VERSION
    row.snapshot.snapshot_version = f"{SCHEMA_VERSION}-{signature[:8]}"
    row.snapshot.state_id = f"{source.snapshot.state_id}-verifier-repair-{signature[:8]}"

    original_validation = row.snapshot.tool_responses["validate_itinerary"][0].model_copy(
        deep=True
    )
    failed_validation = original_validation.model_copy(deep=True)
    failed_validation.data = {
        **dict(original_validation.data or {}),
        "hard_pass": False,
        "hard_violations": [
            {
                # Use only codes emitted by the production validator.  The code
                # identifies the failed constraint, never the desired policy
                # action; user constraints and verifier text determine whether
                # the correct response is retry, trade-off, or safe abort.
                "code": violation_code,
                "message": _render_template_text(
                    str(template["violation"]), template_context
                ),
                "details": {"verifier_repair_case": True},
            }
        ],
    }
    row.snapshot.tool_responses["validate_itinerary"] = [failed_validation]
    if target == "retry_solve":
        # The target action triggers a real local replan. Supply an independent
        # second solver/verifier response so the decision is demonstrably
        # repairable instead of succeeding only because the one-step reward
        # ignores downstream execution.
        row.snapshot.tool_responses["solve_itinerary"].append(
            row.snapshot.tool_responses["solve_itinerary"][0].model_copy(deep=True)
        )
        row.snapshot.tool_responses["validate_itinerary"].append(original_validation)

    prefix_actions = _prefix_actions(row)
    prompt_messages, review_state = _replay_prompt(row, prefix_actions)
    replayed_options = controller_tradeoff_options(
        dict(review_state.get("capability") or {})
    )
    if target == "propose_tradeoff":
        if (
            supervised_options != replayed_options
            or not supervised_options
        ):
            raise ValueError(
                "tradeoff options do not match the authority reconstructed by replay"
            )
    elif supervised_options or template.get("require_options"):
        raise ValueError("non-tradeoff template contains controller-owned options")
    template_expected_arguments = dict(template.get("expected_arguments") or {})
    if target == "retry_solve":
        # The model chooses only whether a verifier-grounded retry is
        # appropriate and explains why.  The bounded fallback solver is a
        # controller decision: initial solves are forced to CP-SAT by the
        # executor, and the sole authorized recovery is greedy.
        if template_expected_arguments != {"strategy": "greedy"}:
            raise ValueError("retry template must declare the controller fallback strategy")
        expected_arguments: dict[str, Any] = {}
        controller_arguments = template_expected_arguments
    else:
        expected_arguments = template_expected_arguments
        controller_arguments = {}
    hidden = dict(row.snapshot.hidden_test_facts)
    hidden["grpo_decision_state"] = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "target_action": target,
        "expected_arguments": expected_arguments,
        "controller_arguments": controller_arguments,
        "grounding_phrases": [
            _render_template_text(str(item), template_context)
            for item in template.get("grounding_phrases") or []
        ],
        "require_options": bool(supervised_options),
        "supervised_options": supervised_options,
        "prefix_actions": prefix_actions,
        "prompt_messages": prompt_messages,
        "source_task_id": source.task.task_id,
        "source_snapshot_version": source.snapshot.snapshot_version,
        "scenario_family": template["family"],
        "flexibility_pattern": pattern_id,
        "counterfactual_group_id": (
            f"{source.task.task_id}:{template['case_id']}"
            if counterfactual_handling is not None
            else None
        ),
        "split": split,
        "review_allowed_actions": list(review_state.get("allowed_actions") or []),
    }
    row.snapshot.hidden_test_facts = hidden
    return row


def _replay_prompt(
    row: GRPOCorpusRow,
    prefix_actions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    environment = TRLReactEnvironment(audit_enabled=False)
    try:
        initial = environment.reset(
            task=row.task.model_dump(mode="json"),
            snapshot=row.snapshot.model_dump(mode="json"),
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": initial},
        ]
        rendered: dict[str, Any] = {"policy_state": {}}
        for item in prefix_actions:
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": item["action"],
                                "arguments": item["arguments"],
                            },
                        }
                    ],
                }
            )
            result = environment._act(item["action"], item["arguments"])
            messages.append(
                {"role": "tool", "name": item["action"], "content": result}
            )
            rendered = json.loads(result)
        state = rendered.get("policy_state") or {}
        if (state.get("current_subtask") or {}).get("task_id") != "review_itinerary":
            raise ValueError(f"{row.task.task_id}: verified prefix did not reach review")
        return messages, state
    finally:
        environment.get_reward()


def _compatible_sources(rows: list[GRPOCorpusRow]) -> list[GRPOCorpusRow]:
    compatible: list[GRPOCorpusRow] = []
    probe = _TEMPLATES["train"][2]
    for ordinal, row in enumerate(_stable(rows)):
        try:
            _prepare_variant(
                row,
                split="probe",
                template=probe,
                ordinal=ordinal,
            )
        except (RuntimeError, ValueError):
            continue
        compatible.append(row)
    return compatible


def _build_rows(
    sources: list[GRPOCorpusRow],
    *,
    split: str,
    templates: tuple[dict[str, Any], ...] | None = None,
) -> list[GRPOCorpusRow]:
    templates = templates or _TEMPLATES[split]
    return [
        _prepare_variant(
            source,
            split=split,
            template=template,
            ordinal=source_index * len(templates) + template_index,
        )
        for source_index, source in enumerate(sources)
        for template_index, template in enumerate(templates)
    ]


def _write_jsonl(path: Path, rows: list[GRPOCorpusRow]) -> None:
    path.write_bytes(
        "".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ).encode("utf-8")
    )


def build(
    *,
    source_dir: Path,
    output_dir: Path,
    train_sources: int = 60,
    validation_sources: int = 16,
    semantic_diverse_train: bool = False,
    rl_challenge_train: bool = False,
    neutral_diverse_train: bool = False,
    strict_counterfactual_train: bool = False,
) -> dict[str, Any]:
    source_train = _compatible_sources(load_grpo_corpus(source_dir / "train.jsonl"))
    source_test = _compatible_sources(load_grpo_corpus(source_dir / "validation.jsonl"))
    required = train_sources + validation_sources
    if len(source_train) < required:
        raise ValueError(f"only {len(source_train)} compatible train sources for {required}")
    if not source_test:
        raise ValueError("no compatible frozen-test sources")

    train_templates = _TEMPLATES["train"]
    train_source_rows = source_train[:train_sources]
    validation_source_rows = source_train[train_sources:required]
    if sum(
        (
            semantic_diverse_train,
            rl_challenge_train,
            neutral_diverse_train,
            strict_counterfactual_train,
        )
    ) > 1:
        raise ValueError("train template profiles are mutually exclusive")
    if strict_counterfactual_train:
        train_templates = _STRICT_COUNTERFACTUAL_TRAIN_TEMPLATES
        _assert_neutral_template_surface(train_templates)
        strict_sources = _strict_counterfactual_sources(source_train)
        if len(strict_sources) < train_sources:
            raise ValueError(
                "only "
                f"{len(strict_sources)} strict-counterfactual-compatible train sources for "
                f"{train_sources}"
            )
        train_source_rows = strict_sources[:train_sources]
        selected_ids = {row.task.task_id for row in train_source_rows}
        validation_source_rows = [
            row for row in source_train if row.task.task_id not in selected_ids
        ][:validation_sources]
        if len(validation_source_rows) < validation_sources:
            raise ValueError(
                "not enough source-disjoint validation rows after selecting strict "
                "counterfactual training sources"
            )
    elif neutral_diverse_train:
        train_templates = _NEUTRAL_DIVERSE_TRAIN_TEMPLATES
        _assert_neutral_template_surface(train_templates)
    elif rl_challenge_train:
        train_templates = _RL_CHALLENGE_TRAIN_TEMPLATES
    elif semantic_diverse_train:
        train_templates = (*train_templates, *_SEMANTIC_DIVERSE_TRAIN_TEMPLATES)
    train = _build_rows(
        train_source_rows,
        split="train",
        templates=train_templates,
    )
    validation = _build_rows(validation_source_rows, split="validation")
    test = _build_rows(source_test, split="test")
    splits = {"train": train, "validation": validation, "test": test}

    task_ids = {name: {row.task.task_id for row in rows} for name, rows in splits.items()}
    source_ids = {
        name: {
            row.snapshot.hidden_test_facts["grpo_decision_state"]["source_task_id"]
            for row in rows
        }
        for name, rows in splits.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if task_ids[left] & task_ids[right]:
            raise ValueError(f"task leakage between {left} and {right}")
        if source_ids[left] & source_ids[right]:
            raise ValueError(f"source-state leakage between {left} and {right}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
    preflight = preflight_grpo_corpus(
        output_dir,
        minimum_train_tasks=len(train),
        require_dependencies=False,
    )
    if not preflight.ready:
        raise ValueError("verifier-repair corpus failed preflight: " + ",".join(preflight.errors))

    split_sha256 = {
        name: _sha256(output_dir / f"{name}.jsonl") for name in splits
    }
    source_split_sha256 = {
        name: _sha256(source_dir / f"{name}.jsonl")
        for name in ("train", "validation")
    }
    corpus_content_sha256 = hashlib.sha256(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "decision_schema_version": DECISION_SCHEMA_VERSION,
                "split_sha256": split_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "decision_schema_version": DECISION_SCHEMA_VERSION,
        "constraint_flexibility_schema_version": "constraint-flexibility.v1",
        "source_dir": str(source_dir),
        "source_rule": (
            "accepted Native ReAct train snapshots split by source state; original validation "
            "snapshots reserved for frozen test; independent hard benchmark never read"
        ),
        "train_template_profile": (
            "strict-counterfactual-v1"
            if strict_counterfactual_train
            else
            "neutral-diverse-v2"
            if neutral_diverse_train
            else
            "rl-challenge-unseen-v1"
            if rl_challenge_train
            else "semantic-diverse-v1"
            if semantic_diverse_train
            else "base-v1"
        ),
        "validation_template_profile": "implicit-counterfactual-grid-v1",
        "template_counts": {
            "train": len(train_templates),
            "validation": len(_TEMPLATES["validation"]),
            "test": len(_TEMPLATES["test"]),
        },
        "template_neutrality": {
            "enforced": neutral_diverse_train or strict_counterfactual_train,
            "forbidden_action_lexical_cues": list(_ACTION_LEXICAL_CUES)
            if neutral_diverse_train or strict_counterfactual_train
            else [],
        },
        "counts": {name: len(rows) for name, rows in splits.items()},
        "source_counts": {
            "train": train_sources,
            "validation": validation_sources,
            "test": len(source_test),
        },
        "family_counts": {
            name: dict(
                sorted(
                    Counter(
                        row.snapshot.hidden_test_facts["grpo_decision_state"][
                            "scenario_family"
                        ]
                        for row in rows
                    ).items()
                )
            )
            for name, rows in splits.items()
        },
        "target_counts": {
            name: dict(
                sorted(
                    Counter(
                        row.snapshot.hidden_test_facts["grpo_decision_state"]["target_action"]
                        for row in rows
                    ).items()
                )
            )
            for name, rows in splits.items()
        },
        "builder_sha256": _sha256(Path(__file__)),
        "source_split_sha256": source_split_sha256,
        "split_sha256": split_sha256,
        "corpus_content_sha256": corpus_content_sha256,
        "task_overlap": [],
        "source_state_overlap": [],
        "frozen_test_in_training": False,
        "preflight": preflight.model_dump(mode="json"),
    }
    (output_dir / "manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("ml/agentic/datasets/native-react-grpo-v1"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sources", type=int, default=60)
    parser.add_argument("--validation-sources", type=int, default=16)
    parser.add_argument(
        "--semantic-diverse-train",
        action="store_true",
        help=(
            "Add train-only verifier paraphrases while preserving validation and test wording."
        ),
    )
    parser.add_argument(
        "--rl-challenge-train",
        action="store_true",
        help=(
            "Build train-source unseen paraphrases for on-policy GRPO exploration only."
        ),
    )
    parser.add_argument(
        "--neutral-diverse-train",
        action="store_true",
        help=(
            "Build the production-eligible neutral, source-contextualized training profile."
        ),
    )
    parser.add_argument(
        "--strict-counterfactual-train",
        action="store_true",
        help=(
            "Build source-matched triples that vary only the typed handling of the failed constraint."
        ),
    )
    args = parser.parse_args()
    print(json.dumps(build(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
