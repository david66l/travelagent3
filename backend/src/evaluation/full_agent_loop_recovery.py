"""Fault-injection contracts for the production ReAct full Agent Loop."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel

from agentic.action_executor import TravelActionExecutor
from agentic.loop import ActionExecutor, ActionOutcome, PolicyAction
from agentic.observations import ObservationEnvelope
from agentic.state import AgentLedgerState, TaskNode
from evaluation.full_agent_loop_benchmark import FullAgentLoopCase


SCHEMA_VERSION = "full-agent-loop-recovery.v1"


class RecoveryFaultSpec(BaseModel):
    scenario: Literal["change_arguments", "retry_same_arguments"]
    evidence_style: Literal["explicit_instruction", "diagnostic_evidence"]
    action: Literal["search_pois"] = "search_pois"
    error_code: Literal["QUERY_TOO_BROAD", "TOOL_TIMEOUT"]
    message: str
    target_keyword: str | None = None
    dropped_keyword: str | None = None


class FullAgentLoopRecoveryCase(BaseModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case: FullAgentLoopCase
    fault: RecoveryFaultSpec

    @property
    def case_id(self) -> str:
        return self.case.case_id


class OneShotFaultExecutor:
    """Fail one matching production action, then delegate to the real executor."""

    def __init__(
        self,
        fault: RecoveryFaultSpec,
        delegate: ActionExecutor | None = None,
    ) -> None:
        self.fault = fault
        self.delegate = delegate or TravelActionExecutor()
        self.injected = False
        self.trace: list[dict[str, Any]] = []

    async def execute(
        self,
        *,
        task: TaskNode,
        action: PolicyAction,
        ledger: AgentLedgerState,
    ) -> ActionOutcome:
        row = {
            "index": len(self.trace),
            "task": task.task_id,
            "action": action.action,
            "arguments": action.arguments,
            "decision_source": action.decision_source,
            "repair_attempts": action.repair_attempts,
            "repair_error_codes": action.repair_error_codes,
            "injected": False,
        }
        self.trace.append(row)
        if not self.injected and action.action == self.fault.action:
            self.injected = True
            row.update(
                {
                    "injected": True,
                    "status": "failed",
                    "error_code": self.fault.error_code,
                }
            )
            observation = ObservationEnvelope.failure(
                tool=action.action,
                code=self.fault.error_code,
                message=self.fault.message,
                retryable=True,
                tool_call_id=action.action_id,
                latency_ms=5,
                details={
                    "fault_injection": True,
                    "scenario": self.fault.scenario,
                    "evidence_style": self.fault.evidence_style,
                },
            )
            return ActionOutcome(
                status="failed",
                observations=[observation],
                error_code=self.fault.error_code,
                error_message=self.fault.message,
                retryable=True,
                tool_calls_used=1,
            )

        outcome = await self.delegate.execute(task=task, action=action, ledger=ledger)
        row.update(
            {
                "status": outcome.status,
                "error_code": outcome.error_code,
            }
        )
        return outcome


def score_recovery(
    benchmark_case: FullAgentLoopRecoveryCase,
    base_record: dict[str, Any],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score first-try recovery separately from eventual full-chain completion."""
    failures: list[str] = []
    injected_index = next(
        (index for index, row in enumerate(trace) if row.get("injected")),
        None,
    )
    recovery: dict[str, Any] | None = None
    injected: dict[str, Any] | None = None
    if injected_index is None:
        failures.append("FAULT_NOT_INJECTED")
    else:
        injected = trace[injected_index]
        recovery = next(
            (
                row
                for row in trace[injected_index + 1 :]
                if row.get("action") == benchmark_case.fault.action
            ),
            None,
        )
        if recovery is None:
            failures.append("RECOVERY_ACTION_MISSING")

    direct_recovery = bool(
        recovery is not None
        and injected_index is not None
        and int(recovery["index"]) == injected_index + 1
    )
    if recovery is not None and not direct_recovery:
        failures.append("RECOVERY_ACTION_NOT_IMMEDIATE")

    initial_keywords = _keywords(injected)
    recovery_keywords = _keywords(recovery)
    arguments_correct = False
    if recovery is not None:
        if benchmark_case.fault.scenario == "change_arguments":
            target = str(benchmark_case.fault.target_keyword or "")
            dropped = str(benchmark_case.fault.dropped_keyword or "")
            initial_text = _keyword_text(initial_keywords)
            target_text = _keyword_text([target])
            dropped_text = _keyword_text([dropped])
            if target_text not in initial_text or dropped_text not in initial_text:
                failures.append("FAULT_PRECONDITION_NOT_MET")
            arguments_correct = (
                _keyword_text(recovery_keywords) == target_text
                and _keyword_text(recovery_keywords) != initial_text
            )
            if not arguments_correct:
                failures.append("RECOVERY_ARGUMENT_CHANGE_INCORRECT")
        else:
            arguments_correct = recovery_keywords == initial_keywords and bool(initial_keywords)
            if not arguments_correct:
                failures.append("RECOVERY_ARGUMENT_RETRY_INCORRECT")

    repair_attempts = int((recovery or {}).get("repair_attempts") or 0)
    first_try_recovery = bool(
        recovery is not None and direct_recovery and arguments_correct and repair_attempts == 0
    )
    if recovery is not None and repair_attempts:
        failures.append("RECOVERY_REQUIRED_POLICY_REPAIR")

    base_failures = list(base_record.get("failures") or [])
    waived_base_failures: list[str] = []
    if (
        benchmark_case.fault.scenario == "retry_same_arguments"
        and arguments_correct
        and "EXACT_POLICY_ACTION_REPEAT" in base_failures
        and _has_only_expected_timeout_repeat(base_record, initial_keywords)
    ):
        base_failures.remove("EXACT_POLICY_ACTION_REPEAT")
        waived_base_failures.append("EXACT_POLICY_ACTION_REPEAT")
    full_chain_passed = not base_failures
    if not full_chain_passed:
        failures.extend(f"FULL_CHAIN:{code}" for code in base_failures)
    passed = first_try_recovery and full_chain_passed
    return {
        **base_record,
        "base_passed": full_chain_passed,
        "base_failures": base_failures,
        "waived_base_failures": waived_base_failures,
        "passed": passed,
        "failures": list(dict.fromkeys(failures)),
        "recovery": {
            "scenario": benchmark_case.fault.scenario,
            "evidence_style": benchmark_case.fault.evidence_style,
            "error_code": benchmark_case.fault.error_code,
            "fault_injected": injected_index is not None,
            "initial_keywords": initial_keywords,
            "recovery_keywords": recovery_keywords,
            "direct_recovery": direct_recovery,
            "arguments_correct": arguments_correct,
            "repair_attempts": repair_attempts,
            "first_try_recovery": first_try_recovery,
            "full_chain_passed": full_chain_passed,
        },
        "fault_trace": trace,
    }


def _keywords(row: dict[str, Any] | None) -> list[str]:
    if not row:
        return []
    return [str(item) for item in (row.get("arguments") or {}).get("keywords") or []]


def _keyword_text(keywords: list[str]) -> str:
    normalized = unicodedata.normalize("NFKC", "".join(keywords)).casefold()
    return re.sub(r"[\W_]+", "", normalized)


def _has_only_expected_timeout_repeat(
    base_record: dict[str, Any],
    initial_keywords: list[str],
) -> bool:
    exact = Counter(
        (
            str(row.get("action")),
            json.dumps(row.get("arguments") or {}, ensure_ascii=False, sort_keys=True),
        )
        for row in base_record.get("actions") or []
        if row.get("source") == "policy"
    )
    duplicates = {key: count for key, count in exact.items() if count > 1}
    expected = (
        "search_pois",
        json.dumps({"keywords": initial_keywords}, ensure_ascii=False, sort_keys=True),
    )
    return duplicates == {expected: 2}


def build_recovery_cases() -> list[FullAgentLoopRecoveryCase]:
    artifacts = [
        "city_knowledge",
        "poi_candidate_set",
        "poi_detail_set",
        "route_matrix",
        "solver_result",
        "validation_report",
        "itinerary_draft",
    ]
    actions = [
        "retrieve_city_knowledge",
        "search_pois",
        "get_poi_detail",
        "get_route_matrix",
        "finalize_research",
        "solve_itinerary",
        "validate_itinerary",
        "compose_draft",
    ]
    rows = (
        (
            "falr-v1-shanghai-change-explicit",
            "上海",
            "2026年9月12日去上海玩两天，预算4000元，同时喜欢历史文化和美食，节奏轻松。",
            "change_arguments",
            "explicit_instruction",
            "历史文化",
            "美食",
        ),
        (
            "falr-v1-chengdu-change-explicit",
            "成都",
            "2026年9月18日去成都玩两天，预算3500元，同时喜欢历史文化和川菜。",
            "change_arguments",
            "explicit_instruction",
            "历史文化",
            "川菜",
        ),
        (
            "falr-v1-nanjing-change-diagnostic",
            "南京",
            "2026年10月8日去南京玩三天，预算4500元，同时喜欢民国建筑和美食。",
            "change_arguments",
            "diagnostic_evidence",
            "民国建筑",
            "美食",
        ),
        (
            "falr-v1-suzhou-change-diagnostic",
            "苏州",
            "2026年10月14日去苏州玩两天，预算3200元，同时喜欢古典园林和美食。",
            "change_arguments",
            "diagnostic_evidence",
            "古典园林",
            "美食",
        ),
        (
            "falr-v1-beijing-retry-explicit",
            "北京",
            "2026年9月22日去北京玩三天，预算5000元，同时喜欢历史建筑和城市公园。",
            "retry_same_arguments",
            "explicit_instruction",
            "历史建筑",
            "城市公园",
        ),
        (
            "falr-v1-hangzhou-retry-explicit",
            "杭州",
            "2026年9月25日去杭州玩两天，预算3000元，同时喜欢自然风景和茶文化。",
            "retry_same_arguments",
            "explicit_instruction",
            "自然风景",
            "茶文化",
        ),
        (
            "falr-v1-xian-retry-diagnostic",
            "西安",
            "2026年10月18日去西安玩三天，预算4800元，同时喜欢历史遗迹和本地美食。",
            "retry_same_arguments",
            "diagnostic_evidence",
            "历史遗迹",
            "本地美食",
        ),
        (
            "falr-v1-guangzhou-retry-diagnostic",
            "广州",
            "2026年10月24日去广州玩两天，预算3600元，同时喜欢岭南建筑和城市公园。",
            "retry_same_arguments",
            "diagnostic_evidence",
            "岭南建筑",
            "城市公园",
        ),
    )
    cases: list[FullAgentLoopRecoveryCase] = []
    for case_id, city, user_input, scenario, style, target, dropped in rows:
        if scenario == "change_arguments":
            code = "QUERY_TOO_BROAD"
            message = (
                f"组合检索范围过宽；下一次只保留“{target}”，并移除“{dropped}”。"
                if style == "explicit_instruction"
                else f"相关度诊断显示“{dropped}”引入大量偏题候选，而“{target}”仍保持高精度。"
            )
        else:
            code = "TOOL_TIMEOUT"
            message = (
                "参数已经通过校验，仅后端发生瞬时超时；请使用完全相同的关键词重新提交。"
                if style == "explicit_instruction"
                else "请求已进入执行阶段，结果返回前连接短暂中断；现有参数本身没有发现问题。"
            )
        base = FullAgentLoopCase(
            case_id=case_id,
            suite="expanded",
            slice=f"recovery_{scenario}_{style}",
            user_input=user_input,
            required_artifacts=artifacts,
            required_actions=actions,
            expected_slots={"destination": city, "travel_days": 2 if "两天" in user_input else 3},
        )
        cases.append(
            FullAgentLoopRecoveryCase(
                case=base,
                fault=RecoveryFaultSpec(
                    scenario=scenario,
                    evidence_style=style,
                    error_code=code,
                    message=message,
                    target_keyword=target if scenario == "change_arguments" else None,
                    dropped_keyword=dropped if scenario == "change_arguments" else None,
                ),
            )
        )
    return cases


def benchmark_hash() -> str:
    payload = [case.model_dump(mode="json") for case in build_recovery_cases()]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


__all__ = [
    "SCHEMA_VERSION",
    "FullAgentLoopRecoveryCase",
    "OneShotFaultExecutor",
    "RecoveryFaultSpec",
    "benchmark_hash",
    "build_recovery_cases",
    "score_recovery",
]
