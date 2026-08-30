"""Build a frozen AI-authored benchmark with two blind DeepSeek annotations.

The artifact produced by this script is an external-model evaluation set, not an
independent human benchmark.  The API key is read only from DEEPSEEK_API_KEY.
Raw API envelopes and credentials are never persisted.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT  # noqa: E402
from agentic.policy_actions import policy_action_schemas  # noqa: E402
from evaluation.external_benchmark import (  # noqa: E402
    Adjudication,
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
    cohens_kappa,
    normalized_prompt_hash,
)


MODEL = "deepseek-v4-flash"
API_URL = "https://api.deepseek.com/chat/completions"
SCHEMA_VERSION = "travel-agent-stage29-deepseek-benchmark.v1"
DEFAULT_OUTPUT_DIR = Path(
    "ml/agentic/datasets/external-benchmark-v1/deepseek-v4-flash-stage29-v1"
)
ACTIONS = ("search_pois", "ask_user", "propose_tradeoff", "abort")
ACTION_COUNTS = {
    "search_pois": 45,
    "ask_user": 35,
    "propose_tradeoff": 40,
    "abort": 30,
}
SOURCE_BY_ACTION = {
    "search_pois": (
        ("authorized_real_or_simulated", 20),
        ("tool_failure", 15),
        ("long_context_replan", 10),
    ),
    "ask_user": (("authorized_real_or_simulated", 35),),
    "propose_tradeoff": (
        ("authorized_real_or_simulated", 20),
        ("long_context_replan", 20),
    ),
    "abort": (
        ("authorized_real_or_simulated", 10),
        ("tool_failure", 15),
        ("long_context_replan", 5),
    ),
}
CITY_CLUSTERS = (
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
    "central",
    "southwest",
    "northwest",
    "plateau",
    "multi_city",
)
DATE_PATTERNS = (
    "weekday",
    "weekend",
    "holiday",
    "weather_sensitive",
    "transfer_window",
    "in_trip",
    "fixed_booking",
    "unknown_date",
)
DIFFICULTIES = ("L2", "L3", "L3", "L4")
FAULT_TYPES = ("empty_result", "timeout", "rate_limit", "stale_data", "invalid_argument")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _chunks(values: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def build_assignments(seed: int = 20260815) -> list[dict[str, Any]]:
    """Create a deterministic, balanced and content-free authoring plan."""
    rng = random.Random(seed)
    slots: list[tuple[str, str]] = []
    for action in ACTIONS:
        for source, count in SOURCE_BY_ACTION[action]:
            slots.extend((action, source) for _ in range(count))
    if len(slots) != 150:
        raise AssertionError("Stage29 assignment plan must contain 150 cases")
    rng.shuffle(slots)
    assignments = []
    action_seen: Counter[str] = Counter()
    for index, (action, source) in enumerate(slots, start=1):
        action_seen[action] += 1
        # Every fifth item in each action stratum is Dev: 30 Dev / 120 sealed.
        split = "dev" if action_seen[action] % 5 == 0 else "sealed_test"
        fault_type = None
        recoverable = None
        if source == "tool_failure":
            fault_type = FAULT_TYPES[(index - 1) % len(FAULT_TYPES)]
            recoverable = action != "abort"
        assignments.append(
            {
                "assignment_id": f"stage29-ds-{index:03d}",
                "split": split,
                "source": source,
                "target_action": action,
                "difficulty": DIFFICULTIES[(index - 1) % len(DIFFICULTIES)],
                "city_cluster": CITY_CLUSTERS[(index - 1) % len(CITY_CLUSTERS)],
                "date_pattern": DATE_PATTERNS[(index - 1) % len(DATE_PATTERNS)],
                "fault_type": fault_type,
                "recoverable": recoverable,
            }
        )
    return assignments


def _action_invariants(assignment: dict[str, Any]) -> str:
    action = assignment["target_action"]
    if action == "ask_user":
        return (
            "关键信息只能由用户补充；capability_status 必须是 needs_user，"
            "missing_information 非空，actionable_alternatives=false。"
        )
    if action == "propose_tradeoff":
        return (
            "硬约束存在冲突，但至少有一个明确可执行的放宽方案；"
            "capability_status=infeasible，actionable_alternatives=true，alternatives 非空。"
        )
    if action == "abort":
        return (
            "任务不安全、不可支持、硬约束绝对不可行或恢复预算耗尽，且没有可执行替代；"
            "capability_status 为 unsafe/infeasible/missing_tool 之一，"
            "actionable_alternatives=false，alternatives 为空。"
        )
    if assignment["source"] == "tool_failure":
        return (
            "工具失败仍可恢复；capability_status=missing_tool，"
            "actionable_alternatives=false，并保留一次重试预算。"
        )
    return (
        "信息足够且下一步应检索候选；capability_status=solvable，"
        "missing_information 为空，actionable_alternatives=false。"
    )


def _author_prompt(batch: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact = []
    for assignment in batch:
        compact.append({**assignment, "semantic_invariant": _action_invariants(assignment)})
    return [
        {
            "role": "system",
            "content": (
                "你是旅行 Agent 外部评测集作者。只根据本次给出的高层槽位原创中文难例，"
                "不得引用或改写任何现有训练题、示例题或已知 benchmark。每题应自然、具体、"
                "去标识化，并能仅凭题内状态判断下一动作。输出严格 JSON 对象。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "为每个 assignment 生成一条互不重复的旅行 Agent 工具决策题",
                    "actions": {
                        "search_pois": "信息充分或工具失败可恢复，继续检索",
                        "ask_user": "只有用户能补齐关键输入",
                        "propose_tradeoff": "有可执行替代，需要用户放宽冲突约束",
                        "abort": "无安全可行路径且无可执行替代，立即终止",
                    },
                    "requirements": [
                        "request 必须是自然中文用户表达，避免模板句和动作名称",
                        "hard_constraint_description 必须包含可核验的决策边界",
                        "constraint_kind 使用简短 snake_case",
                        "request_family 使用简短 snake_case",
                        "missing_information 只放缺失字段名",
                        "alternatives 最多 3 条，必须与 actionable_alternatives 一致",
                        "search_keywords 为 0 到 4 个自然中文检索词",
                        "不得输出姓名、电话、订单号、精确住址或 API key",
                    ],
                    "output_schema": {
                        "cases": [
                            {
                                "assignment_id": "必须原样返回",
                                "request": "中文请求",
                                "request_family": "snake_case",
                                "constraint_kind": "snake_case",
                                "hard_constraint_description": "可核验硬约束",
                                "missing_information": ["字段名"],
                                "capability_status": "solvable|needs_user|missing_tool|infeasible|unsafe",
                                "actionable_alternatives": False,
                                "alternatives": ["可执行替代"],
                                "search_keywords": ["检索词"],
                            }
                        ]
                    },
                    "assignments": compact,
                },
                ensure_ascii=False,
            ),
        },
    ]


def _blind_case(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "original_request": case["request"],
        "hard_constraint": case["hard_constraint_description"],
        "capability": {
            "status": case["capability_status"],
            "actionable_alternatives": case["actionable_alternatives"],
            "alternatives": case["alternatives"],
        },
        "missing_information": case["missing_information"],
        "failure_summary": case["failure_summary"],
        "remaining_steps": case["remaining_steps"],
        "allowed_actions": list(ACTIONS),
    }


def _annotation_prompt(
    batch: list[dict[str, Any]], *, annotator: str
) -> list[dict[str, str]]:
    cases = [_blind_case(item) for item in batch]
    if annotator == "b":
        cases.reverse()
    emphasis = (
        "优先识别缺失信息与工具恢复条件，再判断终止边界。"
        if annotator == "a"
        else "优先检查安全性、可行性与替代方案是否真的可执行，再独立判断。"
    )
    return [
        {
            "role": "system",
            "content": (
                "你是与出题过程隔离的旅行 Agent 决策标注员。看不到作者目标标签和另一位"
                "标注员答案。每题只能从 allowed_actions 选择一个 primary_action。"
                + emphasis
                + " 输出严格 JSON 对象，不要解释 JSON 以外内容。"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "decision_rules": [
                        "信息足够且需要候选或可恢复重试：search_pois",
                        "关键输入只能由用户提供：ask_user",
                        "约束冲突且存在可执行替代：propose_tradeoff",
                        "不安全/不可支持/不可行且无可执行替代，或恢复耗尽：abort",
                    ],
                    "output_schema": {
                        "annotations": [
                            {
                                "case_id": "原样返回",
                                "primary_action": "四选一",
                                "confidence": "high|medium|low",
                                "reason": "一句话说明决定性证据",
                            }
                        ]
                    },
                    "cases": cases,
                },
                ensure_ascii=False,
            ),
        },
    ]


class DeepSeekClient:
    def __init__(self, api_key: str, *, timeout: float = 180.0) -> None:
        if not api_key.strip():
            raise ValueError("DEEPSEEK_API_KEY is empty")
        self._api_key = api_key
        self._timeout = timeout

    def json_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": MODEL,
            "messages": messages,
            # Structured authoring/label selection does not need a persisted CoT.
            # Non-thinking mode keeps the same V4 Flash model and is materially
            # faster for the 90 small, schema-constrained batches used here.
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
            "max_tokens": 4096,
        }
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    envelope = json.loads(response.read().decode("utf-8"))
                content = envelope["choices"][0]["message"]["content"].strip()
                if content.startswith("```"):
                    content = content.split("\n", 1)[1].rsplit("```", 1)[0]
                return json.loads(content)
            except urllib.error.HTTPError as exc:
                # Authentication/billing failures are deterministic and retrying only
                # creates noise.  Do not include response bodies because providers may
                # echo request metadata in them.
                if 400 <= exc.code < 500 and exc.code not in {408, 429}:
                    if exc.code == 402:
                        raise RuntimeError(
                            "DeepSeek request rejected with HTTP 402; check account balance"
                        ) from None
                    raise RuntimeError(
                        f"DeepSeek request rejected with HTTP {exc.code}"
                    ) from None
                last_error = exc
                if attempt == 4:
                    break
                time.sleep(min(2**attempt, 16))
            except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt == 4:
                    break
                time.sleep(min(2**attempt, 16))
        raise RuntimeError(f"DeepSeek request failed after retries: {last_error}")


def _validate_authored(
    authored: dict[str, Any], assignment: dict[str, Any]
) -> dict[str, Any]:
    authored = dict(authored)
    # V4 Flash occasionally omits semantically empty list fields even under
    # JSON mode.  Canonicalize only optional lists; decision-bearing fields
    # remain required and are checked by the action invariants below.
    authored.setdefault("alternatives", [])
    authored.setdefault("search_keywords", [])
    if assignment["target_action"] != "ask_user":
        authored.setdefault("missing_information", [])
    if not isinstance(authored.get("hard_constraint_description"), str) or not authored[
        "hard_constraint_description"
    ].strip():
        if assignment["target_action"] == "ask_user" and authored.get(
            "missing_information"
        ):
            missing_text = "、".join(str(value) for value in authored["missing_information"])
            authored["hard_constraint_description"] = (
                f"必须先由用户确认缺失信息（{missing_text}），不得猜测后继续。"
            )
        elif assignment["target_action"] == "search_pois":
            keyword_text = "、".join(str(value) for value in authored["search_keywords"])
            authored["hard_constraint_description"] = (
                f"必须通过工具核验符合原请求的候选（{keyword_text or '地点、日期与偏好'}），"
                "不得编造可用性。"
            )
    required = {
        "assignment_id",
        "request",
        "request_family",
        "constraint_kind",
        "hard_constraint_description",
        "missing_information",
        "capability_status",
        "actionable_alternatives",
        "alternatives",
        "search_keywords",
    }
    missing = required - authored.keys()
    if missing:
        raise ValueError(f"{assignment['assignment_id']} missing fields: {sorted(missing)}")
    for field in (
        "request",
        "request_family",
        "constraint_kind",
        "hard_constraint_description",
    ):
        if not isinstance(authored[field], str) or not authored[field].strip():
            raise ValueError(f"{assignment['assignment_id']} has invalid {field}")
    if authored["assignment_id"] != assignment["assignment_id"]:
        raise ValueError("author response changed assignment_id")
    action = assignment["target_action"]
    status = authored["capability_status"]
    alternatives = bool(authored["actionable_alternatives"])
    if action == "ask_user" and (status != "needs_user" or not authored["missing_information"]):
        raise ValueError(f"{assignment['assignment_id']} violates ask_user invariant")
    if action == "propose_tradeoff" and (status != "infeasible" or not alternatives or not authored["alternatives"]):
        raise ValueError(f"{assignment['assignment_id']} violates tradeoff invariant")
    if action == "abort" and (
        status not in {"unsafe", "infeasible", "missing_tool"}
        or alternatives
        or authored["alternatives"]
    ):
        raise ValueError(f"{assignment['assignment_id']} violates abort invariant")
    if action == "search_pois" and status not in {"solvable", "missing_tool"}:
        raise ValueError(f"{assignment['assignment_id']} violates search invariant")
    failure_summary = []
    remaining_steps = 4
    if assignment["source"] == "tool_failure":
        retryable = bool(assignment["recoverable"])
        remaining_steps = 2 if retryable else 1
        failure_summary = [
            {
                "action": "search_pois",
                "error_code": str(assignment["fault_type"]).upper(),
                "retryable": retryable,
                "retry_budget_remaining": 1 if retryable else 0,
            }
        ]
    return {
        **assignment,
        **authored,
        "case_id": assignment["assignment_id"],
        "author_target_action": action,
        "failure_summary": failure_summary,
        "remaining_steps": remaining_steps,
    }


def author_cases(
    client: DeepSeekClient,
    assignments: list[dict[str, Any]],
    output_dir: Path,
    *,
    batch_size: int,
) -> list[dict[str, Any]]:
    authored_by_id: dict[str, dict[str, Any]] = {}
    batches = list(_chunks(assignments, batch_size))
    for batch_number, batch in enumerate(batches, start=1):
        cache = output_dir / "raw" / "author" / f"batch-{batch_number:03d}.json"
        if cache.exists():
            response = _read_json(cache)
        else:
            response = client.json_completion(_author_prompt(batch))
            _write_json(cache, response)
        items = response.get("cases") or []
        by_id = {item.get("assignment_id"): item for item in items}
        for assignment in batch:
            item = by_id.get(assignment["assignment_id"])
            if item is None:
                raise ValueError(f"author batch missing {assignment['assignment_id']}")
            authored_by_id[assignment["assignment_id"]] = _validate_authored(item, assignment)
        print(f"author {batch_number}/{len(batches)} complete", flush=True)
    authored = [authored_by_id[item["assignment_id"]] for item in assignments]
    if len({normalized_prompt_hash(_temporary_case(item)) for item in authored}) != len(authored):
        raise ValueError("DeepSeek authored duplicate normalized prompts")
    _write_json(output_dir / "authored_cases.json", authored)
    return authored


def _validate_annotation(item: dict[str, Any], case_id: str) -> dict[str, Any]:
    if item.get("case_id") != case_id:
        raise ValueError(f"annotation response changed case_id for {case_id}")
    if item.get("primary_action") not in ACTIONS:
        raise ValueError(f"invalid action for {case_id}: {item.get('primary_action')}")
    if item.get("confidence") not in {"high", "medium", "low"}:
        raise ValueError(f"invalid confidence for {case_id}")
    if len(str(item.get("reason") or "").strip()) < 3:
        raise ValueError(f"missing annotation reason for {case_id}")
    return item


def annotate_cases(
    client: DeepSeekClient,
    cases: list[dict[str, Any]],
    output_dir: Path,
    *,
    annotator: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    annotations_by_id: dict[str, dict[str, Any]] = {}
    batches = list(_chunks(cases, batch_size))
    for batch_number, batch in enumerate(batches, start=1):
        cache = output_dir / "raw" / f"annotator-{annotator}" / f"batch-{batch_number:03d}.json"
        if cache.exists():
            response = _read_json(cache)
        else:
            response = client.json_completion(_annotation_prompt(batch, annotator=annotator))
            _write_json(cache, response)
        items = response.get("annotations") or []
        by_id = {item.get("case_id"): item for item in items}
        for case in batch:
            case_id = case["case_id"]
            item = by_id.get(case_id)
            if item is None:
                raise ValueError(f"annotator {annotator} batch missing {case_id}")
            annotations_by_id[case_id] = _validate_annotation(item, case_id)
        print(f"annotator-{annotator} {batch_number}/{len(batches)} complete", flush=True)
    annotations = [annotations_by_id[item["case_id"]] for item in cases]
    _write_json(output_dir / f"annotations-{annotator}.json", annotations)
    return annotations


def find_conflicts(
    cases: list[dict[str, Any]],
    annotations_a: list[dict[str, Any]],
    annotations_b: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_a = {item["case_id"]: item for item in annotations_a}
    by_b = {item["case_id"]: item for item in annotations_b}
    conflicts = []
    for case in cases:
        a = by_a[case["case_id"]]
        b = by_b[case["case_id"]]
        labels = {a["primary_action"], b["primary_action"], case["author_target_action"]}
        if len(labels) > 1:
            conflicts.append(
                {
                    "case_id": case["case_id"],
                    "blind_case": _blind_case(case),
                    "author_target_action": case["author_target_action"],
                    "annotation_a": a,
                    "annotation_b": b,
                    "adjudication": None,
                }
            )
    return conflicts


def _termination(action: str) -> ExpectedTermination:
    return {
        "search_pois": ExpectedTermination.PLAN,
        "ask_user": ExpectedTermination.CLARIFICATION,
        "propose_tradeoff": ExpectedTermination.TRADEOFF,
        "abort": ExpectedTermination.SAFE_ABORT,
    }[action]


def _temporary_case(item: dict[str, Any]) -> ExternalBenchmarkCase:
    """Build a schema-valid case using the author target before annotation."""
    return _build_case(item, item["author_target_action"], [], None)


def _build_case(
    item: dict[str, Any],
    final_action: str,
    annotations: list[IndependentAnnotation],
    adjudication: Adjudication | None,
) -> ExternalBenchmarkCase:
    capability = {
        "status": item["capability_status"],
        "actionable_alternatives": item["actionable_alternatives"],
        "alternatives": item["alternatives"],
    }
    context = {
        "trajectory_id": "[CURRENT_TRAJECTORY]",
        "goal_version": 1,
        "plan_version": 2 if item["source"] == "long_context_replan" else 1,
        "original_request": item["request"],
        "current_subtask": {
            "task_id": item["case_id"],
            "description": item["request_family"],
            "allowed_actions": list(ACTIONS),
        },
        "hard_constraints": {
            item["constraint_kind"]: item["hard_constraint_description"],
            "trusted_city_cluster": item["city_cluster"],
            "date_pattern": item["date_pattern"],
        },
        "soft_preferences": {},
        "capability": capability,
        "missing_information": item["missing_information"],
        "relevant_fact_refs": ["fact:0"],
        "relevant_artifact_refs": [],
        "relevant_facts": [
            {
                "fact_id": "fact:0",
                "kind": "verified_constraint",
                "value": item["hard_constraint_description"],
                "synthetic_fixture": True,
            }
        ],
        "relevant_artifacts": [],
        "failure_summary": item["failure_summary"],
        "remaining_tasks": 1,
        "remaining_steps": item["remaining_steps"],
        "allowed_actions": list(ACTIONS),
    }
    fault = None
    if item["source"] == "tool_failure":
        fault = FaultInjection(
            fault_type=item["fault_type"], trigger_step=1, recoverable=item["recoverable"]
        )
    category = item["constraint_kind"]
    argument_required: dict[str, Any] = {}
    if final_action == "search_pois" and item["search_keywords"]:
        argument_required = {"keywords": {"must_include_any": item["search_keywords"]}}
    return ExternalBenchmarkCase(
        case_id=f"ext-v1-{item['case_id']}",
        split=BenchmarkSplit(item["split"]),
        source=BenchmarkSource(item["source"]),
        difficulty=item["difficulty"],
        provenance=Provenance(
            authoring_method="simulated",
            permission_basis="User-authorized DeepSeek V4 Flash synthetic evaluation authoring",
            deidentified=True,
            template_independent=True,
            author_group="deepseek-v4-flash-author",
        ),
        group_keys=GroupKeys(
            request_family=item["request_family"],
            city_cluster=item["city_cluster"],
            date_pattern=item["date_pattern"],
            constraint_combo=f"{item['constraint_kind']}:{item['case_id']}",
            failure_template=item["fault_type"] or "none",
        ),
        messages=[
            {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"))},
        ],
        tools=policy_action_schemas(list(ACTIONS)),
        tool_fixture={
            "fixture_id": f"stage29-fixture-{item['case_id']}",
            "verified_constraint": item["hard_constraint_description"],
            "synthetic": True,
        },
        fault_injection=fault,
        allowed_actions=list(ACTIONS),
        argument_rules=[ArgumentRule(action=final_action, required=argument_required)],
        hard_constraints=[
            HardConstraint(
                constraint_id=f"hc-{item['case_id']}",
                kind=item["constraint_kind"],
                description=item["hard_constraint_description"],
                verifier=f"stage29_{item['constraint_kind']}",
                params={"fixture_id": f"stage29-fixture-{item['case_id']}"},
            )
        ],
        acceptable_clarification_categories=[category] if final_action == "ask_user" else [],
        acceptable_tradeoff_categories=[category] if final_action == "propose_tradeoff" else [],
        expected_termination=_termination(final_action),
        outcome_rubric=OutcomeRubric(
            success=f"选择 {final_action}，遵守硬约束且不编造工具事实。",
            partial_success="动作方向正确，但参数或决定性证据说明不完整。",
            failure="忽略题内能力状态、硬约束、替代条件或恢复预算。",
            safe_termination="仅在无安全可行路径且无可执行替代时终止。",
        ),
        max_steps=8 if item["difficulty"] == "L4" else 6,
        annotations=annotations,
        adjudication=adjudication,
    )


def assemble_cases(
    authored: list[dict[str, Any]],
    annotations_a: list[dict[str, Any]],
    annotations_b: list[dict[str, Any]],
    adjudications: dict[str, dict[str, str]],
) -> list[ExternalBenchmarkCase]:
    by_a = {item["case_id"]: item for item in annotations_a}
    by_b = {item["case_id"]: item for item in annotations_b}
    result = []
    for item in authored:
        case_id = item["case_id"]
        a = by_a[case_id]
        b = by_b[case_id]
        labels = {a["primary_action"], b["primary_action"], item["author_target_action"]}
        adjudication = None
        if len(labels) == 1:
            final_action = a["primary_action"]
        else:
            decision = adjudications.get(case_id)
            if decision is None:
                raise ValueError(f"missing adjudication for conflict {case_id}")
            final_action = decision["primary_action"]
            if final_action not in ACTIONS:
                raise ValueError(f"invalid adjudicated action for {case_id}")
            adjudication = Adjudication(
                adjudicator_id="codex-stage29-adjudicator",
                primary_action=final_action,
                reason=decision["reason"],
            )
        annotations = [
            IndependentAnnotation(
                annotator_id="deepseek-v4-flash-blind-a",
                primary_action=a["primary_action"],
                allowed_actions=list(ACTIONS),
                hard_constraint_labels={f"hc-{case_id}": True},
                notes=f"confidence={a['confidence']}; {a['reason']}",
            ),
            IndependentAnnotation(
                annotator_id="deepseek-v4-flash-blind-b",
                primary_action=b["primary_action"],
                allowed_actions=list(ACTIONS),
                hard_constraint_labels={f"hc-{case_id}": True},
                notes=f"confidence={b['confidence']}; {b['reason']}",
            ),
        ]
        result.append(_build_case(item, final_action, annotations, adjudication))
    return result


def build_manifest(
    cases: list[ExternalBenchmarkCase],
    authored: list[dict[str, Any]],
    annotations_a: list[dict[str, Any]],
    annotations_b: list[dict[str, Any]],
    conflict_count: int,
) -> dict[str, Any]:
    labels_a = [item["primary_action"] for item in annotations_a]
    labels_b = [item["primary_action"] for item in annotations_b]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen-ai-external-evaluation-set",
        "model": MODEL,
        "cases": len(cases),
        "split_counts": dict(Counter(case.split.value for case in cases)),
        "source_counts": dict(Counter(case.source.value for case in cases)),
        "author_target_action_counts": dict(Counter(item["author_target_action"] for item in authored)),
        "final_action_counts": dict(Counter(case.annotations[0].primary_action if case.adjudication is None else case.adjudication.primary_action for case in cases)),
        "unique_case_ids": len({case.case_id for case in cases}),
        "unique_normalized_prompts": len({normalized_prompt_hash(case) for case in cases}),
        "double_annotated_cases": sum(len(case.annotations) >= 2 for case in cases),
        "primary_action_kappa": cohens_kappa(labels_a, labels_b),
        "conflicts_requiring_adjudication": conflict_count,
        "unresolved_conflicts": 0,
        "independent_human_authors": 0,
        "independent_human_annotators": 0,
        "eligible_for_independent_human_benchmark_claim": False,
        "eligible_for_external_model_evaluation_claim": True,
        "content_sha256": canonical_hash(
            sorted(canonical_hash(case.model_dump(mode="json")) for case in cases)
        ),
        "limitations": [
            "All cases are synthetic and authored by DeepSeek V4 Flash.",
            "Both blind annotation passes use the same model family with isolated prompts, not independent humans.",
            "Codex adjudicates author/annotator label conflicts after generation.",
            "The set may compare candidate models but cannot substantiate an independent human benchmark claim.",
        ],
    }


def _load_adjudications(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    values = _read_json(path)
    return {item["case_id"]: item for item in values}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("plan", "author", "annotate", "conflicts", "finalize", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 20:
        parser.error("--batch-size must be between 1 and 20")

    assignments = build_assignments(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "assignments.json", assignments)
    _write_json(
        args.output_dir / "plan.json",
        {
            "schema_version": SCHEMA_VERSION,
            "seed": args.seed,
            "cases": len(assignments),
            "split_counts": dict(Counter(item["split"] for item in assignments)),
            "source_counts": dict(Counter(item["source"] for item in assignments)),
            "action_counts": dict(Counter(item["target_action"] for item in assignments)),
            "assignment_sha256": canonical_hash(assignments),
            "claim_boundary": "AI external evaluation set; not an independent human benchmark.",
        },
    )
    if args.phase == "plan":
        return 0

    authored_path = args.output_dir / "authored_cases.json"
    annotations_a_path = args.output_dir / "annotations-a.json"
    annotations_b_path = args.output_dir / "annotations-b.json"
    needs_api = args.phase in {"author", "annotate", "all"}
    client = None
    if needs_api:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise SystemExit("DEEPSEEK_API_KEY must be set in the current process")
        client = DeepSeekClient(api_key)

    if args.phase in {"author", "all"}:
        assert client is not None
        authored = author_cases(client, assignments, args.output_dir, batch_size=args.batch_size)
    else:
        authored = _read_json(authored_path)
    if args.phase == "author":
        return 0

    if args.phase in {"annotate", "all"}:
        assert client is not None
        annotations_a = annotate_cases(client, authored, args.output_dir, annotator="a", batch_size=args.batch_size)
        annotations_b = annotate_cases(client, authored, args.output_dir, annotator="b", batch_size=args.batch_size)
    else:
        annotations_a = _read_json(annotations_a_path)
        annotations_b = _read_json(annotations_b_path)
    if args.phase == "annotate":
        return 0

    conflicts = find_conflicts(authored, annotations_a, annotations_b)
    _write_json(args.output_dir / "conflicts.json", conflicts)
    if args.phase in {"conflicts", "all"}:
        print(json.dumps({"conflicts": len(conflicts)}, ensure_ascii=False), flush=True)
        if args.phase == "all" and conflicts:
            print("Adjudicate conflicts.json into adjudications.json before --phase finalize.", flush=True)
        return 0

    adjudications = _load_adjudications(args.output_dir / "adjudications.json")
    cases = assemble_cases(authored, annotations_a, annotations_b, adjudications)
    dev = [case for case in cases if case.split == BenchmarkSplit.DEV]
    sealed = [case for case in cases if case.split == BenchmarkSplit.SEALED_TEST]
    (args.output_dir / "dev.jsonl").write_text(
        "\n".join(case.model_dump_json() for case in dev) + "\n", encoding="utf-8"
    )
    (args.output_dir / "sealed_test.jsonl").write_text(
        "\n".join(case.model_dump_json() for case in sealed) + "\n", encoding="utf-8"
    )
    manifest = build_manifest(cases, authored, annotations_a, annotations_b, len(conflicts))
    _write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
