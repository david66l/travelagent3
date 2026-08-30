"""Classify paired Stage 3 failures without changing efficacy outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PairKey = tuple[str, int, int]


def _load(path: Path) -> dict[PairKey, dict[str, Any]]:
    rows: dict[PairKey, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (
            str(row["task_id"]),
            int(row["sample_index"]),
            int(row["rollout_seed"]),
        )
        if key in rows:
            raise ValueError(f"duplicate rollout key at {path}:{line_number}: {key}")
        rows[key] = row
    return rows


def _success(row: dict[str, Any]) -> bool:
    return row.get("gate_status") == "passed" and bool(
        (row.get("audit_metrics") or {}).get("hard_pass")
    )


def _error_codes(row: dict[str, Any]) -> list[str]:
    codes = {
        str(action.get("error_code"))
        for action in row.get("actions") or []
        if action.get("error_code")
    }
    for error in row.get("policy_errors") or []:
        if isinstance(error, dict) and error.get("code"):
            codes.add(str(error["code"]))
    return sorted(codes)


def _classify(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    snapshot_mismatch_owner: str,
) -> tuple[str, str]:
    codes = set(_error_codes(left)) | set(_error_codes(right))
    if "POLICY_ARGUMENT_INVALID" in codes:
        return (
            "policy_argument_schema_violation",
            "model_fault",
        )
    if "SNAPSHOT_ARGUMENT_MISMATCH" in codes:
        return (
            "snapshot_argument_contract_mismatch",
            snapshot_mismatch_owner,
        )
    if any((row.get("audit_metrics") or {}).get("duplicate_calls", 0) for row in (left, right)):
        return "repeated_call_without_recovery", "model_fault"
    return "unclassified_shared_failure", "needs_review"


def build_report(
    baseline_path: Path,
    candidate_path: Path,
    *,
    snapshot_mismatch_owner: str = "needs_review",
) -> dict[str, Any]:
    if snapshot_mismatch_owner not in {
        "evaluation_harness_fault",
        "model_fault",
        "needs_review",
    }:
        raise ValueError(f"invalid snapshot mismatch owner: {snapshot_mismatch_owner}")
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate rollout keys differ")

    shared_failures = []
    candidate_only_improvements = []
    baseline_only_regressions = []
    root_causes: Counter[str] = Counter()
    fault_owners: Counter[str] = Counter()
    improvement_causes: Counter[str] = Counter()
    regression_causes: Counter[str] = Counter()
    for key in sorted(baseline):
        left = baseline[key]
        right = candidate[key]
        left_success = _success(left)
        right_success = _success(right)
        if not left_success and right_success:
            root_cause, fault_owner = _classify(
                left,
                left,
                snapshot_mismatch_owner=snapshot_mismatch_owner,
            )
            improvement_causes[root_cause] += 1
            candidate_only_improvements.append(
                {
                    "task_id": key[0],
                    "sample_index": key[1],
                    "rollout_seed": key[2],
                    "baseline_root_cause": root_cause,
                    "fault_owner": fault_owner,
                    "baseline_error_codes": _error_codes(left),
                }
            )
            continue
        if left_success and not right_success:
            root_cause, fault_owner = _classify(
                right,
                right,
                snapshot_mismatch_owner=snapshot_mismatch_owner,
            )
            regression_causes[root_cause] += 1
            baseline_only_regressions.append(
                {
                    "task_id": key[0],
                    "sample_index": key[1],
                    "rollout_seed": key[2],
                    "candidate_root_cause": root_cause,
                    "fault_owner": fault_owner,
                    "candidate_error_codes": _error_codes(right),
                }
            )
            continue
        if left_success and right_success:
            continue
        root_cause, fault_owner = _classify(
            left,
            right,
            snapshot_mismatch_owner=snapshot_mismatch_owner,
        )
        root_causes[root_cause] += 1
        fault_owners[fault_owner] += 1
        shared_failures.append(
            {
                "task_id": key[0],
                "sample_index": key[1],
                "rollout_seed": key[2],
                "root_cause": root_cause,
                "fault_owner": fault_owner,
                "baseline_error_codes": _error_codes(left),
                "candidate_error_codes": _error_codes(right),
                "baseline_termination": left.get("termination_reason"),
                "candidate_termination": right.get("termination_reason"),
            }
        )

    return {
        "schema_version": "stage3-paired-failure-analysis.v1",
        "scope": "diagnostic classification only; original success labels are unchanged",
        "snapshot_mismatch_owner": snapshot_mismatch_owner,
        "paired_rollouts": len(baseline),
        "shared_failure_count": len(shared_failures),
        "root_cause_counts": dict(sorted(root_causes.items())),
        "fault_owner_counts": dict(sorted(fault_owners.items())),
        "candidate_only_improvement_count": len(candidate_only_improvements),
        "candidate_only_improvement_causes": dict(sorted(improvement_causes.items())),
        "baseline_only_regression_count": len(baseline_only_regressions),
        "baseline_only_regression_causes": dict(sorted(regression_causes.items())),
        "recommended_actions": {
            "evaluation_harness_fault": (
                "add a context-aware immutable snapshot contract and rerun a new frozen suite"
            ),
            "model_fault": (
                "route failures by root cause instead of applying one decoding fix to every model fault"
            ),
            "by_root_cause": {
                "policy_argument_schema_violation": (
                    "enable state-scoped Qwen tool-envelope constrained decoding and retain "
                    "rejection telemetry"
                ),
                "snapshot_argument_contract_mismatch": (
                    "when the immutable snapshot is correct, add semantically diverse verified "
                    "repair SFT before another on-policy GRPO round"
                ),
                "repeated_call_without_recovery": (
                    "train no-progress recovery choices and keep the bounded repeat guard enabled"
                ),
            },
        },
        "shared_failures": shared_failures,
        "candidate_only_improvements": candidate_only_improvements,
        "baseline_only_regressions": baseline_only_regressions,
    }


def render_markdown(report: dict[str, Any]) -> str:
    rows = [
        "# Stage3 成对共同失败分析",
        "",
        f"- 配对轨迹：{report['paired_rollouts']}",
        f"- 共同失败：{report['shared_failure_count']}",
        "- 本报告只做根因诊断，不修改原始成功标签。",
        "",
        "## 根因",
        "",
        "| 根因 | 数量 | 归属 |",
        "|---|---:|---|",
    ]
    owners = {item["root_cause"]: item["fault_owner"] for item in report["shared_failures"]}
    rows.extend(
        f"| {cause} | {count} | {owners[cause]} |"
        for cause, count in report["root_cause_counts"].items()
    )
    rows.extend(
        [
            "",
            "## 处理原则",
            "",
            "- 快照参数不一致先人工判定归属：夹具错误要修测试集；夹具正确则属于语义决策错误，应进入已验证的修复 SFT 数据。",
            "- 非法额外字段属于模型/解码协议问题，应通过状态级 JSON Schema 约束并保留拒绝日志。",
            "- 修复后重新跑冻结评测；不得回写或篡改本轮原始报告。",
            "",
        ]
    )
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-rollouts", type=Path, required=True)
    parser.add_argument("--candidate-rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--snapshot-mismatch-owner",
        choices=("evaluation_harness_fault", "model_fault", "needs_review"),
        default="needs_review",
        help="Owner assigned after manual semantic review of snapshot mismatch arguments.",
    )
    args = parser.parse_args()
    report = build_report(
        args.baseline_rollouts,
        args.candidate_rollouts,
        snapshot_mismatch_owner=args.snapshot_mismatch_owner,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
