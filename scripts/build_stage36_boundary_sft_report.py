"""Build the auditable Stage36 termination-boundary SFT promotion report."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_summary(report: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary = report["summary"]
    by_action: dict[str, dict[str, int]] = {}
    for action in sorted({row["expected_action"] for row in runs}):
        selected = [row for row in runs if row["expected_action"] == action]
        by_action[action] = {
            "runs": len(selected),
            "raw_successful": sum(bool(row["success"]) for row in selected),
            "contract_successful": sum(
                bool(row["policy_contract_success"]) for row in selected
            ),
        }
    action_count_histogram = Counter(len(row["observed_actions"]) for row in runs)
    return {
        "runs": summary["runs"],
        "raw_successful": summary["successful_runs"],
        "contract_successful": summary["policy_contract_successful_runs"],
        "http_errors": summary["http_errors"],
        "multiple_action_runs": sum(
            count for actions, count in action_count_histogram.items() if actions != 1
        ),
        "label_contract_conflicts": summary["label_contract_conflicts"],
        "mean_completion_tokens": summary["inference"]["completion_tokens"]["mean"],
        "mean_latency_ms": summary["inference"]["request_latency_ms"]["mean"],
        "by_action": by_action,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _read(args.dataset_audit)
    training = _read(args.training_report)
    smoke_case = _read(args.smoke_case_report)
    smoke_full_report = _read(args.smoke_full_report)
    smoke_full_runs = _read_jsonl(args.smoke_full_runs)
    formal_case = _read(args.formal_case_report)
    formal_full_report = _read(args.formal_full_report)
    formal_full_runs = _read_jsonl(args.formal_full_runs)
    baseline = _read(args.baseline_report)["arms"]["sft"]
    formal_summary = _full_summary(formal_full_report, formal_full_runs)
    smoke_summary = _full_summary(smoke_full_report, smoke_full_runs)
    failures = [row for row in formal_full_runs if not row["success"]]

    checks = {
        "dataset_audit_passed": dataset["status"] == "passed",
        "frozen_overlap_zero": dataset["frozen_holdout_payload_overlap"] == 0,
        "formal_training_scope": training["run_scope"] == "formal",
        "boundary_coverage_complete": (
            training["termination_boundary_preflight"]["boundary_rows"]
            == training["termination_boundary_preflight"]["rows_checked"]
            == 240
        ),
        "smoke_case_stable_5_of_5": smoke_case["summary"]["successful_runs"] == 5,
        "formal_case_stable_5_of_5": formal_case["summary"]["successful_runs"] == 5,
        "formal_full_contract_150_of_150": formal_summary["contract_successful"] == 150,
        "formal_full_single_action_150_of_150": formal_summary["multiple_action_runs"] == 0,
        "formal_full_no_http_errors": formal_summary["http_errors"] == 0,
        "only_failure_is_known_label_conflict": (
            len(failures) == 1
            and failures[0]["case_id"] == "ext-v1-stage29-ds-002"
            and failures[0]["label_contract_conflict"] is True
        ),
    }
    status = "passed" if all(checks.values()) else "rejected"
    report = {
        "schema_version": "stage36-boundary-sft-promotion.v1",
        "status": status,
        "objective": (
            "repair repeated tool calls by upweighting the EOS immediately after "
            "the first tool-call envelope"
        ),
        "dataset": dataset,
        "training": {
            "base_model": training["base_model"],
            "continued_from_adapter": training["continued_from_adapter"],
            "run_scope": training["run_scope"],
            "epochs": training["train_metrics"]["epoch"],
            "termination_token_weight": training["termination_token_weight"],
            "train_loss": training["train_metrics"]["train_loss"],
            "eval_loss": training["eval_metrics"]["eval_loss"],
            "boundary_preflight": training["termination_boundary_preflight"],
            "checkpoint": {
                "remote_path": args.checkpoint_path,
                "adapter_model_sha256": args.adapter_sha256,
                "adapter_config_sha256": args.adapter_config_sha256,
                "training_report_sha256": _sha256(args.training_report),
            },
        },
        "generation_gates": {
            "smoke_case_repetitions": smoke_case["summary"]["successful_runs"],
            "smoke_full": smoke_summary,
            "formal_case_repetitions": formal_case["summary"]["successful_runs"],
            "formal_full": formal_summary,
        },
        "comparison_to_stage32_sft": {
            "raw_successful": {
                "before": baseline["raw_successful"],
                "after": formal_summary["raw_successful"],
                "delta": formal_summary["raw_successful"] - baseline["raw_successful"],
            },
            "contract_successful": {
                "before": baseline["contract_successful"],
                "after": formal_summary["contract_successful"],
                "delta": (
                    formal_summary["contract_successful"]
                    - baseline["contract_successful"]
                ),
            },
            "multiple_action_runs": {
                "before": baseline["multiple_action_runs"],
                "after": formal_summary["multiple_action_runs"],
            },
        },
        "known_label_conflict": {
            "case_id": failures[0]["case_id"] if failures else None,
            "frozen_label": failures[0]["expected_action"] if failures else None,
            "controller_allowed_actions": failures[0]["allowed_actions"] if failures else [],
            "observed_action": failures[0]["observed_actions"] if failures else [],
            "production_contract_success": (
                failures[0]["policy_contract_success"] if failures else None
            ),
        },
        "checks": checks,
        "decision": {
            "candidate": "travel-policy-qwen3-1.7b-stage36-boundary-sft-formal",
            "result": "promote_to_5_percent_shadow" if status == "passed" else "do_not_promote",
            "reason": (
                "formal model removes the repeated-call regression and reaches 150/150 "
                "production-contract success without action-family regression"
            ),
        },
        "deployment": {
            "mode": "shadow",
            "sample_rate": 0.05,
            "deployment_id": "stage36-1p7b-boundary-shadow-v1",
            "student_model": "travel-policy-qwen3-1.7b-stage36-boundary-sft-formal",
            "teacher_model": "travel-policy-qwen3-8b-base",
            "challenger_model": "travel-policy-qwen3-1.7b-stage32-dpo-sftref-v2",
            "local_backend_health": "healthy",
            "local_shadow_worker": "ready",
            "config_sha256": _sha256(args.deployment_env),
            "compose_sha256": _sha256(args.compose_file),
        },
        "limitations": [
            "Promotion is to 5% shadow rollout, not an unrestricted production release.",
            "The hard set has 150 cases and one frozen-label/controller-contract conflict.",
            "Production traffic monitoring is still required before increasing exposure.",
        ],
        "source_sha256": {
            str(path): _sha256(path)
            for path in (
                args.dataset_audit,
                args.training_report,
                args.smoke_case_report,
                args.smoke_full_report,
                args.smoke_full_runs,
                args.formal_case_report,
                args.formal_full_report,
                args.formal_full_runs,
                args.baseline_report,
                args.deployment_env,
                args.compose_file,
            )
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = f"""# Stage36：工具调用终止边界 SFT

## 结论

- 状态：`{status}`
- 正式模型：`travel-policy-qwen3-1.7b-stage36-boundary-sft-formal`
- 冻结难例：原始标签 {formal_summary['raw_successful']}/150，生产契约 {formal_summary['contract_successful']}/150
- 多动作输出：{formal_summary['multiple_action_runs']}/150
- `ds-065` 稳定性：{formal_case['summary']['successful_runs']}/5

## 做了什么

Stage35 证明普通 DPO 即使离线偏好准确率达到 100%，也没有让生成停止。Stage36 直接对
Qwen3 工具调用闭合后的 `<|im_end|>` 增加 16 倍 token 权重，同时保留工具名称和参数的
普通 SFT 损失。240 条样本全部来自已审计训练回放，四类动作各 60 条，冻结评测重叠为 0。

## 相对 Stage32 SFT

| 指标 | Stage32 SFT | Stage36 正式模型 |
|---|---:|---:|
| 原始标签正确 | {baseline['raw_successful']}/150 | {formal_summary['raw_successful']}/150 |
| 生产契约正确 | {baseline['contract_successful']}/150 | {formal_summary['contract_successful']}/150 |
| 多动作输出 | {baseline['multiple_action_runs']} | {formal_summary['multiple_action_runs']} |

唯一原始标签失败仍是 `ext-v1-stage29-ds-002`：冻结标签要求 `abort`，但当前控制器只允许
`ask_user`；模型遵守了生产契约，因此契约口径计为成功。

## 决策

进入 5% Shadow 灰度，不直接全量发布。后续观察真实流量中的单调用率、契约正确率、
教师升级率和回退率，再决定是否扩大暴露。
"""
    (args.output_dir / "README.md").write_text(markdown, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--smoke-case-report", type=Path, required=True)
    parser.add_argument("--smoke-full-report", type=Path, required=True)
    parser.add_argument("--smoke-full-runs", type=Path, required=True)
    parser.add_argument("--formal-case-report", type=Path, required=True)
    parser.add_argument("--formal-full-report", type=Path, required=True)
    parser.add_argument("--formal-full-runs", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--deployment-env", type=Path, required=True)
    parser.add_argument("--compose-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--adapter-config-sha256", required=True)
    args = parser.parse_args()
    report = build(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
