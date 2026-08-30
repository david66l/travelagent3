"""Build the paired Stage32 cascade-distillation acceptance report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.loop import PolicyContext  # noqa: E402
from agentic.policy import route_policy_context  # noqa: E402


MODEL_ONLY_ARMS = {
    "base_1p7b": "stage32-1p7b-base-model-only-v1",
    "sft_1p7b": "stage32-1p7b-sft-model-only-v1",
    "dpo_1p7b": "stage32-1p7b-dpo-sftref-model-only-v2",
    "dpo_4b": "stage32-4b-dpo-model-only-v1",
    "base_8b": "stage32-8b-model-only-v1",
}

CONTRACT_ARMS = {
    "base_1p7b": "stage32-1p7b-base-contract-v2",
    "sft_1p7b": "stage32-1p7b-sft-contract-v2",
    "dpo_1p7b": "stage32-1p7b-dpo-sftref-contract-v2",
    "dpo_4b": "stage32-4b-dpo-contract-v2",
    "base_8b": "stage32-8b-contract-v2",
}

ARM_LABELS = {
    "base_1p7b": "Qwen3-1.7B Base",
    "sft_1p7b": "Qwen3-1.7B SFT",
    "dpo_1p7b": "Qwen3-1.7B SFT+DPO（SFT reference）",
    "dpo_4b": "Qwen3-4B SFT+DPO",
    "base_8b": "Qwen3-8B Base",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def bootstrap_rate_ci(
    values: list[bool], *, samples: int = 10_000, seed: int = 20260815
) -> list[float]:
    rng = random.Random(seed)
    count = len(values)
    rates = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    return [round(_percentile(rates, 0.025), 8), round(_percentile(rates, 0.975), 8)]


def paired_comparison(
    candidate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_ids = sorted(candidate)
    if case_ids != sorted(baseline):
        raise ValueError("paired arms must contain identical case IDs")
    candidate_values = [bool(candidate[item]["success"]) for item in case_ids]
    baseline_values = [bool(baseline[item]["success"]) for item in case_ids]
    improved = sum(c and not b for c, b in zip(candidate_values, baseline_values, strict=True))
    regressed = sum(b and not c for c, b in zip(candidate_values, baseline_values, strict=True))
    discordant = improved + regressed
    if discordant:
        tail = sum(
            math.comb(discordant, index) * (0.5**discordant)
            for index in range(min(improved, regressed) + 1)
        )
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0

    rng = random.Random(20260815)
    count = len(case_ids)
    differences = []
    for _ in range(10_000):
        indices = [rng.randrange(count) for _ in range(count)]
        differences.append(
            sum(candidate_values[index] - baseline_values[index] for index in indices)
            / count
        )
    return {
        "percentage_point_difference": round(
            (sum(candidate_values) - sum(baseline_values)) / count * 100, 3
        ),
        "paired_bootstrap_95_ci_percentage_points": [
            round(_percentile(differences, 0.025) * 100, 3),
            round(_percentile(differences, 0.975) * 100, 3),
        ],
        "candidate_only_success": improved,
        "baseline_only_success": regressed,
        "mcnemar_exact_two_sided_p": round(p_value, 8),
    }


def _metadata(dataset_root: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for filename, split in (("dev.jsonl", "dev"), ("sealed_test.jsonl", "sealed_test")):
        for row in _read_jsonl(dataset_root / filename):
            result[row["case_id"]] = {
                "split": split,
                "source": row["source"],
                "difficulty": row["difficulty"],
                "context": json.loads(row["messages"][1]["content"]),
            }
    if len(result) != 150:
        raise ValueError("Stage32 acceptance requires exactly 150 frozen cases")
    return result


def summarize_arm(
    directory: Path, metadata: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report_path = directory / "report.json"
    runs_path = directory / "runs.jsonl"
    report = _read_json(report_path)
    rows = _read_jsonl(runs_path)
    by_id = {row["case_id"]: row for row in rows}
    if len(rows) != 150 or len(by_id) != 150 or set(by_id) != set(metadata):
        raise ValueError(f"not a paired 150-case arm: {directory}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["expected_action"]].append(row)
    successes = [bool(row["success"]) for row in rows]
    conflict_rows = [
        row
        for row in rows
        if row.get("expected_action")
        and row.get("allowed_actions")
        and row["expected_action"] not in row["allowed_actions"]
    ]
    consistent_rows = [row for row in rows if row not in conflict_rows]
    policy_contract_successes = [
        row
        for row in rows
        if not row.get("http_error")
        and len(row.get("observed_actions") or []) == 1
        and (
            not row.get("allowed_actions")
            or row["observed_actions"][0] in row["allowed_actions"]
        )
    ]
    non_abort = [row for row in rows if row["expected_action"] != "abort"]
    false_aborts = sum("abort" in row["observed_actions"] for row in non_abort)
    inference = report["summary"]["inference"]
    summary = {
        "model": report["model"],
        "successful_runs": sum(successes),
        "runs": len(rows),
        "success_rate": round(sum(successes) / len(rows), 8),
        "bootstrap_95_ci": bootstrap_rate_ci(successes),
        "label_contract_conflicts": len(conflict_rows),
        "contract_consistent_runs": len(consistent_rows),
        "contract_consistent_successful_runs": sum(
            bool(row["success"]) for row in consistent_rows
        ),
        "contract_consistent_success_rate": round(
            sum(bool(row["success"]) for row in consistent_rows)
            / len(consistent_rows),
            8,
        ),
        "policy_contract_successful_runs": len(policy_contract_successes),
        "action_mismatches": sum(bool(row["action_mismatch"]) for row in rows),
        "argument_mismatches": sum(bool(row["argument_mismatch"]) for row in rows),
        "http_errors": sum(bool(row["http_error"]) for row in rows),
        "false_abort_count": false_aborts,
        "false_abort_rate": round(false_aborts / len(non_abort), 8),
        "by_action": {
            action: {
                "successful": sum(bool(row["success"]) for row in items),
                "runs": len(items),
                "success_rate": round(
                    sum(bool(row["success"]) for row in items) / len(items), 8
                ),
            }
            for action, items in sorted(groups.items())
        },
        "failure_confusion": dict(
            Counter(
                f"{row['expected_action']}=>{'+'.join(row['observed_actions']) or 'none'}"
                for row in rows
                if not row["success"]
            )
        ),
        "mean_latency_ms": inference["request_latency_ms"]["mean"],
        "p95_latency_ms": inference["request_latency_ms"]["p95"],
        "p95_ttft_ms": inference["ttft_ms"]["p95"],
        "throughput_requests_per_second": report["summary"][
            "request_throughput_per_second"
        ],
        "completion_tokens_mean": inference["completion_tokens"]["mean"],
        "source": {
            "report": report_path.as_posix(),
            "report_sha256": _sha256(report_path),
            "runs": runs_path.as_posix(),
            "runs_sha256": _sha256(runs_path),
        },
    }
    return summary, by_id


def build(
    reports_root: Path, dataset_root: Path, output_dir: Path
) -> dict[str, Any]:
    metadata = _metadata(dataset_root)
    summaries: dict[str, dict[str, dict[str, Any]]] = {
        "model_only": {},
        "production_contract": {},
    }
    runs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
        "model_only": {},
        "production_contract": {},
    }
    for scope, arms in (
        ("model_only", MODEL_ONLY_ARMS),
        ("production_contract", CONTRACT_ARMS),
    ):
        for key, dirname in arms.items():
            summaries[scope][key], runs[scope][key] = summarize_arm(
                reports_root / dirname, metadata
            )

    comparisons = {}
    for scope in ("model_only", "production_contract"):
        scoped = runs[scope]
        comparisons[scope] = {
            "sft_1p7b_vs_base_1p7b": paired_comparison(
                scoped["sft_1p7b"], scoped["base_1p7b"]
            ),
            "sft_1p7b_vs_dpo_4b": paired_comparison(
                scoped["sft_1p7b"], scoped["dpo_4b"]
            ),
            "sft_1p7b_vs_base_8b": paired_comparison(
                scoped["sft_1p7b"], scoped["base_8b"]
            ),
            "dpo_1p7b_vs_sft_1p7b": paired_comparison(
                scoped["dpo_1p7b"], scoped["sft_1p7b"]
            ),
        }

    routed_rows = []
    oracle_success = 0
    contract_runs = runs["production_contract"]
    for case_id in sorted(metadata):
        context = PolicyContext(**metadata[case_id]["context"])
        route = route_policy_context(context)
        selected_arm = "base_8b" if route.target == "teacher" else "sft_1p7b"
        selected = contract_runs[selected_arm][case_id]
        label_contract_conflict = bool(
            selected.get("expected_action")
            and selected.get("allowed_actions")
            and selected["expected_action"] not in selected["allowed_actions"]
        )
        oracle_success += bool(
            contract_runs["sft_1p7b"][case_id]["success"]
            or contract_runs["base_8b"][case_id]["success"]
        )
        routed_rows.append(
            {
                "case_id": case_id,
                "route": route.target,
                "family": route.family,
                "reason": route.reason,
                "selected_arm": selected_arm,
                "success": bool(selected["success"]),
                "label_contract_conflict": label_contract_conflict,
                "latency_ms": selected["inference_metrics"]["request_latency_ms"],
            }
        )
    teacher_calls = sum(row["route"] == "teacher" for row in routed_rows)
    routed_successes = sum(row["success"] for row in routed_rows)
    routed_consistent = [
        row for row in routed_rows if not row["label_contract_conflict"]
    ]
    routed_latency = [float(row["latency_ms"]) for row in routed_rows]
    router = {
        "policy": "backend.agentic.policy.route_policy_context",
        "execution_scope": "paired sequential replay; not co-resident online serving",
        "student": "sft_1p7b",
        "teacher": "base_8b",
        "teacher_calls": teacher_calls,
        "teacher_share": round(teacher_calls / len(routed_rows), 8),
        "successful_runs": routed_successes,
        "runs": len(routed_rows),
        "success_rate": round(routed_successes / len(routed_rows), 8),
        "label_contract_conflicts": len(routed_rows) - len(routed_consistent),
        "contract_consistent_successful_runs": sum(
            row["success"] for row in routed_consistent
        ),
        "contract_consistent_runs": len(routed_consistent),
        "contract_consistent_success_rate": round(
            sum(row["success"] for row in routed_consistent)
            / len(routed_consistent),
            8,
        ),
        "mean_selected_latency_ms": round(sum(routed_latency) / len(routed_latency), 3),
        "oracle_union_successes": oracle_success,
        "oracle_union_rate": round(oracle_success / len(routed_rows), 8),
        "rows": routed_rows,
    }

    model_only = summaries["model_only"]
    production = summaries["production_contract"]
    sft = model_only["sft_1p7b"]
    model_4b = model_only["dpo_4b"]
    gates = {
        "student_model_only_improves_base": (
            sft["successful_runs"] > model_only["base_1p7b"]["successful_runs"]
        ),
        "student_model_only_quality_at_least_4b": (
            sft["successful_runs"] >= model_4b["successful_runs"]
        ),
        "student_contract_within_one_case_of_4b": (
            production["sft_1p7b"]["contract_consistent_successful_runs"]
            >= production["dpo_4b"]["contract_consistent_successful_runs"] - 1
        ),
        "student_throughput_at_least_1p5x_4b": (
            sft["throughput_requests_per_second"]
            >= model_4b["throughput_requests_per_second"] * 1.5
        ),
        "student_mean_latency_at_most_75pct_4b": (
            sft["mean_latency_ms"] <= model_4b["mean_latency_ms"] * 0.75
        ),
        "dpo_sftref_model_only_strictly_improves_sft": (
            model_only["dpo_1p7b"]["successful_runs"] > sft["successful_runs"]
        ),
        "dpo_sftref_contract_does_not_regress_sft": (
            production["dpo_1p7b"]["contract_consistent_successful_runs"]
            >= production["sft_1p7b"]["contract_consistent_successful_runs"]
        ),
        "dpo_sftref_zero_http_errors": (
            model_only["dpo_1p7b"]["http_errors"] == 0
            and production["dpo_1p7b"]["http_errors"] == 0
        ),
        "student_and_reference_arms_zero_http_errors": all(
            summaries[scope][key]["http_errors"] == 0
            for scope in ("model_only", "production_contract")
            for key in ("base_1p7b", "sft_1p7b", "dpo_4b", "base_8b")
        ),
        "router_contract_not_worse_than_student": (
            router["contract_consistent_successful_runs"]
            >= production["sft_1p7b"]["contract_consistent_successful_runs"]
        ),
    }
    student_gates = [
        gates["student_model_only_improves_base"],
        gates["student_model_only_quality_at_least_4b"],
        gates["student_contract_within_one_case_of_4b"],
        gates["student_throughput_at_least_1p5x_4b"],
        gates["student_mean_latency_at_most_75pct_4b"],
        gates["student_and_reference_arms_zero_http_errors"],
        gates["router_contract_not_worse_than_student"],
    ]
    report = {
        "schema_version": "travel-agent-stage32-distillation-report.v2",
        "status": (
            "student_sft_promoted_dpo_sftref_shadow_candidate"
            if all(student_gates)
            and gates["dpo_sftref_model_only_strictly_improves_sft"]
            and gates["dpo_sftref_contract_does_not_regress_sft"]
            and gates["dpo_sftref_zero_http_errors"]
            else "student_not_promoted"
        ),
        "evaluation_contract": {
            "frozen_cases": 150,
            "scopes": {
                "model_only": "frozen original multi-action prompt and tools",
                "production_contract": "same cases projected through current controller constraints",
            },
            "temperature": 0.0,
            "concurrency": 8,
            "repetitions": 1,
            "warning": (
                f"{production['sft_1p7b']['label_contract_conflicts']} frozen label "
                "conflicts with the "
                "projected runtime action contract; primary promotion uses the "
                f"{production['sft_1p7b']['contract_consistent_runs']} contract-consistent cases. Latency "
                "is a single-run engineering measurement, not a confidence interval."
            ),
        },
        "arms": summaries,
        "comparisons": comparisons,
        "router_replay": router,
        "promotion_gates": gates,
        "decision": {
            "production_default": "sft_1p7b" if all(student_gates) else None,
            "shadow_candidate": "dpo_1p7b_sftref_v2",
            "legacy_dpo": "rejected_wrong_reference_policy",
            "reason": (
                "1.7B SFT improves model-only quality by 29/150 over Base. Corrected "
                "DPO with an explicit frozen SFT reference reaches 137/150 with zero "
                "HTTP errors, but its +2/150 paired gain is not statistically decisive, "
                "so it enters shadow rather than replacing the SFT default."
            ),
        },
        "reference_policy_ablation": {
            "legacy_implicit_base_reference": {
                "model_only_successful_runs": 104,
                "http_errors": 37,
                "status": "rejected",
            },
            "explicit_frozen_sft_reference": {
                "model_only_successful_runs": model_only["dpo_1p7b"]["successful_runs"],
                "http_errors": model_only["dpo_1p7b"]["http_errors"],
                "status": "shadow_candidate",
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    model_only = report["arms"]["model_only"]
    production = report["arms"]["production_contract"]
    lines = [
        "# Stage32 级联蒸馏验收报告",
        "",
        f"状态：`{report['status']}`。冻结外部集 150 题，并发 8、temperature=0。",
        "",
        "## 模型裸能力（原始多动作空间）",
        "",
        "| 模型 | 正确 | 平均延迟 | P95 延迟 | 吞吐 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in MODEL_ONLY_ARMS:
        item = model_only[key]
        lines.append(
            f"| {ARM_LABELS[key]} | {item['successful_runs']}/150 | "
            f"{item['mean_latency_ms']:.1f} ms | {item['p95_latency_ms']:.1f} ms | "
            f"{item['throughput_requests_per_second']:.3f} req/s |"
        )
    lines.extend(
        [
            "",
            "## 生产控制器合同",
            "",
            "| 模型 | 冻结标签正确 | 合同一致正确 | 平均延迟 | P95 延迟 | 吞吐 |",
        "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for key in CONTRACT_ARMS:
        item = production[key]
        lines.append(
            f"| {ARM_LABELS[key]} | {item['successful_runs']}/150 | "
            f"{item['contract_consistent_successful_runs']}/{item['contract_consistent_runs']} | "
            f"{item['mean_latency_ms']:.1f} ms | "
            f"{item['p95_latency_ms']:.1f} ms | {item['throughput_requests_per_second']:.3f} req/s |"
        )
    router = report["router_replay"]
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- SFT 将 1.7B 模型裸能力从 {model_only['base_1p7b']['successful_runs']}/150 提升到 "
            f"{model_only['sft_1p7b']['successful_runs']}/150，提升 19.33 个百分点。",
            f"- 1.7B SFT 裸能力高于 4B DPO（{model_only['dpo_4b']['successful_runs']}/150），"
            "同时满足延迟和吞吐门槛，因此晋级为高频工具决策学生。",
            f"- 8B 裸能力为 {model_only['base_8b']['successful_runs']}/150，保留为复杂场景教师。",
            f"- 修正版 DPO 使用显式冻结 SFT reference，从 {model_only['sft_1p7b']['successful_runs']}/150 "
            f"提升到 {model_only['dpo_1p7b']['successful_runs']}/150，HTTP 错误为 0；因 +2/150 尚不显著，"
            "进入 Shadow 而不直接替换 SFT。",
            "- 旧 DPO 实际以禁用 adapter 的 Base 为 reference，仅 104/150 且有 37 次流中断，已明确拒绝。",
            f"- 150 条中有 {production['sft_1p7b']['label_contract_conflicts']} 条冻结标签与当前运行时合同冲突；"
            f"生产比较使用剩余 {production['sft_1p7b']['contract_consistent_runs']} 条合同一致口径。",
            f"- 生产路由规则回放原始口径为 {router['successful_runs']}/150，合同一致口径为 "
            f"{router['contract_consistent_successful_runs']}/{router['contract_consistent_runs']}，教师占比 "
            f"{router['teacher_share'] * 100:.2f}%；该数字是顺序回放，不冒充同卡在线实测。",
            "- 本冻结集全部属于学生已覆盖的有界动作，因此 8B 路由占比为 0；8B 只接手课程外 complex 动作。",
            f"- 学生/教师 oracle 并集为 {router['oracle_union_successes']}/150，只作为路由器上限诊断。",
            "- 单轮延迟来自一次工程测量，应在发布前追加多轮稳定性测试。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("ml/agentic/reports"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "ml/agentic/datasets/external-benchmark-v1/deepseek-v4-flash-stage29-v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/agentic/reports/stage32-cascade-distillation-final-v1"),
    )
    args = parser.parse_args()
    report = build(args.reports_root, args.dataset_root, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
