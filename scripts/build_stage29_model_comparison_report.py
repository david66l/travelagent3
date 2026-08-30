"""Build the paired Stage29 model comparison and policy-visible router replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ARMS = (
    ("base_4b", "Qwen3-4B Base", "stage29-deepseek-base4b-v1"),
    ("stage21_sft_4b", "Qwen3-4B Stage21 SFT", "stage29-deepseek-stage21-sft4b-v1"),
    ("stage22_dpo_4b", "Qwen3-4B Stage22 SFT+DPO", "stage29-deepseek-stage22-dpo4b-v1"),
    ("stage28_sft_4b", "Qwen3-4B Stage28 SFT V2", "stage29-deepseek-stage28-sft4b-v2"),
    ("stage28_dpo_4b", "Qwen3-4B Stage28 SFT+DPO V2", "stage29-deepseek-stage28-dpo4b-v2"),
    ("teacher_8b", "Qwen3-8B", "stage29-deepseek-teacher8b-v1"),
)


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
    if not values:
        raise ValueError("bootstrap requires non-empty values")
    rng = random.Random(seed)
    count = len(values)
    rates = [
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    return [round(_percentile(rates, 0.025), 8), round(_percentile(rates, 0.975), 8)]


def bootstrap_paired_difference_ci(
    candidate: list[bool],
    baseline: list[bool],
    *,
    samples: int = 10_000,
    seed: int = 20260815,
) -> list[float]:
    if len(candidate) != len(baseline) or not candidate:
        raise ValueError("paired bootstrap requires equal non-empty inputs")
    rng = random.Random(seed)
    count = len(candidate)
    differences = []
    for _ in range(samples):
        indices = [rng.randrange(count) for _ in range(count)]
        differences.append(
            sum(candidate[index] - baseline[index] for index in indices) / count
        )
    return [
        round(_percentile(differences, 0.025), 8),
        round(_percentile(differences, 0.975), 8),
    ]


def exact_mcnemar_p(candidate: list[bool], baseline: list[bool]) -> dict[str, Any]:
    improved = sum(c and not b for c, b in zip(candidate, baseline, strict=True))
    regressed = sum(b and not c for c, b in zip(candidate, baseline, strict=True))
    discordant = improved + regressed
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index) * (0.5**discordant)
            for index in range(0, min(improved, regressed) + 1)
        )
        p_value = min(1.0, 2 * tail)
    return {
        "candidate_only_success": improved,
        "baseline_only_success": regressed,
        "discordant_pairs": discordant,
        "exact_two_sided_p": round(p_value, 8),
    }


def route_to_teacher(context: dict[str, Any]) -> bool:
    """Predeclared router using only policy-visible state, never gold labels."""
    capability = context.get("capability") or {}
    status = capability.get("status")
    if status in {"infeasible", "unsafe"}:
        return True
    if status == "missing_tool":
        failures = context.get("failure_summary") or []
        return any(
            not failure.get("retryable", False)
            or int(failure.get("retry_budget_remaining", 0)) <= 0
            for failure in failures
        )
    return False


def _load_case_metadata(dataset_root: Path) -> dict[str, dict[str, Any]]:
    metadata = {}
    for split_file, split in (("dev.jsonl", "dev"), ("sealed_test.jsonl", "sealed_test")):
        for case in _read_jsonl(dataset_root / split_file):
            context = json.loads(case["messages"][1]["content"])
            metadata[case["case_id"]] = {
                "split": split,
                "source": case["source"],
                "difficulty": case["difficulty"],
                "context": context,
            }
    return metadata


def _summarize_arm(directory: Path, metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    report_path = directory / "report.json"
    runs_path = directory / "runs.jsonl"
    report = _read_json(report_path)
    runs = _read_jsonl(runs_path)
    if report.get("schema_version") != "vllm-http-benchmark.v1":
        raise ValueError(f"unexpected report schema: {report_path}")
    if len(runs) != 150 or len({row["case_id"] for row in runs}) != 150:
        raise ValueError(f"Stage29 arm is not a paired 150-case run: {directory}")
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_action[row["expected_action"]].append(row)
        case_meta = metadata[row["case_id"]]
        by_split[case_meta["split"]].append(row)
        by_source[case_meta["source"]].append(row)
    successes = [bool(row["success"]) for row in runs]
    non_abort = [row for row in runs if row["expected_action"] != "abort"]
    false_aborts = sum("abort" in row["observed_actions"] for row in non_abort)
    inference = report["summary"]["inference"]
    return {
        "model": report["model"],
        "runs": len(runs),
        "successful_runs": sum(successes),
        "success_rate": round(sum(successes) / len(runs), 8),
        "bootstrap_95_ci": bootstrap_rate_ci(successes),
        "action_mismatches": sum(bool(row["action_mismatch"]) for row in runs),
        "http_errors": sum(bool(row["http_error"]) for row in runs),
        "multiple_action_runs": sum(len(row["observed_actions"]) > 1 for row in runs),
        "abort_recall": round(
            sum(bool(row["success"]) for row in by_action["abort"])
            / len(by_action["abort"]),
            8,
        ),
        "false_abort_rate": round(false_aborts / len(non_abort), 8),
        "false_abort_count": false_aborts,
        "by_action": {
            action: {
                "successful": sum(bool(row["success"]) for row in items),
                "runs": len(items),
                "success_rate": round(
                    sum(bool(row["success"]) for row in items) / len(items), 8
                ),
            }
            for action, items in sorted(by_action.items())
        },
        "by_split": {
            split: {
                "successful": sum(bool(row["success"]) for row in items),
                "runs": len(items),
            }
            for split, items in sorted(by_split.items())
        },
        "by_source": {
            source: {
                "successful": sum(bool(row["success"]) for row in items),
                "runs": len(items),
            }
            for source, items in sorted(by_source.items())
        },
        "failure_confusion": dict(
            Counter(
                f"{row['expected_action']}=>{'+'.join(row['observed_actions']) or 'none'}"
                for row in runs
                if not row["success"]
            )
        ),
        "failure_case_ids": sorted(row["case_id"] for row in runs if not row["success"]),
        "latency_ms": inference["request_latency_ms"],
        "completion_tokens_mean": inference["completion_tokens"]["mean"],
        "throughput_requests_per_second": report["summary"]["request_throughput_per_second"],
        "source": {
            "report": report_path.as_posix(),
            "report_sha256": _sha256(report_path),
            "runs": runs_path.as_posix(),
            "runs_sha256": _sha256(runs_path),
        },
        "_runs": runs,
    }


def build(reports_root: Path, dataset_root: Path, audit_root: Path) -> dict[str, Any]:
    metadata = _load_case_metadata(dataset_root)
    if len(metadata) != 150:
        raise ValueError("Stage29 dataset must contain exactly 150 cases")
    arms = {}
    for key, label, directory in ARMS:
        summary = _summarize_arm(reports_root / directory, metadata)
        summary["label"] = label
        arms[key] = summary
    expected_case_ids = set(metadata)
    for arm in arms.values():
        if {row["case_id"] for row in arm["_runs"]} != expected_case_ids:
            raise ValueError("model arms do not use the same frozen case IDs")

    base_runs = {row["case_id"]: row for row in arms["base_4b"]["_runs"]}
    comparisons = {}
    for key, arm in arms.items():
        if key == "base_4b":
            continue
        runs = {row["case_id"]: row for row in arm["_runs"]}
        ordered_ids = sorted(expected_case_ids)
        candidate = [bool(runs[case_id]["success"]) for case_id in ordered_ids]
        baseline = [bool(base_runs[case_id]["success"]) for case_id in ordered_ids]
        comparisons[f"{key}_vs_base"] = {
            "percentage_point_difference": round(
                (sum(candidate) - sum(baseline)) / len(candidate) * 100, 3
            ),
            "paired_bootstrap_95_ci_percentage_points": [
                round(value * 100, 3)
                for value in bootstrap_paired_difference_ci(candidate, baseline)
            ],
            "mcnemar": exact_mcnemar_p(candidate, baseline),
        }

    ordered_ids = sorted(expected_case_ids)
    stage28_sft_by_id = {
        row["case_id"]: row for row in arms["stage28_sft_4b"]["_runs"]
    }
    stage28_dpo_by_id = {
        row["case_id"]: row for row in arms["stage28_dpo_4b"]["_runs"]
    }
    stage28_dpo_values = [
        bool(stage28_dpo_by_id[case_id]["success"]) for case_id in ordered_ids
    ]
    stage28_sft_values = [
        bool(stage28_sft_by_id[case_id]["success"]) for case_id in ordered_ids
    ]
    stage28_direct = {
        "percentage_point_difference": round(
            (sum(stage28_dpo_values) - sum(stage28_sft_values))
            / len(ordered_ids)
            * 100,
            3,
        ),
        "paired_bootstrap_95_ci_percentage_points": [
            round(value * 100, 3)
            for value in bootstrap_paired_difference_ci(
                stage28_dpo_values, stage28_sft_values
            )
        ],
        "mcnemar": exact_mcnemar_p(stage28_dpo_values, stage28_sft_values),
    }

    student_runs = {
        row["case_id"]: row for row in arms["stage28_dpo_4b"]["_runs"]
    }
    teacher_runs = {row["case_id"]: row for row in arms["teacher_8b"]["_runs"]}
    routed = []
    for case_id in sorted(expected_case_ids):
        teacher = route_to_teacher(metadata[case_id]["context"])
        selected = teacher_runs[case_id] if teacher else student_runs[case_id]
        routed.append(
            {
                "case_id": case_id,
                "route": "teacher_8b" if teacher else "student_stage28_dpo_4b",
                "success": bool(selected["success"]),
            }
        )
    routed_successes = [row["success"] for row in routed]
    teacher_calls = sum(row["route"] == "teacher_8b" for row in routed)

    manifest = _read_json(dataset_root / "manifest.json")
    audit = _read_json(audit_root / "report.json")
    if manifest["content_sha256"] != audit["content_sha256"] or not audit["passed"]:
        raise ValueError("frozen dataset hash or audit gate mismatch")

    public_arms = {}
    for key, arm in arms.items():
        public_arms[key] = {name: value for name, value in arm.items() if name != "_runs"}
    best_4b_key = max(
        (key for key in public_arms if key != "teacher_8b"),
        key=lambda key: public_arms[key]["success_rate"],
    )
    return {
        "schema_version": "travel-agent-stage29-model-comparison.v1",
        "status": "complete_external_model_evaluation",
        "benchmark": {
            "cases": 150,
            "dev": 30,
            "sealed_test": 120,
            "authorship": "DeepSeek V4 Flash synthetic",
            "double_annotation": "two isolated DeepSeek V4 Flash passes",
            "adjudicator": "Codex",
            "content_sha256": manifest["content_sha256"],
            "contamination_audit_passed": audit["training_contamination"]["passed"],
            "independent_human_benchmark": False,
        },
        "inference_protocol": {
            "temperature": 0,
            "repetitions": 1,
            "concurrency": 8,
            "max_tokens": 192,
            "confidence_interval": "10,000-case bootstrap; reflects case sampling, not decoding variance",
        },
        "arms": public_arms,
        "paired_vs_base": comparisons,
        "stage28_dpo_vs_sft": stage28_direct,
        "best_4b": {
            "key": best_4b_key,
            "label": public_arms[best_4b_key]["label"],
            "successful_runs": public_arms[best_4b_key]["successful_runs"],
            "success_rate": public_arms[best_4b_key]["success_rate"],
        },
        "policy_visible_router_replay": {
            "policy": (
                "8B for infeasible/unsafe or exhausted non-retryable tool state; "
                "Stage28 DPO 4B otherwise"
            ),
            "uses_gold_label": False,
            "successful_cases": sum(routed_successes),
            "cases": len(routed),
            "success_rate": round(sum(routed_successes) / len(routed), 8),
            "bootstrap_95_ci": bootstrap_rate_ci(routed_successes),
            "teacher_cases": teacher_calls,
            "teacher_share": round(teacher_calls / len(routed), 8),
            "failure_case_ids": [row["case_id"] for row in routed if not row["success"]],
        },
        "decision": {
            "default_4b_candidate": "stage28_dpo_4b",
            "complex_state_model": "teacher_8b",
            "reason": (
                "Stage28 DPO V2 is the strongest 4B arm on the frozen set, while 8B retains "
                "a substantial quality lead on policy-visible complex states."
            ),
            "caution": (
                "The Stage28 DPO gain over Base must be read with its paired confidence interval; "
                "one deterministic rollout does not measure decoding stability."
            ),
        },
        "limitations": [
            "Synthetic external-model benchmark; not an independent human benchmark.",
            "Both annotation passes use the same model family.",
            "One deterministic rollout per case; bootstrap intervals cover case sampling only.",
            "Latency is diagnostic because Base/LoRA/8B serving configurations differ.",
            "Scoring covers tool-action selection, not end-to-end itinerary quality.",
        ],
        "source": {
            "manifest": (dataset_root / "manifest.json").as_posix(),
            "manifest_sha256": _sha256(dataset_root / "manifest.json"),
            "audit": (audit_root / "report.json").as_posix(),
            "audit_sha256": _sha256(audit_root / "report.json"),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TravelAgent Stage29：DeepSeek 外部模型评测",
        "",
        "> 150 条 DeepSeek V4 Flash 合成题，双路盲标，Codex 裁决；不是独立人工 benchmark。",
        "",
        "## 总体结果",
        "",
        "| 模型 | 正确/总数 | 正确率 | Bootstrap 95% CI | Abort Recall | False Abort | 平均延迟 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, _, _ in ARMS:
        arm = report["arms"][key]
        ci = arm["bootstrap_95_ci"]
        lines.append(
            f"| {arm['label']} | {arm['successful_runs']}/{arm['runs']} | "
            f"{arm['success_rate'] * 100:.2f}% | {ci[0] * 100:.2f}%–{ci[1] * 100:.2f}% | "
            f"{arm['abort_recall'] * 100:.2f}% | {arm['false_abort_rate'] * 100:.2f}% | "
            f"{arm['latency_ms']['mean']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            "## 分动作正确率",
            "",
            "| 模型 | Search | Ask user | Tradeoff | Abort |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key, _, _ in ARMS:
        arm = report["arms"][key]
        cells = []
        for action in ("search_pois", "ask_user", "propose_tradeoff", "abort"):
            item = arm["by_action"][action]
            cells.append(f"{item['successful']}/{item['runs']}")
        lines.append(f"| {arm['label']} | " + " | ".join(cells) + " |")
    router = report["policy_visible_router_replay"]
    best = report["best_4b"]
    stage28_direct = report["stage28_dpo_vs_sft"]
    lines.extend(
        [
            "",
            "## 决策",
            "",
            f"- 最佳 4B：{best['label']}，{best['successful_runs']}/150（{best['success_rate'] * 100:.2f}%）。",
            f"- 8B：{report['arms']['teacher_8b']['successful_runs']}/150（{report['arms']['teacher_8b']['success_rate'] * 100:.2f}%）。",
            f"- Stage28 DPO 相对 Stage28 SFT：{stage28_direct['percentage_point_difference']:+.2f} 个百分点；"
            f"配对 95% CI 为 {stage28_direct['paired_bootstrap_95_ci_percentage_points'][0]:+.2f} 到 "
            f"{stage28_direct['paired_bootstrap_95_ci_percentage_points'][1]:+.2f}，尚不能视为稳定显著优势。",
            f"- 可见状态路由：{router['successful_cases']}/150（{router['success_rate'] * 100:.2f}%），"
            f"8B 调用 {router['teacher_cases']} 条（{router['teacher_share'] * 100:.2f}%）。",
            "- 推荐将 Stage28 DPO V2 作为新的 4B 候选，复杂/不可行/安全终止状态继续路由到 8B。",
            "",
            "## 边界",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=Path("ml/agentic/reports"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "ml/agentic/datasets/external-benchmark-v1/deepseek-v4-flash-stage29-v1"
        ),
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path("ml/agentic/reports/stage29-deepseek-external-benchmark-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/agentic/reports/stage29-deepseek-model-comparison-v1"),
    )
    args = parser.parse_args()
    report = build(args.reports_root, args.dataset_root, args.audit_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "best_4b": report["best_4b"],
                "teacher_8b": report["arms"]["teacher_8b"]["success_rate"],
                "router": report["policy_visible_router_replay"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
