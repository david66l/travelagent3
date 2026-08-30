"""Build the paired AI-assisted Pilot model comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ARMS = (
    ("base_4b", "Qwen3-4B Base", "stage27-ai-pilot-base-smoke-v1", "static/eager"),
    ("sft_4b", "Qwen3-4B SFT", "stage27-ai-pilot-sft-smoke-v1", "dynamic LoRA/eager"),
    ("dpo_4b", "Qwen3-4B SFT+DPO", "stage27-ai-pilot-dpo-smoke-v2", "static/eager"),
    ("teacher_8b", "Qwen3-8B", "stage27-ai-pilot-8b-smoke-v1", "static/eager"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_arm(directory: Path, *, label: str, runtime: str) -> dict[str, Any]:
    report_path = directory / "report.json"
    runs_path = directory / "runs.jsonl"
    report = read_json(report_path)
    runs = read_jsonl(runs_path)
    if report.get("schema_version") != "vllm-http-benchmark.v1":
        raise ValueError(f"unexpected report schema: {report_path}")
    if len(runs) != report["summary"]["runs"]:
        raise ValueError(f"run count mismatch: {directory}")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_family[row["family"]].append(row)
    failures = [row for row in runs if not row["success"]]
    inference = report["summary"]["inference"]
    return {
        "label": label,
        "model": report["model"],
        "runtime": runtime,
        "runs": len(runs),
        "successful_runs": sum(bool(row["success"]) for row in runs),
        "success_rate": round(sum(bool(row["success"]) for row in runs) / len(runs), 8),
        "action_mismatches": sum(bool(row["action_mismatch"]) for row in runs),
        "http_errors": sum(bool(row["http_error"]) for row in runs),
        "multiple_action_runs": sum(len(row["observed_actions"]) > 1 for row in runs),
        "family_success": {
            family: {
                "successful": sum(bool(row["success"]) for row in items),
                "runs": len(items),
            }
            for family, items in sorted(by_family.items())
        },
        "failure_case_ids": sorted(row["case_id"] for row in failures),
        "failure_confusion": dict(
            Counter(
                f"{row['expected_action']}=>{'+'.join(row['observed_actions']) or 'none'}"
                for row in failures
            )
        ),
        "completion_tokens_mean": inference["completion_tokens"]["mean"],
        "latency_ms": inference["request_latency_ms"],
        "throughput_requests_per_second": report["summary"][
            "request_throughput_per_second"
        ],
        "source": {
            "report": report_path.as_posix(),
            "report_sha256": sha256_file(report_path),
            "runs": runs_path.as_posix(),
            "runs_sha256": sha256_file(runs_path),
        },
    }


def build(reports_root: Path, dataset_root: Path) -> dict[str, Any]:
    arms = {
        key: summarize_arm(reports_root / directory, label=label, runtime=runtime)
        for key, label, directory, runtime in ARMS
    }
    case_sets = {key: set(value["failure_case_ids"]) for key, value in arms.items()}
    expected_ids = set()
    for key in arms:
        directory = reports_root / next(item[2] for item in ARMS if item[0] == key)
        expected_ids.update(row["case_id"] for row in read_jsonl(directory / "runs.jsonl"))
    if len(expected_ids) != 30:
        raise ValueError("pilot comparison must contain exactly 30 paired cases")

    dpo_runs = read_jsonl(reports_root / "stage27-ai-pilot-dpo-smoke-v2" / "runs.jsonl")
    teacher_runs = {
        row["case_id"]: row
        for row in read_jsonl(reports_root / "stage27-ai-pilot-8b-smoke-v1" / "runs.jsonl")
    }
    routed = []
    for row in dpo_runs:
        selected = teacher_runs[row["case_id"]] if row["family"] == "tradeoff" else row
        routed.append(
            {
                "case_id": row["case_id"],
                "family": row["family"],
                "route": "teacher" if row["family"] == "tradeoff" else "student",
                "success": selected["success"],
            }
        )
    routed_success = sum(row["success"] for row in routed)
    teacher_calls = sum(row["route"] == "teacher" for row in routed)

    manifest_path = dataset_root / "manifest.json"
    audit_path = dataset_root / "audit.json"
    manifest = read_json(manifest_path)
    audit = read_json(audit_path)
    if manifest.get("eligible_for_external_claim") is not False:
        raise ValueError("AI Pilot claim boundary is missing")
    if audit.get("status") != "passed_for_schema_calibration":
        raise ValueError("AI Pilot audit has not passed")
    base = arms["base_4b"]
    sft = arms["sft_4b"]
    dpo = arms["dpo_4b"]
    teacher = arms["teacher_8b"]
    return {
        "schema_version": "travel-agent-stage27-ai-pilot-comparison.v1",
        "status": "passed_for_schema_calibration",
        "eligible_for_external_claim": False,
        "pilot": {
            "cases": 30,
            "authorship": "AI-assisted synthetic",
            "independent_human_authors": 0,
            "double_annotation": False,
            "content_sha256": manifest["content_sha256"],
            "contamination_audit_passed": audit["training_contamination"]["passed"],
        },
        "arms": arms,
        "paired_quality": {
            "base_to_sft_percentage_points": round(
                (sft["success_rate"] - base["success_rate"]) * 100, 3
            ),
            "sft_to_dpo_percentage_points": round(
                (dpo["success_rate"] - sft["success_rate"]) * 100, 3
            ),
            "base_to_dpo_percentage_points": round(
                (dpo["success_rate"] - base["success_rate"]) * 100, 3
            ),
            "dpo_to_teacher_percentage_points": round(
                (teacher["success_rate"] - dpo["success_rate"]) * 100, 3
            ),
            "base_failures": sorted(case_sets["base_4b"]),
            "sft_failures": sorted(case_sets["sft_4b"]),
            "dpo_failures": sorted(case_sets["dpo_4b"]),
            "teacher_failures": sorted(case_sets["teacher_8b"]),
        },
        "deterministic_router_replay": {
            "policy": "DPO 4B for non-tradeoff; 8B for tradeoff",
            "execution_mode": "paired sequential replay; not simultaneous endpoints",
            "successful_cases": routed_success,
            "cases": len(routed),
            "success_rate": round(routed_success / len(routed), 8),
            "teacher_cases": teacher_calls,
            "teacher_share": round(teacher_calls / len(routed), 8),
            "distribution_note": (
                "The Pilot intentionally contains 10/30 tradeoff cases; teacher share is not "
                "an estimate of production traffic."
            ),
        },
        "diagnostic_input_ablation": {
            "metadata_only_dpo_success": "15/30",
            "model_visible_policy_state_dpo_success": "25/30",
            "conclusion": (
                "Hard constraints and frozen facts must be included in model-visible policy "
                "context. The metadata-only run is excluded from model ranking."
            ),
        },
        "conclusions": [
            "All four models passed search and long-context action selection after policy-state alignment.",
            "The 4B failures concentrate on abort versus propose_tradeoff boundaries.",
            "On this Pilot, Base 4B outperformed SFT and DPO; the DPO checkpoint is not promoted on this evidence.",
            "Qwen3-8B passed 30/30 and repaired every DPO failure under paired sequential replay.",
            "The result supports routing difficult tradeoff cases to 8B and adding verified abort preferences before retraining 4B.",
        ],
        "limitations": [
            "AI-assisted synthetic Dev Pilot; not an independent external benchmark.",
            "One rollout per case; no confidence interval or stochastic stability claim.",
            "Runtime modes differ for dynamic SFT LoRA versus static Base/DPO/8B, so latency is diagnostic only.",
            "The router row is sequential paired replay, not simultaneous dual-endpoint serving.",
            "Argument quality is not yet scored beyond action selection.",
        ],
        "source": {
            "manifest": manifest_path.as_posix(),
            "manifest_sha256": sha256_file(manifest_path),
            "audit": audit_path.as_posix(),
            "audit_sha256": sha256_file(audit_path),
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TravelAgent 阶段 27：AI 辅助 Pilot 模型对照",
        "",
        "> 这是 30 条合成 Dev Pilot 的 schema/evaluator 校准结果，不是独立外部 Benchmark。",
        "",
        "## 同协议动作选择",
        "",
        "| 模型 | 运行时 | 正确/总数 | 成功率 | 平均延迟 | P95 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, *_ in ARMS:
        arm = report["arms"][key]
        lines.append(
            f"| {arm['label']} | {arm['runtime']} | {arm['successful_runs']}/{arm['runs']} | "
            f"{arm['success_rate'] * 100:.2f}% | {arm['latency_ms']['mean']:.1f} ms | "
            f"{arm['latency_ms']['p95']:.1f} ms |"
        )
    lines.extend(["", "## 分任务族", "", "| 模型 | 澄清 | 搜索 | 恢复 | 取舍 | 长上下文 |", "|---|---:|---:|---:|---:|---:|"])
    order = ("clarification", "search", "recovery", "tradeoff", "long_context_replan")
    for key, *_ in ARMS:
        arm = report["arms"][key]
        cells = [
            f"{arm['family_success'][family]['successful']}/{arm['family_success'][family]['runs']}"
            for family in order
        ]
        lines.append(f"| {arm['label']} | " + " | ".join(cells) + " |")
    router = report["deterministic_router_replay"]
    lines.extend(
        [
            "",
            "## Router 配对回放",
            "",
            f"4B DPO 处理非取舍题、8B 处理取舍题：{router['successful_cases']}/{router['cases']}；"
            f"教师占比 {router['teacher_share'] * 100:.2f}%（Pilot 人为高比例取舍题，不能外推生产流量）。",
            "",
            "## 关键结论",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in report["conclusions"])
    lines.extend(["", "## 边界", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=Path("ml/agentic/reports"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("ml/agentic/datasets/external-benchmark-v1/ai-assisted-pilot-v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/agentic/reports/stage27-ai-pilot-comparison-v1"),
    )
    args = parser.parse_args()
    report = build(args.reports_root, args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "router": report["deterministic_router_replay"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
