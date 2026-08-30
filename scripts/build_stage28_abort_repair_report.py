"""Build the Stage28 abort-boundary comparison from frozen HTTP run records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VARIANTS = (
    ("base_4b", "Qwen3-4B Base", "stage27-ai-pilot-base-smoke-v1"),
    ("old_sft_4b", "Stage21 SFT", "stage27-ai-pilot-sft-smoke-v1"),
    ("old_dpo_4b", "Stage22 SFT+DPO", "stage27-ai-pilot-dpo-smoke-v2"),
    ("repair_sft_v1", "Stage28 SFT V1", "stage28-ai-pilot-sft-abort-calibrated-v1"),
    ("repair_sft_v2", "Stage28 SFT V2", "stage28-ai-pilot-sft-abort-diverse-v2"),
    ("repair_dpo_v2", "Stage28 SFT+DPO V2", "stage28-ai-pilot-dpo-abort-diverse-v2"),
    ("teacher_8b", "Qwen3-8B", "stage27-ai-pilot-8b-smoke-v1"),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def build(reports_root: Path, output_dir: Path) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    case_sets: list[set[str]] = []
    for key, label, directory in VARIANTS:
        root = reports_root / directory
        report = json.loads((root / "report.json").read_text(encoding="utf-8"))
        runs = _read_jsonl(root / "runs.jsonl")
        case_sets.append({str(row["case_id"]) for row in runs})
        failures = [row for row in runs if not row["success"]]
        observed = Counter(
            (row.get("observed_actions") or ["none"])[0] for row in runs
        )
        summary = report["summary"]
        variants[key] = {
            "label": label,
            "report_dir": directory,
            "successful_runs": summary["successful_runs"],
            "total_runs": summary["runs"],
            "accuracy": round(summary["successful_runs"] / summary["runs"], 6),
            "action_mismatches": summary["action_mismatches"],
            "argument_mismatches": summary["argument_mismatches"],
            "observed_action_counts": dict(observed),
            "failure_cases": [
                {
                    "case_id": row["case_id"],
                    "expected_action": row["expected_action"],
                    "observed_actions": row.get("observed_actions") or [],
                }
                for row in failures
            ],
            "latency_ms_mean": summary["inference"]["request_latency_ms"]["mean"],
        }
    if any(items != case_sets[0] for items in case_sets[1:]):
        raise ValueError("Stage28 comparison variants do not share the same case set")

    result = {
        "schema_version": "travel-agent-stage28-abort-repair-comparison.v1",
        "benchmark_scope": "AI-assisted Pilot; not an independent external benchmark",
        "cases": len(case_sets[0]),
        "variants": variants,
        "decision": {
            "promoted_4b_candidate": "repair_sft_v2",
            "reason": (
                "28/30 versus old SFT 26/30 and old DPO 25/30; all 26 non-abort "
                "cases remain correct; DPO V2 adds no Pilot accuracy"
            ),
            "remaining_failures": ["ext-v1-ai-pilot-017", "ext-v1-ai-pilot-018"],
            "production_route": "route complex tradeoff decisions to Qwen3-8B",
            "routed_pilot_accuracy": 1.0,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage 28 必要终止边界修复对照",
        "",
        "> 该结果来自 AI-assisted Pilot，不是独立外部 benchmark。",
        "",
        "| 模型 | 正确率 | 错误数 | 平均请求延迟 |",
        "|---|---:|---:|---:|",
    ]
    for key, _, _ in VARIANTS:
        item = variants[key]
        lines.append(
            f"| {item['label']} | {item['successful_runs']}/{item['total_runs']} "
            f"({item['accuracy']:.2%}) | {item['action_mismatches']} | "
            f"{item['latency_ms_mean']:.1f} ms |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "Stage28 SFT V2 为当前 4B 候选：28/30，且所有非 abort 样本保持正确。",
            "DPO V2 同为 28/30，没有获得外部增益，因此保留为后训练证据但不晋升。",
            "剩余两题是固定闭馆和固定时序硬冲突；不继续使用同一 Pilot 调参，复杂取舍仍路由 Qwen3-8B。",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("ml/agentic/reports"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/agentic/reports/stage28-abort-repair-comparison-v1"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.reports_root, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
