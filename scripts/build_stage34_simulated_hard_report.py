"""Build the paired Stage34 simulated hard-user model comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ARM_LABELS = {
    "base": "Qwen3-1.7B Base",
    "sft": "Qwen3-1.7B SFT",
    "dpo": "Qwen3-1.7B SFT+DPO",
    "teacher": "Qwen3-8B Base",
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _summarize(rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    by_action: dict[str, dict[str, int | float]] = {}
    for action in sorted({str(row["expected_action"]) for row in rows}):
        selected = [row for row in rows if row["expected_action"] == action]
        raw = sum(bool(row["success"]) for row in selected)
        contract = sum(bool(row["policy_contract_success"]) for row in selected)
        by_action[action] = {
            "runs": len(selected),
            "raw_successful": raw,
            "raw_success_rate": round(raw / len(selected), 8),
            "contract_successful": contract,
            "contract_success_rate": round(contract / len(selected), 8),
        }
    summary = report["summary"]
    inference = summary["inference"]["request_latency_ms"]
    return {
        "model": report["model"],
        "runs": len(rows),
        "raw_successful": sum(bool(row["success"]) for row in rows),
        "contract_successful": sum(
            bool(row["policy_contract_success"]) for row in rows
        ),
        "http_errors": sum(bool(row.get("http_error")) for row in rows),
        "multiple_action_runs": sum(len(row["observed_actions"]) != 1 for row in rows),
        "label_contract_conflicts": sum(
            bool(row["label_contract_conflict"]) for row in rows
        ),
        "latency_ms": {
            "mean": inference["mean"],
            "p50": inference["p50"],
            "p95": inference["p95"],
        },
        "by_action": by_action,
    }


def _paired(
    candidate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    *,
    success_key: str,
) -> dict[str, Any]:
    ids = sorted(baseline)
    candidate_only = sum(
        bool(candidate[item][success_key]) and not bool(baseline[item][success_key])
        for item in ids
    )
    baseline_only = sum(
        bool(baseline[item][success_key]) and not bool(candidate[item][success_key])
        for item in ids
    )
    return {
        "candidate_only_success": candidate_only,
        "baseline_only_success": baseline_only,
        "both_success": sum(
            bool(candidate[item][success_key]) and bool(baseline[item][success_key])
            for item in ids
        ),
        "both_failure": sum(
            not bool(candidate[item][success_key])
            and not bool(baseline[item][success_key])
            for item in ids
        ),
        "difference_percentage_points": round(
            (candidate_only - baseline_only) / len(ids) * 100, 4
        ),
    }


def _signature(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "actions": row["observed_actions"],
            "arguments": row.get("observed_arguments"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _policy_visible_router(
    base: dict[str, dict[str, Any]], sft: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Replay a route using only controller-visible allowed actions."""
    selected: list[tuple[str, dict[str, Any]]] = []
    for case_id in sorted(base):
        use_sft = "search_pois" in base[case_id]["allowed_actions"]
        selected.append(("sft", sft[case_id]) if use_sft else ("base", base[case_id]))
    latencies = [
        float(row["inference_metrics"]["request_latency_ms"]) for _, row in selected
    ]
    return {
        "policy": "SFT whenever search_pois is allowed; Base otherwise",
        "uses_gold_label": False,
        "runs": len(selected),
        "route_counts": dict(sorted(Counter(route for route, _ in selected).items())),
        "raw_successful": sum(bool(row["success"]) for _, row in selected),
        "contract_successful": sum(
            bool(row["policy_contract_success"]) for _, row in selected
        ),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 3),
            "p50": round(_percentile(latencies, 0.5), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
        },
    }


def build(arm_dirs: dict[str, Path]) -> dict[str, Any]:
    reports = {key: _read_json(path / "report.json") for key, path in arm_dirs.items()}
    rows = {key: _read_jsonl(path / "runs.jsonl") for key, path in arm_dirs.items()}
    ids = {key: {str(row["case_id"]) for row in values} for key, values in rows.items()}
    if any(len(values) != 150 for values in rows.values()):
        raise ValueError("Stage34 requires exactly 150 runs per arm")
    if len({frozenset(value) for value in ids.values()}) != 1:
        raise ValueError("Stage34 arms must contain the same case IDs")
    by_id = {
        key: {str(row["case_id"]): row for row in values}
        for key, values in rows.items()
    }
    comparisons = {}
    for candidate in ("sft", "dpo", "teacher"):
        comparisons[f"{candidate}_vs_base"] = {
            "raw": _paired(by_id[candidate], by_id["base"], success_key="success"),
            "production_contract": _paired(
                by_id[candidate],
                by_id["base"],
                success_key="policy_contract_success",
            ),
            "decision_signature_divergences": sum(
                _signature(by_id[candidate][case_id])
                != _signature(by_id["base"][case_id])
                for case_id in sorted(by_id["base"])
            ),
            "action_sequence_divergences": sum(
                by_id[candidate][case_id]["observed_actions"]
                != by_id["base"][case_id]["observed_actions"]
                for case_id in sorted(by_id["base"])
            ),
            "argument_divergences": sum(
                by_id[candidate][case_id].get("observed_arguments")
                != by_id["base"][case_id].get("observed_arguments")
                for case_id in sorted(by_id["base"])
            ),
        }
    comparisons["dpo_vs_sft"] = {
        "raw": _paired(by_id["dpo"], by_id["sft"], success_key="success"),
        "production_contract": _paired(
            by_id["dpo"], by_id["sft"], success_key="policy_contract_success"
        ),
        "decision_signature_divergences": sum(
            _signature(by_id["dpo"][case_id]) != _signature(by_id["sft"][case_id])
            for case_id in sorted(by_id["sft"])
        ),
        "action_sequence_divergences": sum(
            by_id["dpo"][case_id]["observed_actions"]
            != by_id["sft"][case_id]["observed_actions"]
            for case_id in sorted(by_id["sft"])
        ),
        "argument_divergences": sum(
            by_id["dpo"][case_id].get("observed_arguments")
            != by_id["sft"][case_id].get("observed_arguments")
            for case_id in sorted(by_id["sft"])
        ),
    }
    failures = {
        key: [
            {
                "case_id": row["case_id"],
                "expected_action": row["expected_action"],
                "observed_actions": row["observed_actions"],
                "label_contract_conflict": row["label_contract_conflict"],
            }
            for row in values
            if not row["success"]
        ]
        for key, values in rows.items()
    }
    return {
        "schema_version": "stage34-simulated-hard-model-comparison.v1",
        "evidence_class": "synthetic_external_model_benchmark",
        "cases": 150,
        "arms": {
            key: {"label": ARM_LABELS[key], **_summarize(rows[key], reports[key])}
            for key in ARM_LABELS
        },
        "paired_comparisons": comparisons,
        "policy_visible_router_replay": _policy_visible_router(
            by_id["base"], by_id["sft"]
        ),
        "failures": failures,
        "sources": {
            key: {
                "report_sha256": _sha256(path / "report.json"),
                "runs_sha256": _sha256(path / "runs.jsonl"),
            }
            for key, path in arm_dirs.items()
        },
        "decision": {
            "champion": "policy-visible Base/SFT router",
            "dpo_promotion": False,
            "teacher_default_route": False,
            "reason": (
                "SFT fixes one Base search error but regresses one abort output; DPO is "
                "action-identical to SFT with no correctness gain, while its 21 argument "
                "variations have no measured utility gain; 8B is slower and has no unique "
                "win over the Base/SFT policy-visible router."
            ),
        },
        "limitations": [
            "All 150 cases were authored and double-annotated by DeepSeek V4 Flash, not humans.",
            "One deterministic decoding run per model does not measure decoding variance.",
            "This benchmark measures policy action selection, not end-to-end itinerary quality.",
            "One frozen label conflicts with the current production action contract.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage34：150 条模拟难用户模型对比",
        "",
        "> DeepSeek V4 Flash 合成、双路标注并完成冲突裁决；不是人工用户 benchmark。",
        "",
        "## 总体结果",
        "",
        "| 模型 | 原标签正确 | 生产合同正确 | 多动作输出 | P50 | P95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ARM_LABELS:
        arm = report["arms"][key]
        lines.append(
            f"| {arm['label']} | {arm['raw_successful']}/150 | "
            f"{arm['contract_successful']}/150 | {arm['multiple_action_runs']} | "
            f"{arm['latency_ms']['p50']:.1f} ms | {arm['latency_ms']['p95']:.1f} ms |"
        )
    comparison = report["paired_comparisons"]
    router = report["policy_visible_router_replay"]
    lines.extend(
        [
            "",
            "## 配对结论",
            "",
            f"- SFT 相对 Base：独赢 {comparison['sft_vs_base']['raw']['candidate_only_success']}，"
            f"独输 {comparison['sft_vs_base']['raw']['baseline_only_success']}，净变化 "
            f"{comparison['sft_vs_base']['raw']['difference_percentage_points']:+.2f}pp。",
            f"- DPO 相对 SFT：动作序列分歧 "
            f"{comparison['dpo_vs_sft']['action_sequence_divergences']}/150，参数分歧 "
            f"{comparison['dpo_vs_sft']['argument_divergences']}/150，正确率净变化 "
            f"{comparison['dpo_vs_sft']['raw']['difference_percentage_points']:+.2f}pp；"
            "当前评测没有证明这些参数变化带来效用增益。",
            f"- 8B 相对 Base：独赢 {comparison['teacher_vs_base']['raw']['candidate_only_success']}，"
            f"独输 {comparison['teacher_vs_base']['raw']['baseline_only_success']}。",
            "",
            "## 可部署路由回放",
            "",
            f"规则：`{router['policy']}`。该规则只读取控制器允许动作，不读取金标。",
            "",
            f"- 原标签正确：{router['raw_successful']}/150；",
            f"- 当前生产合同正确：{router['contract_successful']}/150；",
            f"- 路由数量：Base {router['route_counts'].get('base', 0)}，SFT "
            f"{router['route_counts'].get('sft', 0)}；",
            f"- 回放 P95：{router['latency_ms']['p95']:.1f} ms。",
            "",
            "## 决策",
            "",
            "- 当前不晋级 DPO：动作和正确性均未改变；21 条参数文本变化尚无效用增益证据。",
            "- 当前不把 8B 设为默认工具决策教师：正确数更少、延迟更高。",
            "- 推荐 Base/SFT 动作族路由：普通澄清、权衡、终止用 Base，搜索用 SFT。",
            "- 下一轮后训练应针对 SFT 的重复工具调用和 DPO 同质化，而不是继续堆同分布样本。",
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
    for key in ARM_LABELS:
        parser.add_argument(f"--{key}-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    arm_dirs = {key: getattr(args, f"{key}_dir") for key in ARM_LABELS}
    report = build(arm_dirs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "arms": {
                    key: {
                        "raw": value["raw_successful"],
                        "contract": value["contract_successful"],
                        "p95_ms": value["latency_ms"]["p95"],
                    }
                    for key, value in report["arms"].items()
                },
                "router": report["policy_visible_router_replay"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
