"""Aggregate Stage30 routed stability and concurrency evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


RUNS = {
    "student_deterministic": "stage30-student-stability-c8-t0-r5-v1",
    "teacher_deterministic": "stage30-teacher-stability-c8-t0-r5-v1",
    "student_stochastic": "stage30-student-stochastic-c8-t02-r3-v1",
    "teacher_stochastic": "stage30-teacher-stochastic-c8-t02-r3-v1",
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


def summarize_repetitions(runs: list[dict[str, Any]], expected_repetitions: int) -> dict[str, Any]:
    by_repetition: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_repetition[int(row["repetition"])].append(row)
        by_case[row["case_id"]].append(row)
    if sorted(by_repetition) != list(range(expected_repetitions)):
        raise ValueError("unexpected repetition IDs")
    if any(len(items) != expected_repetitions for items in by_case.values()):
        raise ValueError("not every routed case appears in every repetition")
    rows = []
    for repetition in sorted(by_repetition):
        items = by_repetition[repetition]
        non_abort = [item for item in items if item["expected_action"] != "abort"]
        false_abort = sum("abort" in item["observed_actions"] for item in non_abort)
        rows.append(
            {
                "repetition": repetition,
                "successful": sum(bool(item["success"]) for item in items),
                "runs": len(items),
                "success_rate": round(
                    sum(bool(item["success"]) for item in items) / len(items), 8
                ),
                "http_errors": sum(bool(item["http_error"]) for item in items),
                "false_abort_count": false_abort,
                "false_abort_rate": round(false_abort / len(non_abort), 8),
            }
        )
    volatile = []
    for case_id, items in by_case.items():
        signatures = {
            (tuple(item["observed_actions"]), bool(item["success"])) for item in items
        }
        if len(signatures) > 1:
            volatile.append(case_id)
    rates = [row["success_rate"] for row in rows]
    return {
        "repetitions": rows,
        "mean_success_rate": round(fmean(rates), 8),
        "minimum_success_rate": min(rates),
        "maximum_success_rate": max(rates),
        "spread_percentage_points": round((max(rates) - min(rates)) * 100, 3),
        "stable_action_cases": len(by_case) - len(volatile),
        "cases": len(by_case),
        "action_stability_rate": round(
            (len(by_case) - len(volatile)) / len(by_case), 8
        ),
        "volatile_case_ids": sorted(volatile),
        "total_http_errors": sum(row["http_errors"] for row in rows),
        "total_false_aborts": sum(row["false_abort_count"] for row in rows),
    }


def _combine_by_repetition(
    student: list[dict[str, Any]], teacher: list[dict[str, Any]], repetitions: int
) -> list[dict[str, Any]]:
    combined = student + teacher
    ids_by_repetition: dict[int, set[str]] = defaultdict(set)
    for row in combined:
        repetition = int(row["repetition"])
        if row["case_id"] in ids_by_repetition[repetition]:
            raise ValueError("student and teacher routed subsets overlap")
        ids_by_repetition[repetition].add(row["case_id"])
    if any(len(ids_by_repetition[index]) != 150 for index in range(repetitions)):
        raise ValueError("each routed repetition must reconstruct all 150 cases")
    return combined


def _performance_row(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    latency = summary["inference"]["request_latency_ms"]
    ttft = summary["inference"]["ttft_ms"]
    return {
        "concurrency": report["concurrency"],
        "runs": summary["runs"],
        "success_rate": round(summary["successful_runs"] / summary["runs"], 8),
        "http_errors": summary["http_errors"],
        "mean_latency_ms": latency["mean"],
        "p95_latency_ms": latency["p95"],
        "mean_ttft_ms": ttft["mean"],
        "p95_ttft_ms": ttft["p95"],
        "throughput_requests_per_second": summary["request_throughput_per_second"],
    }


def performance_matrix(reports_root: Path, role: str) -> list[dict[str, Any]]:
    reports = []
    for concurrency in (1, 4, 16):
        reports.append(
            _read_json(
                reports_root
                / f"stage30-{role}-performance-c{concurrency}-t0-r1-v1"
                / "report.json"
            )
        )
    reports.append(
        _read_json(
            reports_root
            / f"stage30-{role}-stability-c8-t0-r5-v1"
            / "report.json"
        )
    )
    return sorted((_performance_row(report) for report in reports), key=lambda row: row["concurrency"])


def _parse_gpu_log(path: Path) -> dict[str, dict[str, int]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        label, rest = line.split(" ", 1)
        values = [int(part.strip().split(" ", 1)[0]) for part in rest.split(",")]
        result[label] = {
            "memory_used_mib": values[0],
            "memory_total_mib": values[1],
            "utilization_percent": values[2],
        }
    return result


def build(reports_root: Path, dataset_root: Path) -> dict[str, Any]:
    loaded_runs = {
        key: _read_jsonl(reports_root / directory / "runs.jsonl")
        for key, directory in RUNS.items()
    }
    deterministic_runs = _combine_by_repetition(
        loaded_runs["student_deterministic"],
        loaded_runs["teacher_deterministic"],
        5,
    )
    stochastic_runs = _combine_by_repetition(
        loaded_runs["student_stochastic"], loaded_runs["teacher_stochastic"], 3
    )
    deterministic = summarize_repetitions(deterministic_runs, 5)
    stochastic = summarize_repetitions(stochastic_runs, 3)
    student_performance = performance_matrix(reports_root, "student")
    teacher_performance = performance_matrix(reports_root, "teacher")
    gpu = _parse_gpu_log(reports_root / "stage30-gpu-metrics.log")
    c8_student = next(row for row in student_performance if row["concurrency"] == 8)
    c8_teacher = next(row for row in teacher_performance if row["concurrency"] == 8)
    c16_http_errors = sum(
        next(row for row in matrix if row["concurrency"] == 16)["http_errors"]
        for matrix in (student_performance, teacher_performance)
    )
    gates = {
        "deterministic_minimum_accuracy_at_least_94_percent": deterministic[
            "minimum_success_rate"
        ]
        >= 0.94,
        "deterministic_spread_at_most_2_percentage_points": deterministic[
            "spread_percentage_points"
        ]
        <= 2.0,
        "deterministic_action_stability_at_least_98_percent": deterministic[
            "action_stability_rate"
        ]
        >= 0.98,
        "deterministic_http_errors_zero": deterministic["total_http_errors"] == 0,
        "deterministic_false_abort_rate_at_most_1_percent": all(
            row["false_abort_rate"] <= 0.01
            for row in deterministic["repetitions"]
        ),
        "stochastic_minimum_accuracy_at_least_90_percent": stochastic[
            "minimum_success_rate"
        ]
        >= 0.90,
        "c8_p95_latency_at_most_5_seconds": max(
            c8_student["p95_latency_ms"], c8_teacher["p95_latency_ms"]
        )
        <= 5000,
        "c16_http_errors_zero": c16_http_errors == 0,
        "gpu_released_after_suite": gpu["final_idle"]["memory_used_mib"] <= 100,
    }
    manifest_path = dataset_root / "manifest.json"
    return {
        "schema_version": "travel-agent-stage30-stability-report.v1",
        "status": "passed_shadow_entry_gate" if all(gates.values()) else "failed_shadow_entry_gate",
        "passed": all(gates.values()),
        "gates": gates,
        "route": _read_json(manifest_path),
        "deterministic_temperature_0": deterministic,
        "stochastic_temperature_0_2": stochastic,
        "performance": {
            "student_stage28_dpo_4b": student_performance,
            "teacher_8b": teacher_performance,
        },
        "gpu": gpu,
        "conclusions": [
            "Temperature-0 routed accuracy is evaluated across five complete paired repetitions.",
            "Temperature-0.2 evaluates light decoding variance across three repetitions.",
            "Concurrency rows are diagnostic because 4B and 8B were served sequentially on one RTX 4090.",
            "Passing this report authorizes Shadow evaluation only, not production Canary traffic.",
        ],
        "source": {
            "route_manifest": manifest_path.as_posix(),
            "route_manifest_sha256": _sha256(manifest_path),
            "run_reports": {
                key: {
                    "runs_sha256": _sha256(reports_root / directory / "runs.jsonl"),
                    "report_sha256": _sha256(reports_root / directory / "report.json"),
                }
                for key, directory in RUNS.items()
            },
        },
    }


def _render_performance(lines: list[str], label: str, rows: list[dict[str, Any]]) -> None:
    lines.extend(["", f"### {label}", "", "| 并发 | 正确率 | 吞吐 req/s | 平均延迟 | P95 延迟 | HTTP 错误 |", "|---:|---:|---:|---:|---:|---:|"])
    for row in rows:
        lines.append(
            f"| {row['concurrency']} | {row['success_rate'] * 100:.2f}% | "
            f"{row['throughput_requests_per_second']:.3f} | {row['mean_latency_ms']:.1f} ms | "
            f"{row['p95_latency_ms']:.1f} ms | {row['http_errors']} |"
        )


def render_markdown(report: dict[str, Any]) -> str:
    deterministic = report["deterministic_temperature_0"]
    stochastic = report["stochastic_temperature_0_2"]
    lines = [
        "# TravelAgent Stage30：路由稳定性与并发准入报告",
        "",
        f"> 结论：**{'通过 Shadow 准入门' if report['passed'] else '未通过 Shadow 准入门'}**",
        "",
        "## 多轮稳定性",
        "",
        "| 模式 | 轮数 | 最低正确率 | 最高正确率 | 波动 | 动作稳定率 | HTTP 错误 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| temperature=0 | 5 | {deterministic['minimum_success_rate'] * 100:.2f}% | "
        f"{deterministic['maximum_success_rate'] * 100:.2f}% | {deterministic['spread_percentage_points']:.2f} pp | "
        f"{deterministic['action_stability_rate'] * 100:.2f}% | {deterministic['total_http_errors']} |",
        f"| temperature=0.2 | 3 | {stochastic['minimum_success_rate'] * 100:.2f}% | "
        f"{stochastic['maximum_success_rate'] * 100:.2f}% | {stochastic['spread_percentage_points']:.2f} pp | "
        f"{stochastic['action_stability_rate'] * 100:.2f}% | {stochastic['total_http_errors']} |",
        "",
        "## 并发矩阵",
    ]
    _render_performance(lines, "Stage28 DPO 4B 路径", report["performance"]["student_stage28_dpo_4b"])
    _render_performance(lines, "Qwen3-8B 路径", report["performance"]["teacher_8b"])
    lines.extend(["", "## 准入门", ""])
    lines.extend(
        f"- {'通过' if passed else '未通过'}：`{name}`"
        for name, passed in report["gates"].items()
    )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "通过本报告只表示可以进入无用户影响的 Shadow 阶段，不表示可以直接开放 Canary 流量。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports-root", type=Path, default=Path("ml/agentic/reports"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "ml/agentic/datasets/external-benchmark-v1/deepseek-v4-flash-stage29-v1/stage30-routed-v1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/agentic/reports/stage30-stability-gate-v1"),
    )
    args = parser.parse_args()
    report = build(args.reports_root, args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "gates": report["gates"],
                "deterministic": report["deterministic_temperature_0"],
                "stochastic": report["stochastic_temperature_0_2"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
