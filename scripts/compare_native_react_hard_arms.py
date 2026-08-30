"""Build an auditable paired comparison for frozen Native ReAct hard arms."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


METRICS = ("total_tokens", "policy_calls", "model_calls", "tool_calls", "latency_ms")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _exact_mcnemar(improvements: int, regressions: int) -> float:
    discordant = improvements + regressions
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index) * (0.5**discordant)
        for index in range(min(improvements, regressions) + 1)
    )
    return round(min(1.0, 2 * tail), 8)


def _cluster_bootstrap_delta(
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    value: Callable[[dict[str, Any]], float],
    *,
    samples: int,
    seed: int,
) -> list[float]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for case_id, row in baseline.items():
        by_family[str(row["family"])].append(case_id)
    families = sorted(by_family)
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        sampled_ids = [
            case_id
            for _family in range(len(families))
            for family in [families[rng.randrange(len(families))]]
            for case_id in by_family[family]
        ]
        deltas.append(
            statistics.fmean(
                value(candidate[case_id]) - value(baseline[case_id])
                for case_id in sampled_ids
            )
        )
    return [
        round(_percentile(deltas, 0.025), 6),
        round(_percentile(deltas, 0.975), 6),
    ]


def _records(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("each arm must contain non-empty records")
    result = {str(row["case_id"]): dict(row) for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate case_id in report")
    for row in result.values():
        metadata = row.get("benchmark_metadata") or {}
        row["family"] = str(metadata.get("family") or "unknown")
        fallbacks = sum(
            bool((action.get("route_trace") or {}).get("fallback_used"))
            for action in row.get("actions") or []
        )
        row["model_calls"] = int(row.get("policy_calls") or 0) + fallbacks
    return result


def _validate_protocol(reports: dict[str, dict[str, Any]]) -> None:
    baseline_name = next(iter(reports))
    baseline = reports[baseline_name]
    benchmark = baseline.get("benchmark") or {}
    runtime = baseline.get("runtime") or {}
    expected = {
        "split": benchmark.get("split"),
        "frozen_file_sha256": benchmark.get("frozen_file_sha256"),
        "dataset_sha256": benchmark.get("dataset_sha256"),
        "selected_case_ids": benchmark.get("selected_case_ids"),
        "base_seed": runtime.get("base_seed"),
        "temperature": runtime.get("temperature"),
        "policy_protocol": runtime.get("policy_protocol"),
    }
    for name, report in reports.items():
        candidate_benchmark = report.get("benchmark") or {}
        candidate_runtime = report.get("runtime") or {}
        actual = {
            "split": candidate_benchmark.get("split"),
            "frozen_file_sha256": candidate_benchmark.get("frozen_file_sha256"),
            "dataset_sha256": candidate_benchmark.get("dataset_sha256"),
            "selected_case_ids": candidate_benchmark.get("selected_case_ids"),
            "base_seed": candidate_runtime.get("base_seed"),
            "temperature": candidate_runtime.get("temperature"),
            "policy_protocol": candidate_runtime.get("policy_protocol"),
        }
        mismatches = [key for key, value in expected.items() if actual[key] != value]
        if mismatches:
            raise ValueError(f"arm {name} protocol mismatch: {', '.join(mismatches)}")
        if set(_records(report)) != set(_records(baseline)):
            raise ValueError(f"arm {name} case ids differ from {baseline_name}")


def _route_audit(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    policy_calls = 0
    traced_calls = 0
    specialist_calls = 0
    generalist_calls = 0
    fallbacks = 0
    specialist_actions: Counter[str] = Counter()
    for row in records.values():
        for action in row.get("actions") or []:
            if action.get("source") != "policy":
                continue
            policy_calls += 1
            trace = action.get("route_trace")
            if not trace:
                continue
            traced_calls += 1
            target = trace.get("executed_target")
            if target == "student":
                specialist_calls += 1
                specialist_actions[str(action.get("action"))] += 1
            elif target == "teacher":
                generalist_calls += 1
            fallbacks += int(bool(trace.get("fallback_used")))
    return {
        "policy_calls": policy_calls,
        "traced_calls": traced_calls,
        "trace_coverage": round(traced_calls / policy_calls, 6) if policy_calls else 1.0,
        "specialist_calls": specialist_calls,
        "generalist_calls": generalist_calls,
        "fallbacks": fallbacks,
        "specialist_action_counts": dict(specialist_actions),
        "specialist_scope_valid": set(specialist_actions) <= {"get_poi_detail"},
    }


def _arm_summary(report: dict[str, Any]) -> dict[str, Any]:
    records = _records(report)
    summary = report.get("summary") or {}
    return {
        "model": (report.get("runtime") or {}).get("policy_model"),
        "topology": (report.get("runtime") or {}).get("policy_topology", "single-model"),
        "passed": sum(bool(row.get("passed")) for row in records.values()),
        "total": len(records),
        "pass_rate": round(
            sum(bool(row.get("passed")) for row in records.values()) / len(records), 6
        ),
        "cluster_bootstrap_95ci": summary.get("cluster_bootstrap_95ci"),
        "verifier_final_hard_pass_rate": summary.get("verifier_final_hard_pass_rate"),
        "bounded_recovery_rate": summary.get("bounded_recovery_rate"),
        "mean_tokens": summary.get("mean_tokens"),
        "mean_policy_calls": summary.get("mean_policy_calls"),
        "mean_tool_calls": summary.get("mean_tool_calls"),
        "mean_latency_ms": summary.get("mean_latency_ms"),
        "failure_counts": summary.get("failure_counts") or {},
        "route_audit": _route_audit(records),
    }


def _paired(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = _records(baseline_report)
    candidate = _records(candidate_report)
    case_ids = sorted(baseline)
    improvements = [
        case_id
        for case_id in case_ids
        if candidate[case_id].get("passed") and not baseline[case_id].get("passed")
    ]
    regressions = [
        case_id
        for case_id in case_ids
        if baseline[case_id].get("passed") and not candidate[case_id].get("passed")
    ]
    baseline_rate = statistics.fmean(bool(baseline[row].get("passed")) for row in case_ids)
    candidate_rate = statistics.fmean(bool(candidate[row].get("passed")) for row in case_ids)
    metric_deltas: dict[str, Any] = {}
    for metric in METRICS:
        mean_delta = statistics.fmean(
            float(candidate[row].get(metric) or 0) - float(baseline[row].get(metric) or 0)
            for row in case_ids
        )
        metric_deltas[metric] = {
            "paired_mean_delta": round(mean_delta, 6),
            "family_cluster_bootstrap_95ci": _cluster_bootstrap_delta(
                baseline,
                candidate,
                lambda record, key=metric: float(record.get(key) or 0),
                samples=samples,
                seed=seed,
            ),
        }
    family_rows: dict[str, Any] = {}
    for family in sorted({row["family"] for row in baseline.values()}):
        ids = [case_id for case_id in case_ids if baseline[case_id]["family"] == family]
        before = statistics.fmean(bool(baseline[row].get("passed")) for row in ids)
        after = statistics.fmean(bool(candidate[row].get("passed")) for row in ids)
        family_rows[family] = {
            "total": len(ids),
            "baseline_pass_rate": round(before, 6),
            "candidate_pass_rate": round(after, 6),
            "delta_pp": round(100 * (after - before), 3),
        }
    def describe(case_id: str) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "family": baseline[case_id]["family"],
            "baseline_failures": baseline[case_id].get("failures") or [],
            "candidate_failures": candidate[case_id].get("failures") or [],
        }
    return {
        "baseline_pass_rate": round(baseline_rate, 6),
        "candidate_pass_rate": round(candidate_rate, 6),
        "absolute_delta_pp": round(100 * (candidate_rate - baseline_rate), 3),
        "family_cluster_bootstrap_delta_95ci_pp": [
            round(100 * value, 3)
            for value in _cluster_bootstrap_delta(
                baseline,
                candidate,
                lambda record: float(bool(record.get("passed"))),
                samples=samples,
                seed=seed,
            )
        ],
        "candidate_only_success": len(improvements),
        "baseline_only_success": len(regressions),
        "exact_mcnemar_p": _exact_mcnemar(len(improvements), len(regressions)),
        "improvements": [describe(case_id) for case_id in improvements],
        "regressions": [describe(case_id) for case_id in regressions],
        "metric_deltas": metric_deltas,
        "by_family": family_rows,
    }


def build_comparison(
    reports: dict[str, dict[str, Any]],
    *,
    baseline: str,
    samples: int = 10_000,
    seed: int = 20260831,
) -> dict[str, Any]:
    if baseline not in reports:
        raise ValueError(f"unknown baseline: {baseline}")
    _validate_protocol(reports)
    comparisons = {
        name: _paired(reports[baseline], report, samples=samples, seed=seed)
        for name, report in reports.items()
        if name != baseline
    }
    return {
        "schema_version": "native-react-hard-paired-comparison.v1",
        "scope": "frozen-dev-model-selection-only",
        "baseline": baseline,
        "bootstrap": {
            "unit": "scenario_family",
            "samples": samples,
            "seed": seed,
        },
        "protocol": {
            **(next(iter(reports.values())).get("benchmark") or {}),
            "temperature": (next(iter(reports.values())).get("runtime") or {}).get(
                "temperature"
            ),
            "policy_protocol": (next(iter(reports.values())).get("runtime") or {}).get(
                "policy_protocol"
            ),
        },
        "arms": {name: _arm_summary(report) for name, report in reports.items()},
        "paired_vs_baseline": comparisons,
        "claim_boundary": (
            "Dev is for model selection and failure analysis. Do not claim final "
            "generalization until the sealed test is opened once."
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Native ReAct 困难集配对对照",
        "",
        "> 当前仅为冻结 Dev 集模型选择证据；最终泛化结论必须等待一次性 sealed Test。",
        "",
        "| Arm | 通过率 | Verifier 硬通过率 | 平均 Token | 平均策略调用 | 路由覆盖 | 专家调用 | 回退 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in report["arms"].items():
        audit = arm["route_audit"]
        lines.append(
            f"| {name} | {arm['passed']}/{arm['total']} ({arm['pass_rate']:.2%}) "
            f"| {float(arm.get('verifier_final_hard_pass_rate') or 0):.2%} "
            f"| {float(arm.get('mean_tokens') or 0):.1f} "
            f"| {float(arm.get('mean_policy_calls') or 0):.2f} "
            f"| {audit['trace_coverage']:.2%} | {audit['specialist_calls']} "
            f"| {audit['fallbacks']} |"
        )
    lines.extend(["", "## 相对 Base 的配对差异", ""])
    for name, paired in report["paired_vs_baseline"].items():
        token_delta = paired["metric_deltas"]["total_tokens"]["paired_mean_delta"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- 成功率变化：{paired['absolute_delta_pp']:+.2f} pp，"
                f"按场景族 Bootstrap 95% CI "
                f"[{paired['family_cluster_bootstrap_delta_95ci_pp'][0]:+.2f}, "
                f"{paired['family_cluster_bootstrap_delta_95ci_pp'][1]:+.2f}] pp。",
                f"- 配对迁移：改善 {paired['candidate_only_success']}，"
                f"回退 {paired['baseline_only_success']}，"
                f"McNemar p={paired['exact_mcnemar_p']:.4f}。",
                f"- 平均 Token 配对变化：{token_delta:+.1f}。",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _parse_arm(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("arm must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("arm must use non-empty NAME=PATH")
    return name.strip(), Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, type=_parse_arm)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    reports = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in args.arm
    }
    if len(reports) != len(args.arm):
        raise ValueError("arm names must be unique")
    report = build_comparison(
        reports,
        baseline=args.baseline,
        samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["arms"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
