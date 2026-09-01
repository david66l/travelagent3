"""Build a paired source-clustered gain report for real verifier-repair loops."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.build_stage3_rl_gain_report import (
    _cluster_bootstrap,
    _cluster_randomization_test,
    _exact_mcnemar_p,
)


_PROTOCOL_FIELDS = (
    "execution_mode",
    "structured_decoding_mode",
    "corpus_sha256",
    "seed",
    "temperature",
    "group_size",
    "max_new_tokens",
    "max_tool_calling_iterations",
    "load_in_4bit",
    "tasks_per_target",
    "target_offset",
    "tasks",
)
_EXPECTED_TARGETS = frozenset({"retry_solve", "propose_tradeoff", "abort"})


def _load(report_dir: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    path = report_dir / "rollouts.jsonl"
    rows = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["task_id"]), int(row["sample_index"]), int(row["rollout_seed"]))
        if key in rows:
            raise ValueError(f"duplicate rollout key at {path}:{line_number}: {key}")
        rows[key] = row
    return rows


def _load_and_validate_summary(
    report_dir: Path,
    rows: dict[tuple[str, int, int], dict[str, Any]],
) -> dict[str, Any]:
    path = report_dir / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "verifier-repair-full-loop-audit.v1":
        raise ValueError(f"unsupported downstream audit schema: {path}")
    missing = [field for field in _PROTOCOL_FIELDS if field not in report]
    if missing:
        raise ValueError(f"downstream audit protocol is incomplete at {path}: {missing}")
    if report.get("execution_mode") != "react":
        raise ValueError(f"downstream audit is not a real React loop: {path}")
    if report.get("structured_decoding_mode") != "native":
        raise ValueError(f"downstream audit did not use native decoding: {path}")
    if int(report.get("rollouts") or -1) != len(rows):
        raise ValueError(f"downstream report/rollout count mismatch: {path}")
    expected_rollouts = int(report["tasks"]) * int(report["group_size"])
    if expected_rollouts != len(rows):
        raise ValueError(f"downstream task/group protocol mismatch: {path}")
    return report


def _validate_paired_protocol(
    baseline_report: dict[str, Any],
    candidate_report: dict[str, Any],
) -> None:
    if baseline_report.get("topology") != "sft-generalist":
        raise ValueError("baseline downstream audit must use the SFT generalist only")
    if candidate_report.get("topology") != (
        "sft-generalist-plus-verifier-repair-specialist"
    ):
        raise ValueError("candidate downstream audit must use the specialist route")
    mismatches = [
        field
        for field in _PROTOCOL_FIELDS
        if baseline_report.get(field) != candidate_report.get(field)
    ]
    if mismatches:
        raise ValueError(f"paired downstream audit protocols differ: {mismatches}")
    baseline_generalist_sha = baseline_report.get("generalist_adapter_sha256")
    candidate_generalist_sha = candidate_report.get("generalist_adapter_sha256")
    if not baseline_generalist_sha or baseline_generalist_sha != candidate_generalist_sha:
        raise ValueError("paired downstream audits do not use the same frozen generalist")
    if baseline_report.get("specialist") or baseline_report.get("specialist_adapter_sha256"):
        raise ValueError("baseline downstream audit unexpectedly contains a specialist")
    if not candidate_report.get("specialist_adapter_sha256"):
        raise ValueError("candidate specialist adapter provenance is missing")


def _source_cluster(row: dict[str, Any]) -> str:
    source_task = str(row.get("source_task_id") or "")
    source_snapshot = str(row.get("source_snapshot_version") or "")
    if not source_task or not source_snapshot:
        raise ValueError("downstream rollout is missing source-state provenance")
    return f"{source_task}:{source_snapshot}"


def build_report(
    baseline_dir: Path,
    candidate_dir: Path,
    *,
    minimum_pairs: int = 128,
    minimum_pairs_per_target: int = 32,
    minimum_source_clusters: int = 16,
    minimum_gain: float = 0.03,
    minimum_candidate_success: float = 0.80,
    maximum_p_value: float = 0.05,
    minimum_activation_rate: float = 0.80,
    maximum_fallback_rate: float = 0.05,
    bootstrap_samples: int = 10000,
    seed: int = 20260831,
) -> dict[str, Any]:
    baseline = _load(baseline_dir)
    candidate = _load(candidate_dir)
    baseline_report = _load_and_validate_summary(baseline_dir, baseline)
    candidate_report = _load_and_validate_summary(candidate_dir, candidate)
    _validate_paired_protocol(baseline_report, candidate_report)
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate downstream rollout keys differ")
    keys = sorted(baseline)
    source_mismatches = [
        key
        for key in keys
        if _source_cluster(baseline[key]) != _source_cluster(candidate[key])
    ]
    if source_mismatches:
        raise ValueError(f"paired downstream source metadata differs: {source_mismatches[:5]}")
    target_mismatches = [
        key
        for key in keys
        if baseline[key].get("target_action") != candidate[key].get("target_action")
    ]
    if target_mismatches:
        raise ValueError(f"paired downstream targets differ: {target_mismatches[:5]}")

    paired = [
        (
            _source_cluster(baseline[key]),
            bool(baseline[key].get("success")),
            bool(candidate[key].get("success")),
        )
        for key in keys
    ]
    total = len(paired)
    baseline_successes = sum(base for _, base, _ in paired)
    candidate_successes = sum(cand for _, _, cand in paired)
    baseline_rate = baseline_successes / total if total else 0.0
    candidate_rate = candidate_successes / total if total else 0.0
    gain = candidate_rate - baseline_rate
    candidate_only = sum(not base and cand for _, base, cand in paired)
    baseline_only = sum(base and not cand for _, base, cand in paired)
    mcnemar_p = _exact_mcnemar_p(candidate_only, baseline_only)
    cluster_p, cluster_method, cluster_samples = _cluster_randomization_test(
        paired,
        seed=seed,
    )
    ci_low, ci_high = _cluster_bootstrap(paired, seed=seed, samples=bootstrap_samples)
    source_clusters = len({cluster for cluster, _, _ in paired})

    eligible = sum(bool(row.get("repair_review_reached")) for row in candidate.values())
    requests = sum(bool(row.get("specialist_requested")) for row in candidate.values())
    activations = sum(bool(row.get("specialist_executed")) for row in candidate.values())
    fallbacks = sum(bool(row.get("specialist_fallback")) for row in candidate.values())
    scope_violations = sum(bool(row.get("scope_violation")) for row in candidate.values())
    activation_rate = activations / eligible if eligible else 0.0
    request_rate = requests / eligible if eligible else 0.0
    fallback_rate = fallbacks / requests if requests else 0.0

    target_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for key in keys:
        target = str(baseline[key].get("target_action") or "unknown")
        counts = target_counts[target]
        counts["rollouts"] += 1
        counts["baseline_successes"] += int(bool(baseline[key].get("success")))
        counts["candidate_successes"] += int(bool(candidate[key].get("success")))
    by_target = {
        target: {
            **dict(counts),
            "baseline_success_rate": counts["baseline_successes"] / counts["rollouts"],
            "candidate_success_rate": counts["candidate_successes"] / counts["rollouts"],
            "absolute_gain": (
                counts["candidate_successes"] - counts["baseline_successes"]
            )
            / counts["rollouts"],
        }
        for target, counts in sorted(target_counts.items())
    }

    baseline_errors = sum(bool(row.get("policy_errors")) for row in baseline.values())
    candidate_errors = sum(bool(row.get("policy_errors")) for row in candidate.values())
    gate_errors = []
    if total < minimum_pairs:
        gate_errors.append("INSUFFICIENT_PAIRED_ROLLOUTS")
    if source_clusters < minimum_source_clusters:
        gate_errors.append("INSUFFICIENT_INDEPENDENT_SOURCE_CLUSTERS")
    if gain < minimum_gain:
        gate_errors.append("INSUFFICIENT_ABSOLUTE_GAIN")
    if candidate_rate < minimum_candidate_success:
        gate_errors.append("CANDIDATE_SUCCESS_BELOW_TARGET")
    if cluster_p > maximum_p_value:
        gate_errors.append("SOURCE_CLUSTER_SIGNIFICANCE_NOT_REACHED")
    if ci_low <= 0:
        gate_errors.append("SOURCE_CLUSTER_INTERVAL_CROSSES_ZERO")
    if activation_rate < minimum_activation_rate:
        gate_errors.append("SPECIALIST_ACTIVATION_TOO_LOW")
    if fallback_rate > maximum_fallback_rate:
        gate_errors.append("SPECIALIST_FALLBACK_TOO_HIGH")
    if scope_violations:
        gate_errors.append("SPECIALIST_SCOPE_VIOLATION")
    if activations > requests:
        gate_errors.append("SPECIALIST_ROUTE_ACCOUNTING_INVALID")
    if candidate_errors > baseline_errors:
        gate_errors.append("POLICY_ERRORS_REGRESSED")
    regressed_targets = [
        target for target, metrics in by_target.items() if metrics["absolute_gain"] < 0
    ]
    missing_targets = sorted(_EXPECTED_TARGETS - set(by_target))
    undersampled_targets = sorted(
        target
        for target, metrics in by_target.items()
        if int(metrics["rollouts"]) < minimum_pairs_per_target
    )
    if missing_targets:
        gate_errors.append("TARGET_ACTION_COVERAGE_MISSING")
    if undersampled_targets:
        gate_errors.append("TARGET_ACTION_SAMPLE_TOO_SMALL")
    if regressed_targets:
        gate_errors.append("TARGET_ACTION_REGRESSION")

    return {
        "schema_version": "verifier-repair-downstream-gain.v1",
        "scope": "paired real review to retry/solve/validate or safe termination efficacy",
        "baseline_report_dir": str(baseline_dir),
        "candidate_report_dir": str(candidate_dir),
        "paired_tasks": len({key[0] for key in keys}),
        "paired_rollouts": total,
        "independent_source_clusters": source_clusters,
        "baseline_successes": baseline_successes,
        "candidate_successes": candidate_successes,
        "baseline_success_rate": baseline_rate,
        "candidate_success_rate": candidate_rate,
        "absolute_gain": gain,
        "paired_outcomes": {
            "candidate_only_success": candidate_only,
            "baseline_only_success": baseline_only,
            "both_success": sum(base and cand for _, base, cand in paired),
            "both_fail": sum(not base and not cand for _, base, cand in paired),
        },
        "exact_mcnemar_two_sided_p": mcnemar_p,
        "source_cluster_randomization_two_sided_p": cluster_p,
        "source_cluster_randomization_method": cluster_method,
        "source_cluster_randomization_samples": cluster_samples,
        "source_cluster_bootstrap_95ci": [ci_low, ci_high],
        "eligible_specialist_decisions": eligible,
        "specialist_route_requests": requests,
        "specialist_request_rate": request_rate,
        "specialist_route_activations": activations,
        "specialist_activation_rate": activation_rate,
        "specialist_fallbacks": fallbacks,
        "specialist_fallback_rate": fallback_rate,
        "scope_violations": scope_violations,
        "baseline_policy_errors": baseline_errors,
        "candidate_policy_errors": candidate_errors,
        "by_target": by_target,
        "missing_targets": missing_targets,
        "undersampled_targets": undersampled_targets,
        "latency": {
            "baseline_mean_ms": statistics.fmean(
                float(row.get("latency_ms") or 0) for row in baseline.values()
            ),
            "candidate_mean_ms": statistics.fmean(
                float(row.get("latency_ms") or 0) for row in candidate.values()
            ),
        },
        "tokens": {
            "baseline_mean": statistics.fmean(
                float(row.get("completion_tokens") or 0) for row in baseline.values()
            ),
            "candidate_mean": statistics.fmean(
                float(row.get("completion_tokens") or 0) for row in candidate.values()
            ),
        },
        "gate": {
            "passed": not gate_errors,
            "errors": gate_errors,
            "thresholds": {
                "minimum_pairs": minimum_pairs,
                "minimum_pairs_per_target": minimum_pairs_per_target,
                "minimum_source_clusters": minimum_source_clusters,
                "minimum_gain": minimum_gain,
                "minimum_candidate_success": minimum_candidate_success,
                "maximum_source_cluster_p": maximum_p_value,
                "minimum_activation_rate": minimum_activation_rate,
                "maximum_fallback_rate": maximum_fallback_rate,
                "per_target_no_regression": True,
                "policy_errors_no_regression": True,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report-dir", type=Path, required=True)
    parser.add_argument("--candidate-report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-pairs", type=int, default=128)
    parser.add_argument("--minimum-pairs-per-target", type=int, default=32)
    parser.add_argument("--minimum-source-clusters", type=int, default=16)
    parser.add_argument("--minimum-gain", type=float, default=0.03)
    parser.add_argument("--minimum-candidate-success", type=float, default=0.80)
    args = parser.parse_args()
    report = build_report(
        args.baseline_report_dir,
        args.candidate_report_dir,
        minimum_pairs=args.minimum_pairs,
        minimum_pairs_per_target=args.minimum_pairs_per_target,
        minimum_source_clusters=args.minimum_source_clusters,
        minimum_gain=args.minimum_gain,
        minimum_candidate_success=args.minimum_candidate_success,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
