"""Build a paired, multi-seed efficacy report for the Stage 3 RL candidate."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_rollouts(report_dirs: list[Path]) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for report_dir in report_dirs:
        path = report_dir / "rollouts.jsonl"
        if not path.is_file():
            raise ValueError(f"missing rollout evidence: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["task_id"]), int(row["sample_index"]), int(row["rollout_seed"]))
            if key in rows:
                raise ValueError(f"duplicate paired rollout key at {path}:{line_number}: {key}")
            rows[key] = row
    return rows


def _success(row: dict[str, Any]) -> bool:
    return row.get("gate_status") == "passed" and bool(
        (row.get("audit_metrics") or {}).get("hard_pass")
    )


def _exact_mcnemar_p(candidate_only: int, baseline_only: int) -> float:
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return 1.0
    tail = min(candidate_only, baseline_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2 * probability)


def _cluster_bootstrap(
    paired: list[tuple[str, bool, bool]], *, seed: int, samples: int
) -> tuple[float, float]:
    by_task: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for task_id, baseline, candidate in paired:
        by_task[task_id].append((baseline, candidate))
    task_ids = sorted(by_task)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        selected = [rng.choice(task_ids) for _ in task_ids]
        observations = [pair for task_id in selected for pair in by_task[task_id]]
        deltas.append(
            sum(candidate - baseline for baseline, candidate in observations) / len(observations)
        )
    deltas.sort()
    return deltas[int(samples * 0.025)], deltas[min(samples - 1, int(samples * 0.975))]


def _variant(task_id: str, row: dict[str, Any] | None = None) -> str:
    verifier_repair = (row or {}).get("verifier_repair") or {}
    if verifier_repair.get("target_action"):
        return f"verifier_repair/{verifier_repair['target_action']}"
    metadata = (row or {}).get("decision_loop") or {}
    if metadata.get("scenario") and metadata.get("evidence_style"):
        label = f"{metadata['scenario']}/{metadata['evidence_style']}"
        if metadata.get("target_position") is not None:
            label += f"/position-{metadata['target_position']}"
        return label
    if "-decision-loop-" in task_id:
        scenario = (
            "change_arguments"
            if "-change-arguments-" in task_id
            else "retry_same_arguments"
        )
        evidence_style = (
            "diagnostic_evidence" if "-diagnostic-" in task_id else "explicit_instruction"
        )
        return f"{scenario}/{evidence_style}"
    return "cross_tool" if "-cross-tool-" in task_id else "search_only"


def build_report(
    baseline_dirs: list[Path],
    candidate_dirs: list[Path],
    *,
    minimum_pairs: int = 128,
    minimum_gain: float = 0.03,
    minimum_candidate_success: float = 0.90,
    maximum_p_value: float = 0.05,
    bootstrap_samples: int = 10000,
    bootstrap_seed: int = 20260827,
) -> dict[str, Any]:
    baseline = _load_rollouts(baseline_dirs)
    candidate = _load_rollouts(candidate_dirs)
    if set(baseline) != set(candidate):
        missing_candidate = sorted(set(baseline) - set(candidate))[:5]
        missing_baseline = sorted(set(candidate) - set(baseline))[:5]
        raise ValueError(
            "paired rollout keys differ; "
            f"missing_candidate={missing_candidate}, missing_baseline={missing_baseline}"
        )

    keys = sorted(baseline)
    paired = [
        (key[0], _success(baseline[key]), _success(candidate[key]))
        for key in keys
    ]
    baseline_successes = sum(item[1] for item in paired)
    candidate_successes = sum(item[2] for item in paired)
    total = len(paired)
    baseline_rate = baseline_successes / total if total else 0.0
    candidate_rate = candidate_successes / total if total else 0.0
    gain = candidate_rate - baseline_rate
    candidate_only = sum(not base and cand for _, base, cand in paired)
    baseline_only = sum(base and not cand for _, base, cand in paired)
    p_value = _exact_mcnemar_p(candidate_only, baseline_only)
    ci_low, ci_high = _cluster_bootstrap(
        paired, seed=bootstrap_seed, samples=bootstrap_samples
    )

    by_variant: dict[str, Counter[str]] = defaultdict(Counter)
    by_city: dict[str, Counter[str]] = defaultdict(Counter)
    for key, (_, base, cand) in zip(keys, paired, strict=True):
        task_id = key[0]
        counts = by_variant[_variant(task_id, baseline[key])]
        counts["pairs"] += 1
        counts["baseline_successes"] += int(base)
        counts["candidate_successes"] += int(cand)
        counts["candidate_only"] += int(not base and cand)
        counts["baseline_only"] += int(base and not cand)
        city = baseline[key].get("city")
        if city:
            city_counts = by_city[str(city)]
            city_counts["pairs"] += 1
            city_counts["baseline_successes"] += int(base)
            city_counts["candidate_successes"] += int(cand)
            city_counts["candidate_only"] += int(not base and cand)
            city_counts["baseline_only"] += int(base and not cand)

    def grouped_metrics(groups: dict[str, Counter[str]]) -> dict[str, dict[str, Any]]:
        return {
            name: {
                **dict(counts),
                "baseline_success_rate": counts["baseline_successes"] / counts["pairs"],
                "candidate_success_rate": counts["candidate_successes"] / counts["pairs"],
                "absolute_gain": (
                    counts["candidate_successes"] - counts["baseline_successes"]
                )
                / counts["pairs"],
            }
            for name, counts in sorted(groups.items())
        }

    variant_metrics = grouped_metrics(by_variant)
    city_metrics = grouped_metrics(by_city)
    gate_errors = []
    if total < minimum_pairs:
        gate_errors.append("INSUFFICIENT_PAIRED_ROLLOUTS")
    if gain < minimum_gain:
        gate_errors.append("INSUFFICIENT_ABSOLUTE_GAIN")
    if candidate_rate < minimum_candidate_success:
        gate_errors.append("CANDIDATE_SUCCESS_BELOW_TARGET")
    if p_value > maximum_p_value:
        gate_errors.append("PAIRED_SIGNIFICANCE_NOT_REACHED")
    if ci_low <= 0:
        gate_errors.append("CLUSTER_BOOTSTRAP_INTERVAL_CROSSES_ZERO")

    return {
        "schema_version": "stage3-rl-gain-report.v1",
        "scope": "paired stochastic policy efficacy; production regression gate is separate",
        "baseline_report_dirs": [str(path) for path in baseline_dirs],
        "candidate_report_dirs": [str(path) for path in candidate_dirs],
        "paired_rollouts": total,
        "tasks": len({item[0] for item in paired}),
        "baseline_successes": baseline_successes,
        "candidate_successes": candidate_successes,
        "baseline_success_rate": baseline_rate,
        "candidate_success_rate": candidate_rate,
        "absolute_gain": gain,
        "relative_error_reduction": (
            (candidate_rate - baseline_rate) / (1 - baseline_rate)
            if baseline_rate < 1
            else 0.0
        ),
        "paired_outcomes": {
            "candidate_only_success": candidate_only,
            "baseline_only_success": baseline_only,
            "both_success": sum(base and cand for _, base, cand in paired),
            "both_fail": sum(not base and not cand for _, base, cand in paired),
        },
        "exact_mcnemar_two_sided_p": p_value,
        "task_cluster_bootstrap_95ci": [ci_low, ci_high],
        "by_variant": variant_metrics,
        "by_city": city_metrics,
        "gate": {
            "passed": not gate_errors,
            "errors": gate_errors,
            "thresholds": {
                "minimum_pairs": minimum_pairs,
                "minimum_gain": minimum_gain,
                "minimum_candidate_success": minimum_candidate_success,
                "maximum_p_value": maximum_p_value,
                "bootstrap_ci_must_exclude_zero": True,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report-dir", type=Path, action="append", required=True)
    parser.add_argument("--candidate-report-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-pairs", type=int, default=128)
    parser.add_argument("--minimum-gain", type=float, default=0.03)
    parser.add_argument("--minimum-candidate-success", type=float, default=0.90)
    parser.add_argument("--maximum-p-value", type=float, default=0.05)
    args = parser.parse_args()
    report = build_report(
        args.baseline_report_dir,
        args.candidate_report_dir,
        minimum_pairs=args.minimum_pairs,
        minimum_gain=args.minimum_gain,
        minimum_candidate_success=args.minimum_candidate_success,
        maximum_p_value=args.maximum_p_value,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
