"""Build a paired SFT-vs-GRPO report for full-loop recovery evaluations."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--grpo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260829)
    return parser.parse_args()


def _exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _cluster_bootstrap(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for left, right in pairs:
        grouped[str(left["case_id"])].append((left, right))
    case_ids = sorted(grouped)
    if not case_ids or samples < 1:
        return 0.0, 0.0
    rng = random.Random(seed)
    differences: list[float] = []
    for _ in range(samples):
        sampled = [rng.choice(case_ids) for _ in case_ids]
        rows = [pair for case_id in sampled for pair in grouped[case_id]]
        differences.append(
            statistics.fmean(float(right["passed"]) - float(left["passed"]) for left, right in rows)
        )
    differences.sort()
    low = differences[int(0.025 * (len(differences) - 1))]
    high = differences[int(0.975 * (len(differences) - 1))]
    return low, high


def _intent_signature(row: dict[str, Any]) -> str:
    intent = row.get("intent") or {}
    return json.dumps(
        {
            "intent": intent.get("intent"),
            "slots": intent.get("slots"),
            "missing_required": intent.get("missing_required"),
        },
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def compare(args: argparse.Namespace) -> dict[str, Any]:
    sft_report = json.loads(args.sft.read_text(encoding="utf-8"))
    grpo_report = json.loads(args.grpo.read_text(encoding="utf-8"))
    if sft_report.get("benchmark_hash") != grpo_report.get("benchmark_hash"):
        raise ValueError("benchmark hashes do not match")
    for field in ("base_seed", "group_size", "temperature", "seed_protocol"):
        if sft_report.get(field) != grpo_report.get(field):
            raise ValueError(f"paired protocol mismatch: {field}")

    sft = {str(row["rollout_key"]): row for row in sft_report.get("records") or []}
    grpo = {str(row["rollout_key"]): row for row in grpo_report.get("records") or []}
    keys = sorted(set(sft) & set(grpo))
    if not keys:
        raise ValueError("reports have no paired rollouts")
    if set(sft) != set(grpo):
        raise ValueError("reports do not contain identical rollout keys")
    pairs = [(sft[key], grpo[key]) for key in keys]
    both = sum(bool(left["passed"]) and bool(right["passed"]) for left, right in pairs)
    sft_only = sum(bool(left["passed"]) and not bool(right["passed"]) for left, right in pairs)
    grpo_only = sum(not bool(left["passed"]) and bool(right["passed"]) for left, right in pairs)
    neither = len(pairs) - both - sft_only - grpo_only
    sft_successes = both + sft_only
    grpo_successes = both + grpo_only
    low, high = _cluster_bootstrap(
        pairs,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    intent_mismatches = [
        key
        for key, (left, right) in zip(keys, pairs, strict=True)
        if _intent_signature(left) != _intent_signature(right)
    ]

    strata: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        recovery = pair[0].get("recovery") or {}
        strata[f"{recovery.get('scenario')}/{recovery.get('evidence_style')}"].append(pair)
    stratum_rows = {
        key: {
            "paired_rollouts": len(rows),
            "sft_success_rate": round(
                sum(bool(left["passed"]) for left, _ in rows) / len(rows), 4
            ),
            "grpo_success_rate": round(
                sum(bool(right["passed"]) for _, right in rows) / len(rows), 4
            ),
            "gain_pp": round(
                100
                * sum(float(right["passed"]) - float(left["passed"]) for left, right in rows)
                / len(rows),
                3,
            ),
        }
        for key, rows in sorted(strata.items())
    }
    report = {
        "schema_version": "full-agent-loop-recovery-comparison.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_hash": sft_report["benchmark_hash"],
        "protocol": {
            "base_seed": sft_report["base_seed"],
            "group_size": sft_report["group_size"],
            "temperature": sft_report["temperature"],
            "seed_protocol": sft_report["seed_protocol"],
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "models": {
            "sft": sft_report.get("policy_model"),
            "grpo": grpo_report.get("policy_model"),
        },
        "summary": {
            "paired_rollouts": len(pairs),
            "sft_successes": sft_successes,
            "grpo_successes": grpo_successes,
            "sft_success_rate": round(sft_successes / len(pairs), 4),
            "grpo_success_rate": round(grpo_successes / len(pairs), 4),
            "absolute_gain_pp": round(
                100 * (grpo_successes - sft_successes) / len(pairs), 3
            ),
            "both_success": both,
            "sft_only_success": sft_only,
            "grpo_only_success": grpo_only,
            "neither_success": neither,
            "mcnemar_exact_two_sided_p": _exact_mcnemar_p(sft_only, grpo_only),
            "cluster_bootstrap_95ci_gain_pp": [round(100 * low, 3), round(100 * high, 3)],
            "intent_mismatch_count": len(intent_mismatches),
            "mean_tokens": {
                "sft": round(statistics.fmean(float(left.get("total_tokens") or 0) for left, _ in pairs), 3),
                "grpo": round(statistics.fmean(float(right.get("total_tokens") or 0) for _, right in pairs), 3),
            },
            "mean_latency_ms": {
                "sft": round(statistics.fmean(float(left.get("latency_ms") or 0) for left, _ in pairs), 3),
                "grpo": round(statistics.fmean(float(right.get("latency_ms") or 0) for _, right in pairs), 3),
            },
            "strata": stratum_rows,
        },
        "intent_mismatch_rollout_keys": intent_mismatches,
        "paired_outcomes": [
            {
                "rollout_key": key,
                "sft_passed": bool(left["passed"]),
                "grpo_passed": bool(right["passed"]),
                "sft_failures": left.get("failures") or [],
                "grpo_failures": right.get("failures") or [],
            }
            for key, (left, right) in zip(keys, pairs, strict=True)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    final = compare(parse_args())
    print(json.dumps(final["summary"], ensure_ascii=False, indent=2))
