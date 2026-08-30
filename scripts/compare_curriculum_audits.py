"""Gate checkpoint promotion with paired fixed-task curriculum audits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def compare(
    before_path: Path,
    after_path: Path,
    *,
    maximum_overall_success_drop: float = 0.02,
    maximum_family_success_drop: float = 0.05,
    maximum_unknown_argument_error_rate: float = 0.0,
    maximum_protected_argument_error_rate: float = 0.0,
) -> dict[str, Any]:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    errors = _paired_contract_errors(before, after)
    before_rows = {item["task_id"]: item for item in before.get("decisions") or []}
    after_rows = {item["task_id"]: item for item in after.get("decisions") or []}
    task_family = _task_families(before)
    paired_ids = sorted(set(before_rows) & set(after_rows))

    before_summary = _summary(before_rows, paired_ids, task_family)
    after_summary = _summary(after_rows, paired_ids, task_family)
    overall_drop = before_summary["success_rate"] - after_summary["success_rate"]
    if overall_drop > maximum_overall_success_drop:
        errors.append(
            "OVERALL_SUCCESS_REGRESSION:"
            f"{before_summary['success_rate']:.6f}->{after_summary['success_rate']:.6f}"
        )
    all_families = sorted(
        set(before_summary["families"]) | set(after_summary["families"])
    )
    for family in all_families:
        before_rate = before_summary["families"].get(family, {}).get("success_rate", 0.0)
        after_rate = after_summary["families"].get(family, {}).get("success_rate", 0.0)
        if before_rate - after_rate > maximum_family_success_drop:
            errors.append(
                f"FAMILY_SUCCESS_REGRESSION:{family}:{before_rate:.6f}->{after_rate:.6f}"
            )
    behavior_gate = after.get("behavior_gate") or {}
    unknown_rate = float(behavior_gate.get("unknown_argument_error_rate") or 0.0)
    protected_rate = float(
        behavior_gate.get("protected_argument_error_rate") or 0.0
    )
    if unknown_rate > maximum_unknown_argument_error_rate:
        errors.append(
            "UNKNOWN_ARGUMENT_ERROR_RATE:"
            f"{unknown_rate:.6f}>{maximum_unknown_argument_error_rate:.6f}"
        )
    if protected_rate > maximum_protected_argument_error_rate:
        errors.append(
            "PROTECTED_ARGUMENT_ERROR_RATE:"
            f"{protected_rate:.6f}>{maximum_protected_argument_error_rate:.6f}"
        )

    return {
        "schema_version": "checkpoint-promotion-gate.v1",
        "promoted": not errors,
        "before_checkpoint": before.get("checkpoint"),
        "after_checkpoint": after.get("checkpoint"),
        "paired_contract": {
            "corpus_file": before.get("corpus_file"),
            "seed": before.get("seed"),
            "seed_protocol": before.get("seed_protocol"),
            "family_offset": before.get("family_offset", 0),
            "temperature": before.get("temperature"),
            "quantization": before.get("quantization", "none"),
            "group_size": before.get("group_size"),
            "task_ids": paired_ids,
        },
        "thresholds": {
            "maximum_overall_success_drop": maximum_overall_success_drop,
            "maximum_family_success_drop": maximum_family_success_drop,
            "maximum_unknown_argument_error_rate": maximum_unknown_argument_error_rate,
            "maximum_protected_argument_error_rate": (
                maximum_protected_argument_error_rate
            ),
        },
        "before": before_summary,
        "after": after_summary,
        "success_rate_delta": round(
            after_summary["success_rate"] - before_summary["success_rate"], 8
        ),
        "mean_reward_delta": round(
            after_summary["mean_reward"] - before_summary["mean_reward"], 8
        ),
        "after_behavior_gate": behavior_gate,
        "gate_errors": sorted(errors),
    }


def _paired_contract_errors(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in (
        "corpus_file",
        "seed",
        "seed_protocol",
        "family_offset",
        "temperature",
        "quantization",
        "group_size",
    ):
        if before.get(field, "none") != after.get(field, "none"):
            errors.append(f"PAIRED_CONTRACT_MISMATCH:{field}")
    before_rows = {item["task_id"]: item for item in before.get("decisions") or []}
    after_rows = {item["task_id"]: item for item in after.get("decisions") or []}
    if set(before_rows) != set(after_rows):
        errors.append("PAIRED_TASK_SET_MISMATCH")
    for task_id in set(before_rows) & set(after_rows):
        if before_rows[task_id].get("group_size") != after_rows[task_id].get("group_size"):
            errors.append(f"PAIRED_GROUP_SIZE_MISMATCH:{task_id}")
        if before_rows[task_id].get("initial_state_fingerprint") != after_rows[task_id].get(
            "initial_state_fingerprint"
        ):
            errors.append(f"PAIRED_INITIAL_STATE_MISMATCH:{task_id}")
    return errors


def _task_families(report: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    # rollouts.jsonl is intentionally not needed: task ordering is stratified
    # by family and report.families records block sizes in insertion order.
    decisions = report.get("decisions") or []
    index = 0
    for family, count in (report.get("families") or {}).items():
        for item in decisions[index : index + int(count)]:
            mapping[item["task_id"]] = family
        index += int(count)
    return mapping


def _summary(
    rows: dict[str, dict[str, Any]],
    task_ids: list[str],
    task_family: dict[str, str],
) -> dict[str, Any]:
    weighted_success = 0.0
    weighted_reward = 0.0
    samples = 0
    families: dict[str, dict[str, float]] = defaultdict(
        lambda: {"weighted_success": 0.0, "weighted_reward": 0.0, "samples": 0.0}
    )
    for task_id in task_ids:
        row = rows[task_id]
        group_size = int(row["group_size"])
        weighted_success += float(row["success_rate"]) * group_size
        weighted_reward += float(row["mean_reward"]) * group_size
        samples += group_size
        bucket = families[task_family.get(task_id, "unknown")]
        bucket["weighted_success"] += float(row["success_rate"]) * group_size
        bucket["weighted_reward"] += float(row["mean_reward"]) * group_size
        bucket["samples"] += group_size
    return {
        "tasks": len(task_ids),
        "samples": samples,
        "success_rate": round(weighted_success / samples, 8) if samples else 0.0,
        "mean_reward": round(weighted_reward / samples, 8) if samples else 0.0,
        "families": {
            family: {
                "samples": int(values["samples"]),
                "success_rate": round(
                    values["weighted_success"] / values["samples"], 8
                ),
                "mean_reward": round(
                    values["weighted_reward"] / values["samples"], 8
                ),
            }
            for family, values in sorted(families.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-overall-success-drop", type=float, default=0.02)
    parser.add_argument("--maximum-family-success-drop", type=float, default=0.05)
    parser.add_argument("--maximum-unknown-argument-error-rate", type=float, default=0.0)
    parser.add_argument("--maximum-protected-argument-error-rate", type=float, default=0.0)
    args = parser.parse_args()
    report = compare(
        args.before,
        args.after,
        maximum_overall_success_drop=args.maximum_overall_success_drop,
        maximum_family_success_drop=args.maximum_family_success_drop,
        maximum_unknown_argument_error_rate=args.maximum_unknown_argument_error_rate,
        maximum_protected_argument_error_rate=args.maximum_protected_argument_error_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["promoted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
