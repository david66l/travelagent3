"""Build the auditable Stage-2 production promotion report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DECISION_CLASSES = (
    "semantic-clarification",
    "terminal-injection",
    "actionable-tradeoff",
    "necessary-abort",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decision_class(task_id: str) -> str:
    for name in DECISION_CLASSES:
        if name in task_id:
            return name
    return "other"


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    per_class = {
        name: {"successful_rollouts": 0, "rollouts": 0} for name in DECISION_CLASSES
    }
    task_ids: list[str] = []
    for decision in report["decisions"]:
        task_id = str(decision["task_id"])
        task_ids.append(task_id)
        name = _decision_class(task_id)
        per_class.setdefault(name, {"successful_rollouts": 0, "rollouts": 0})
        group_size = int(decision["group_size"])
        successes = round(float(decision["success_rate"]) * group_size)
        per_class[name]["successful_rollouts"] += successes
        per_class[name]["rollouts"] += group_size
    for metrics in per_class.values():
        metrics["success_rate"] = (
            metrics["successful_rollouts"] / metrics["rollouts"]
            if metrics["rollouts"]
            else 0.0
        )
    gate = report["behavior_gate"]
    return {
        "checkpoint": report["checkpoint"],
        "temperature": report["temperature"],
        "seed": report["seed"],
        "group_size": report["group_size"],
        "task_ids": task_ids,
        "successful_rollouts": gate["successful_rollouts"],
        "rollouts": gate["rollouts"],
        "success_rate": gate["success_rate"],
        "invalid_actions": gate["invalid_actions"],
        "policy_output_errors": gate["policy_output_errors"],
        "policy_argument_errors": gate["policy_argument_errors"],
        "per_class": per_class,
    }


def build(
    *,
    baseline_path: Path,
    stress_path: Path,
    production_paths: list[Path],
    blind_path: Path,
    adapter_path: Path,
    b0_training_path: Path,
    b0_eval_paths: list[Path],
    r1_training_path: Path,
) -> dict[str, Any]:
    baseline = summarize(_load(baseline_path))
    stress = summarize(_load(stress_path))
    production = [summarize(_load(path)) for path in production_paths]
    blind = summarize(_load(blind_path))
    b0_training = _load(b0_training_path)
    b0_evals = [summarize(_load(path)) for path in b0_eval_paths]
    r1_training = _load(r1_training_path)

    errors: list[str] = []
    if len(production) < 2:
        errors.append("PRODUCTION_REQUIRES_TWO_SEEDS")
    if production and any(item["task_ids"] != production[0]["task_ids"] for item in production):
        errors.append("PRODUCTION_TASK_SET_MISMATCH")
    for item in production:
        if item["success_rate"] < 0.95:
            errors.append(f"PRODUCTION_SUCCESS_BELOW_95:seed={item['seed']}")
        if item["per_class"]["necessary-abort"]["successful_rollouts"] < 28:
            errors.append(f"PRODUCTION_ABORT_BELOW_28:seed={item['seed']}")
        for name in DECISION_CLASSES[:-1]:
            before = baseline["per_class"][name]["successful_rollouts"]
            after = item["per_class"][name]["successful_rollouts"]
            if after < before:
                errors.append(f"PRODUCTION_CLASS_REGRESSION:{name}:seed={item['seed']}")
        if item["invalid_actions"] or item["policy_output_errors"] or item["policy_argument_errors"]:
            errors.append(f"PRODUCTION_PROTOCOL_ERROR:seed={item['seed']}")
    if blind["success_rate"] < 0.95:
        errors.append("BLIND_SUCCESS_BELOW_95")
    if any(
        blind["per_class"][name]["success_rate"] < (0.80 if name == "necessary-abort" else 0.99)
        for name in DECISION_CLASSES
    ):
        errors.append("BLIND_CLASS_GATE_FAILED")

    evidence_paths = [
        baseline_path,
        stress_path,
        *production_paths,
        blind_path,
        b0_training_path,
        *b0_eval_paths,
        r1_training_path,
    ]
    return {
        "schema_version": "stage2-production-candidate.v1",
        "status": "offline_qualified" if not errors else "rejected",
        "candidate": str(adapter_path.parent),
        "inference_profile": {
            "temperature": production[0]["temperature"] if production else None,
            "rationale": "low-variance decoding for discrete tool routing",
        },
        "model_artifact": {
            "adapter": str(adapter_path),
            "adapter_sha256": _sha256(adapter_path),
        },
        "same_protocol_model_stress": {
            "baseline": baseline,
            "candidate": stress,
            "successful_rollout_delta": stress["successful_rollouts"]
            - baseline["successful_rollouts"],
            "necessary_abort_delta": stress["per_class"]["necessary-abort"][
                "successful_rollouts"
            ]
            - baseline["per_class"]["necessary-abort"]["successful_rollouts"],
            "scope": "model robustness evidence; not the production decoding gate",
        },
        "production_seed_results": production,
        "blind_result": blind,
        "post_training_ablation": {
            "trajectory_b0": {
                "training_status": b0_training.get("status", "unknown"),
                "evaluation_results": b0_evals,
                "promotion": "rejected_no_behavior_gain",
            },
            "turn_r1": {
                "training_status": r1_training.get("status", "unknown"),
                "gate_errors": r1_training.get("turn_credit_gate_errors", []),
                "turn_credit_totals": r1_training.get("turn_credit_totals", {}),
                "promotion": "rejected_by_training_evidence_gate",
            },
        },
        "promotion_gate": {"passed": not errors, "errors": errors},
        "evidence_sha256": {str(path): _sha256(path) for path in evidence_paths},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--production", type=Path, nargs="+", required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--b0-training", type=Path, required=True)
    parser.add_argument("--b0-eval", type=Path, nargs="+", required=True)
    parser.add_argument("--r1-training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build(
        baseline_path=args.baseline,
        stress_path=args.stress,
        production_paths=args.production,
        blind_path=args.blind,
        adapter_path=args.adapter,
        b0_training_path=args.b0_training,
        b0_eval_paths=args.b0_eval,
        r1_training_path=args.r1_training,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["promotion_gate"], ensure_ascii=False, indent=2))
    return 0 if report["promotion_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
