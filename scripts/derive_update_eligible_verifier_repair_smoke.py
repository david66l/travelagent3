"""Derive one preregistered reward-bearing GRPO diagnostic task."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DERIVATION_SCHEMA_VERSION = "update-eligible-verifier-repair-smoke.v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ).encode("utf-8")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision_state(row: dict[str, Any]) -> dict[str, Any]:
    state = (
        row.get("snapshot", {})
        .get("hidden_test_facts", {})
        .get("grpo_decision_state")
    )
    if not isinstance(state, dict):
        raise ValueError("row is missing grpo_decision_state")
    return state


def _eligible(
    *,
    decision: dict[str, Any],
    rollouts: list[dict[str, Any]],
    minimum_reward_std: float,
) -> bool:
    advantages = [
        float(item["standardized_advantage"])
        for item in decision.get("advantages") or []
    ]
    rewards = [float(item["reward"]) for item in decision.get("advantages") or []]
    infrastructure_clean = (
        len(rollouts) == 4
        and all(not item.get("policy_errors") for item in rollouts)
        and all(item.get("actions") for item in rollouts)
        and all(
            not action.get("error_code")
            for item in rollouts
            for action in item.get("actions") or []
        )
    )
    return (
        decision.get("group_size") == 4
        and decision.get("route") == "grpo_update"
        and decision.get("eligible_for_update") is True
        and decision.get("zero_variance") is False
        and float(decision.get("reward_std") or 0.0) >= minimum_reward_std
        and len(set(rewards)) >= 2
        and any(value >= 1.0 for value in rewards)
        and any(value > 0 for value in advantages)
        and any(value < 0 for value in advantages)
        and infrastructure_clean
    )


def derive(
    *,
    source_dir: Path,
    audit_report: Path,
    output_dir: Path,
    target_action: str,
    minimum_reward_std: float = 0.1,
) -> dict[str, Any]:
    """Select the preregistered eligible row for one explicit target action."""
    if minimum_reward_std <= 0:
        raise ValueError("minimum_reward_std must be positive")
    if target_action not in {"retry_solve", "propose_tradeoff"}:
        raise ValueError(
            "target_action must be retry_solve or propose_tradeoff for an update smoke"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")

    source_manifest_path = source_dir / "manifest.json"
    train_path = source_dir / "train.jsonl"
    validation_path = source_dir / "validation.jsonl"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    train = _read_jsonl(train_path)
    validation = _read_jsonl(validation_path)
    audit = json.loads(audit_report.read_text(encoding="utf-8"))
    decisions = {item["task_id"]: item for item in audit.get("decisions") or []}
    rollouts_by_task: dict[str, list[dict[str, Any]]] = {}
    rollout_path = audit_report.parent / "rollouts.jsonl"
    for item in _read_jsonl(rollout_path):
        rollouts_by_task.setdefault(str(item.get("task_id") or ""), []).append(item)

    selected_row: dict[str, Any] | None = None
    selected_decision: dict[str, Any] | None = None
    for row in train:
        state = _decision_state(row)
        if state.get("target_action") != target_action:
            continue
        task_id = str(row.get("task", {}).get("task_id") or "")
        decision = decisions.get(task_id)
        if decision and _eligible(
            decision=decision,
            rollouts=rollouts_by_task.get(task_id, []),
            minimum_reward_std=minimum_reward_std,
        ):
            selected_row = row
            selected_decision = decision
            break
    if selected_row is None or selected_decision is None:
        raise ValueError(
            f"audit did not contain an eligible {target_action!r} task"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train.jsonl", [selected_row])
    _write_jsonl(output_dir / "validation.jsonl", validation)
    state = _decision_state(selected_row)
    manifest = {
        "schema_version": source_manifest.get("schema_version"),
        "decision_schema_version": source_manifest.get("decision_schema_version"),
        "constraint_flexibility_schema_version": source_manifest.get(
            "constraint_flexibility_schema_version"
        ),
        "derivation_schema_version": DERIVATION_SCHEMA_VERSION,
        "reporting_eligibility": "diagnostic_smoke_only",
        "source_dir": str(source_dir),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_split_sha256": {
            "train": _sha256(train_path),
            "validation": _sha256(validation_path),
        },
        "audit_report": str(audit_report),
        "audit_report_sha256": _sha256(audit_report),
        "audit_rollouts_sha256": _sha256(rollout_path),
        "selection_policy": (
            f"first source train row with target_action={target_action}, group_size=4, "
            f"reward_std>={minimum_reward_std:g}, at least one reward>=1, positive and "
            "negative advantages, and no policy/action infrastructure errors"
        ),
        "preregistered_target_action": target_action,
        "minimum_reward_std": minimum_reward_std,
        "selected": {
            "task_id": selected_row["task"]["task_id"],
            "target_action": state.get("target_action"),
            "flexibility_pattern": state.get("flexibility_pattern"),
            "initial_state_fingerprint": selected_decision.get(
                "initial_state_fingerprint"
            ),
            "reward_std": selected_decision.get("reward_std"),
            "rewards": [
                item.get("reward") for item in selected_decision.get("advantages") or []
            ],
            "standardized_advantages": [
                item.get("standardized_advantage")
                for item in selected_decision.get("advantages") or []
            ],
        },
        "counts": {"train": 1, "validation": len(validation)},
        "split_sha256": {
            name: _sha256(output_dir / f"{name}.jsonl")
            for name in ("train", "validation")
        },
        "test": "omitted; derivation does not read or write the sealed test split",
        "frozen_test_in_training": False,
    }
    (output_dir / "manifest.json").write_bytes(
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-action",
        choices=("retry_solve", "propose_tradeoff"),
        required=True,
        help="Preregister the single decision target eligible for this diagnostic.",
    )
    parser.add_argument("--minimum-reward-std", type=float, default=0.1)
    args = parser.parse_args()
    print(json.dumps(derive(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
