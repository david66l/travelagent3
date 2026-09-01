"""Derive a balanced, non-reportable GRPO smoke corpus from a verified corpus."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any


TARGET_ACTIONS = ("retry_solve", "propose_tradeoff", "abort")
DERIVATION_SCHEMA_VERSION = "balanced-verifier-repair-smoke.v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
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


def _target(row: dict[str, Any]) -> str:
    return str(_decision_state(row).get("target_action") or "")


def _pattern(row: dict[str, Any]) -> str:
    return str(_decision_state(row).get("flexibility_pattern") or "")


def _source_key(row: dict[str, Any]) -> tuple[str, str]:
    state = _decision_state(row)
    return (
        str(state.get("source_task_id") or ""),
        str(state.get("source_snapshot_version") or ""),
    )


def _select_train(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one row per action while covering three different contract patterns."""
    indexed = list(enumerate(rows))
    candidates = {
        target: [(index, row) for index, row in indexed if _target(row) == target]
        for target in TARGET_ACTIONS
    }
    if any(not items for items in candidates.values()):
        raise ValueError("train split does not contain every verifier-repair target")
    choices = []
    for combination in itertools.product(*(candidates[target] for target in TARGET_ACTIONS)):
        patterns = {_pattern(row) for _, row in combination}
        if "" in patterns or len(patterns) != len(TARGET_ACTIONS):
            continue
        choices.append(combination)
    if not choices:
        raise ValueError("cannot cover all targets with distinct flexibility patterns")
    selected = min(
        choices,
        key=lambda combination: (
            max(index for index, _ in combination),
            sum(index for index, _ in combination),
            tuple(index for index, _ in combination),
        ),
    )
    return [row for _, row in sorted(selected, key=lambda item: item[0])]


def _select_balanced_prefix(
    rows: list[dict[str, Any]],
    *,
    per_target: int,
) -> list[dict[str, Any]]:
    if per_target <= 0:
        raise ValueError("per_target must be positive")
    remaining = {target: per_target for target in TARGET_ACTIONS}
    selected: list[dict[str, Any]] = []
    for row in rows:
        target = _target(row)
        if remaining.get(target, 0) <= 0:
            continue
        selected.append(row)
        remaining[target] -= 1
    if any(remaining.values()):
        raise ValueError(f"split cannot provide balanced target counts: {remaining}")
    return selected


def derive(
    *,
    source_dir: Path,
    output_dir: Path,
    validation_per_target: int = 4,
) -> dict[str, Any]:
    """Write an immutable smoke-only derivative without reading the test split."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {output_dir}")
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_train_path = source_dir / "train.jsonl"
    source_validation_path = source_dir / "validation.jsonl"
    train = _select_train(_read_jsonl(source_train_path))
    validation = _select_balanced_prefix(
        _read_jsonl(source_validation_path),
        per_target=validation_per_target,
    )
    if {_source_key(row) for row in train} & {_source_key(row) for row in validation}:
        raise ValueError("source-state overlap between derived train and validation")

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": train, "validation": validation}
    for name, rows in splits.items():
        _write_jsonl(output_dir / f"{name}.jsonl", rows)

    manifest = {
        "schema_version": source_manifest.get("schema_version"),
        "decision_schema_version": source_manifest.get("decision_schema_version"),
        "constraint_flexibility_schema_version": source_manifest.get(
            "constraint_flexibility_schema_version"
        ),
        "derivation_schema_version": DERIVATION_SCHEMA_VERSION,
        "reporting_eligibility": "exploratory_smoke_only",
        "source_dir": str(source_dir),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_split_sha256": {
            "train": _sha256(source_train_path),
            "validation": _sha256(source_validation_path),
        },
        "selection_policy": {
            "train": "one row per target with three distinct flexibility patterns",
            "validation": f"first {validation_per_target} rows per target in source order",
            "test": "omitted; smoke derivation must not inspect or tune on test",
        },
        "counts": {name: len(rows) for name, rows in splits.items()},
        "target_counts": {
            name: dict(sorted(Counter(_target(row) for row in rows).items()))
            for name, rows in splits.items()
        },
        "flexibility_pattern_counts": {
            name: dict(sorted(Counter(_pattern(row) for row in rows).items()))
            for name, rows in splits.items()
        },
        "source_counts": {
            name: len({_source_key(row) for row in rows}) for name, rows in splits.items()
        },
        "selected_train": [
            {
                "task_id": row["task"]["task_id"],
                "target_action": _target(row),
                "flexibility_pattern": _pattern(row),
                "source_key": list(_source_key(row)),
            }
            for row in train
        ],
        "split_sha256": {
            name: _sha256(output_dir / f"{name}.jsonl") for name in splits
        },
        "task_overlap": [],
        "source_state_overlap": [],
        "frozen_test_in_training": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-per-target", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(derive(**vars(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
