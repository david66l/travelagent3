"""Route decision-loop GRPO data from empirically learnable semantic strata."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.environment import environment_fingerprint  # noqa: E402
from agentic.grpo_training import load_grpo_corpus, preflight_grpo_corpus  # noqa: E402


def _metadata(row: Any) -> dict[str, Any]:
    metadata = row.snapshot.hidden_test_facts.get("decision_loop_curriculum")
    if not isinstance(metadata, dict):
        raise ValueError(f"{row.task.task_id} lacks decision-loop metadata")
    return metadata


def _stratum(row: Any) -> tuple[str, str]:
    metadata = _metadata(row)
    return str(metadata["scenario"]), str(metadata["evidence_style"])


def _write_jsonl(path: Path, rows: list[Any]) -> str:
    path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    source_dir: Path,
    audit_report: Path,
    output_dir: Path,
    *,
    minimum_train_tasks: int = 100,
) -> dict[str, Any]:
    """Expand audited non-zero-variance groups to their semantic train strata."""
    report = json.loads(audit_report.read_text(encoding="utf-8"))
    decisions = report.get("decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("audit report contains no group decisions")

    source_train = load_grpo_corpus(source_dir / "train.jsonl")
    source_validation = load_grpo_corpus(source_dir / "validation.jsonl")
    by_id = {row.task.task_id: row for row in source_train}
    update_decisions = [
        item
        for item in decisions
        if item.get("route") == "grpo_update" and item.get("zero_variance") is False
    ]
    if not update_decisions:
        raise ValueError("audit found no learnable decision-loop groups")
    unknown = sorted(
        str(item.get("task_id"))
        for item in update_decisions
        if item.get("task_id") not in by_id
    )
    if unknown:
        raise ValueError(f"audited tasks are absent from source train: {unknown[:3]}")

    learnable_strata = {_stratum(by_id[str(item["task_id"])]) for item in update_decisions}
    train = [row for row in source_train if _stratum(row) in learnable_strata]
    validation = [row for row in source_validation if _stratum(row) in learnable_strata]
    if len(train) < minimum_train_tasks:
        raise ValueError(
            f"routed train split is too small: {len(train)}<{minimum_train_tasks}"
        )
    if not validation:
        raise ValueError("routed validation split is empty")

    train_ids = {row.task.task_id for row in train}
    validation_ids = {row.task.task_id for row in validation}
    if train_ids & validation_ids:
        raise ValueError("routed train and validation task IDs overlap")
    fingerprints = [
        environment_fingerprint(row.task, row.snapshot) for row in [*train, *validation]
    ]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("routed environment fingerprints overlap")

    output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {
        "train": _write_jsonl(output_dir / "train.jsonl", train),
        "validation": _write_jsonl(output_dir / "validation.jsonl", validation),
    }
    preflight = preflight_grpo_corpus(
        output_dir,
        minimum_train_tasks=minimum_train_tasks,
        require_dependencies=False,
    )
    if not preflight.ready:
        raise ValueError(f"routed curriculum failed preflight: {preflight.errors}")

    route_counts = Counter(str(item.get("route")) for item in decisions)
    manifest = {
        "schema_version": "stage3-decision-loop-routed-grpo.v1",
        "source_dir": str(source_dir),
        "audit_report": str(audit_report),
        "audit_report_sha256": hashlib.sha256(audit_report.read_bytes()).hexdigest(),
        "selection_unit": "failure scenario x evidence style",
        "selection_rule": (
            "expand semantic strata with at least one audited non-zero-variance "
            "group routed to grpo_update"
        ),
        "audit_routes": dict(sorted(route_counts.items())),
        "audited_update_tasks": len(update_decisions),
        "learnable_strata": [
            {"scenario": scenario, "evidence_style": evidence_style}
            for scenario, evidence_style in sorted(learnable_strata)
        ],
        "counts": {"train": len(train), "validation": len(validation)},
        "split_sha256": hashes,
        "preflight": preflight.model_dump(mode="json"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-train-tasks", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.source_dir,
                args.audit_report,
                args.output_dir,
                minimum_train_tasks=args.minimum_train_tasks,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
