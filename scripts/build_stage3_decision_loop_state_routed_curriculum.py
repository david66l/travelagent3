"""Build an on-policy GRPO curriculum from exact learnable audited states."""

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


def _coverage(rows: list[Any]) -> dict[str, dict[str, int]]:
    factors = {
        "scenario": Counter(),
        "evidence_style": Counter(),
        "target_position": Counter(),
        "city": Counter(),
    }
    for row in rows:
        metadata = _metadata(row)
        factors["scenario"][str(metadata["scenario"])] += 1
        factors["evidence_style"][str(metadata["evidence_style"])] += 1
        factors["target_position"][str(metadata["target_position"])] += 1
        factors["city"][str(row.task.slots.get("destination"))] += 1
    return {name: dict(sorted(counts.items())) for name, counts in factors.items()}


def build(
    source_dir: Path,
    audit_report: Path | list[Path],
    output_dir: Path,
    *,
    minimum_train_tasks: int = 64,
) -> dict[str, Any]:
    """Keep exact audited states with mixed outcomes; never expand whole strata."""
    audit_reports = [audit_report] if isinstance(audit_report, Path) else audit_report
    if not audit_reports:
        raise ValueError("at least one audit report is required")
    decisions: list[dict[str, Any]] = []
    for path in audit_reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        report_decisions = report.get("decisions")
        if not isinstance(report_decisions, list) or not report_decisions:
            raise ValueError(f"audit report contains no group decisions: {path}")
        decisions.extend(report_decisions)

    source_train = load_grpo_corpus(source_dir / "train.jsonl")
    source_validation = load_grpo_corpus(source_dir / "validation.jsonl")
    by_id = {row.task.task_id: row for row in source_train}
    selected_decisions = [
        item
        for item in decisions
        if item.get("route") == "grpo_update"
        and item.get("eligible_for_update") is True
        and item.get("zero_variance") is False
        and 0 < float(item.get("success_rate", 0)) < 1
    ]
    selected_ids = sorted({str(item.get("task_id")) for item in selected_decisions})
    unknown = sorted(task_id for task_id in selected_ids if task_id not in by_id)
    if unknown:
        raise ValueError(f"audited tasks are absent from source train: {unknown[:3]}")

    train = [row for row in source_train if row.task.task_id in set(selected_ids)]
    validation = source_validation
    if len(train) < minimum_train_tasks:
        raise ValueError(
            f"exact-state routed train split is too small: {len(train)}<{minimum_train_tasks}"
        )

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
        raise ValueError(f"state-routed curriculum failed preflight: {preflight.errors}")

    manifest = {
        "schema_version": "stage3-decision-loop-state-routed-grpo.v2",
        "source_dir": str(source_dir),
        "audit_reports": [str(path) for path in audit_reports],
        "audit_report_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in audit_reports
        },
        "selection_unit": "exact audited environment state",
        "selection_rule": (
            "route=grpo_update, eligible_for_update=true, zero_variance=false, "
            "and 0<success_rate<1"
        ),
        "audited_decisions": len(decisions),
        "audited_unique_tasks": len(
            {str(item.get("task_id")) for item in decisions}
        ),
        "selected_train_tasks": len(train),
        "counts": {"train": len(train), "validation": len(validation)},
        "coverage": {
            "train": _coverage(train),
            "validation": _coverage(validation),
        },
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
    parser.add_argument("--audit-report", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-train-tasks", type=int, default=64)
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
