"""Route Stage 3 GRPO tasks by empirically learnable recovery strata."""

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(row: Any) -> dict[str, Any]:
    value = row.snapshot.hidden_test_facts.get("stage3_multiturn_recovery")
    if not isinstance(value, dict):
        raise ValueError(f"{row.task.task_id} lacks Stage 3 recovery metadata")
    return value


def _stratum(row: Any) -> tuple[str, bool]:
    metadata = _metadata(row)
    return str(metadata["message_template_sha256"]), bool(metadata["cross_tool"])


def _write(path: Path, rows: list[Any]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def build(source_dir: Path, audit_report: Path, output_dir: Path) -> dict[str, Any]:
    report = json.loads(audit_report.read_text(encoding="utf-8"))
    decisions = report.get("decisions") or []
    if not decisions:
        raise ValueError("audit report contains no group decisions")

    source_train = load_grpo_corpus(source_dir / "train.jsonl")
    source_validation = load_grpo_corpus(source_dir / "validation.jsonl")
    by_id = {row.task.task_id: row for row in source_train}
    update_decisions = [item for item in decisions if item.get("route") == "grpo_update"]
    if not update_decisions:
        raise ValueError("audit found no learnable GRPO strata")
    unknown = [item["task_id"] for item in update_decisions if item["task_id"] not in by_id]
    if unknown:
        raise ValueError(f"audited tasks are absent from source train: {unknown[:3]}")

    learnable_strata = {_stratum(by_id[item["task_id"]]) for item in update_decisions}
    train = [row for row in source_train if _stratum(row) in learnable_strata]
    validation = [row for row in source_validation if _stratum(row) in learnable_strata]
    if not train or not validation:
        raise ValueError("learnable strata produced an empty split")

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
    _write(output_dir / "train.jsonl", train)
    _write(output_dir / "validation.jsonl", validation)
    preflight = preflight_grpo_corpus(
        output_dir,
        minimum_train_tasks=500,
        require_dependencies=False,
    )
    if not preflight.ready:
        raise ValueError(f"routed curriculum failed preflight: {preflight.errors}")

    route_counts = Counter(str(item.get("route")) for item in decisions)
    manifest = {
        "schema_version": "stage3-routed-multiturn-grpo.v1",
        "source_dir": str(source_dir),
        "audit_report": str(audit_report),
        "audit_report_sha256": _sha256(audit_report),
        "selection_unit": "recovery message template hash x cross-tool flag",
        "selection_rule": (
            "retain every train/validation task whose stratum had at least one "
            "audited group routed to grpo_update"
        ),
        "audit_routes": dict(sorted(route_counts.items())),
        "audited_update_tasks": len(update_decisions),
        "learnable_strata": [
            {"message_template_sha256": digest, "cross_tool": cross_tool}
            for digest, cross_tool in sorted(learnable_strata)
        ],
        "counts": {"train": len(train), "validation": len(validation)},
        "variant_counts": {
            split: dict(
                Counter(
                    "cross_tool" if _metadata(row)["cross_tool"] else "search_only"
                    for row in rows
                )
            )
            for split, rows in (("train", train), ("validation", validation))
        },
        "excluded_routes": ["evaluation-only stratum", "sft_repair-only stratum", "reject"],
        "preflight": preflight.model_dump(mode="json"),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.source_dir, args.audit_report, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
