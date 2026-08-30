"""Build a train-only GRPO corpus from empirically learnable policy groups.

The stochastic audit is the source of truth: only ``grpo_update`` tasks may
provide optimization targets.  Already-solved tasks are retained solely as a
small anti-regression support set, while ``sft_repair`` and rejected tasks are
never promoted into GRPO training.
"""

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
from agentic.grpo_training import (  # noqa: E402
    GRPOCorpusRow,
    load_grpo_corpus,
    preflight_grpo_corpus,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata(row: GRPOCorpusRow) -> dict[str, Any] | None:
    value = row.snapshot.hidden_test_facts.get("decision_boundary_training")
    return value if isinstance(value, dict) else None


def _stable(rows: list[GRPOCorpusRow], salt: str) -> list[GRPOCorpusRow]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{salt}:{row.task.task_id}".encode()).hexdigest(),
    )


def _write(path: Path, rows: list[GRPOCorpusRow]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _load_routes(report_paths: list[Path]) -> dict[str, str]:
    routes: dict[str, str] = {}
    for report_path in report_paths:
        raw = report_path.read_text(encoding="utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if isinstance(payload, list):
            decisions = payload
        elif isinstance(payload, dict) and "decisions" in payload:
            decisions = payload.get("decisions")
        elif isinstance(payload, dict) and {"task_id", "route"} <= payload.keys():
            decisions = [payload]
        else:
            decisions = None
        if not isinstance(decisions, list) or not decisions:
            raise ValueError(f"audit report contains no group decisions: {report_path}")
        for decision in decisions:
            task_id = str(decision.get("task_id") or "")
            route = str(decision.get("route") or "")
            if not task_id or route not in {
                "grpo_update",
                "evaluation",
                "sft_repair",
                "reject",
            }:
                raise ValueError("audit report contains an invalid group decision")
            if task_id in routes:
                raise ValueError(f"audit reports contain duplicate task: {task_id}")
            routes[task_id] = route
    return routes


def build(
    source_dir: Path,
    audit_report: Path | list[Path],
    output_dir: Path,
    *,
    support_per_variant: int = 4,
    anchor_count: int = 8,
) -> dict[str, Any]:
    audit_reports = [audit_report] if isinstance(audit_report, Path) else list(audit_report)
    if not audit_reports:
        raise ValueError("at least one audit report is required")
    routes = _load_routes(audit_reports)
    source_train = load_grpo_corpus(source_dir / "train.jsonl")
    source_by_id = {row.task.task_id: row for row in source_train}
    unknown = sorted(set(routes) - set(source_by_id))
    if unknown:
        raise ValueError(f"audit tasks are absent from source train split: {unknown[:3]}")

    update_ids = {task_id for task_id, route in routes.items() if route == "grpo_update"}
    if not update_ids:
        raise ValueError("audit found no learnable grpo_update groups")
    train = [source_by_id[task_id] for task_id in sorted(update_ids)]

    # Preserve both sides of every audited decision boundary without allowing
    # all-failure groups to masquerade as GRPO targets.
    support_pool = [
        row
        for row in source_train
        if row.task.task_id not in update_ids
        and routes.get(row.task.task_id) == "evaluation"
        and _metadata(row) is not None
    ]
    support_cells: Counter[str] = Counter()
    for variant in ("actionable_tradeoff", "necessary_abort"):
        candidates = _stable(
            [row for row in support_pool if _metadata(row).get("variant") == variant],  # type: ignore[union-attr]
            f"support:{variant}",
        )
        selected = candidates[:support_per_variant]
        train.extend(selected)
        support_cells[variant] += len(selected)

    anchors = _stable(
        [
            row
            for row in source_train
            if routes.get(row.task.task_id) == "evaluation"
            and _metadata(row) is None
        ],
        "support:anchors",
    )[:anchor_count]
    train.extend(anchors)
    train = _stable(list({row.task.task_id: row for row in train}.values()), "train:output")

    validation = load_grpo_corpus(source_dir / "validation.jsonl")
    train_ids = {row.task.task_id for row in train}
    validation_ids = {row.task.task_id for row in validation}
    if train_ids & validation_ids:
        raise ValueError("routed train and validation task ids overlap")
    fingerprints = [
        environment_fingerprint(row.task, row.snapshot) for row in [*train, *validation]
    ]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("routed train and validation environment fingerprints overlap")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "train.jsonl", train)
    _write(output_dir / "validation.jsonl", validation)
    preflight = preflight_grpo_corpus(
        output_dir,
        minimum_train_tasks=1,
        require_dependencies=False,
    )
    if not preflight.ready:
        raise ValueError("routed GRPO curriculum failed preflight")

    route_counts = Counter(routes.values())
    manifest = {
        "schema_version": "routed-grpo-curriculum.v1",
        "source_dir": str(source_dir),
        "audit_reports": [str(path) for path in audit_reports],
        "audit_report_sha256": {str(path): _sha256(path) for path in audit_reports},
        "selection_policy": {
            "optimization_targets": "grpo_update only",
            "excluded_from_optimization": ["sft_repair", "reject"],
            "anti_regression_support": "audited evaluation tasks only",
            "support_per_variant": support_per_variant,
            "anchor_count": anchor_count,
        },
        "audit_routes": dict(route_counts),
        "counts": {"train": len(train), "validation": len(validation)},
        "train_update_tasks": len(update_ids),
        "train_support_cells": dict(support_cells),
        "train_anchors": len(anchors),
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
    parser.add_argument("--support-per-variant", type=int, default=4)
    parser.add_argument("--anchor-count", type=int, default=8)
    args = parser.parse_args()
    if args.support_per_variant < 0 or args.anchor_count < 0:
        parser.error("support and anchor counts must be non-negative")
    print(
        json.dumps(
            build(
                args.source_dir,
                args.audit_report,
                args.output_dir,
                support_per_variant=args.support_per_variant,
                anchor_count=args.anchor_count,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
