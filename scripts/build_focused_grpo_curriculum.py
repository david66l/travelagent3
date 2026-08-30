"""Build a train-only focused GRPO curriculum from a verified boundary corpus."""

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


def _metadata(row: GRPOCorpusRow) -> dict[str, Any] | None:
    value = row.snapshot.hidden_test_facts.get("decision_boundary_training")
    return value if isinstance(value, dict) else None


def _stable_select(
    rows: list[GRPOCorpusRow],
    count: int,
    *,
    salt: str,
) -> list[GRPOCorpusRow]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{salt}:{row.task.task_id}".encode()
        ).hexdigest(),
    )
    if len(ordered) < count:
        raise ValueError(f"only {len(ordered)} rows available for {salt}; need {count}")
    return [row.model_copy(deep=True) for row in ordered[:count]]


def _select_split(
    rows: list[GRPOCorpusRow],
    *,
    quotas: dict[tuple[str, str], int],
    anchor_count: int,
    split: str,
) -> list[GRPOCorpusRow]:
    selected: list[GRPOCorpusRow] = []
    for cell, count in quotas.items():
        kind, variant = cell
        candidates = [
            row
            for row in rows
            if (metadata := _metadata(row)) is not None
            and metadata.get("boundary_kind") == kind
            and metadata.get("variant") == variant
        ]
        selected.extend(
            _stable_select(candidates, count, salt=f"{split}:{kind}:{variant}")
        )
    anchors = [row for row in rows if _metadata(row) is None]
    selected.extend(_stable_select(anchors, anchor_count, salt=f"{split}:anchors"))
    return sorted(
        selected,
        key=lambda row: hashlib.sha256(
            f"focused:{split}:{row.task.task_id}".encode()
        ).hexdigest(),
    )


def _write_jsonl(path: Path, rows: list[GRPOCorpusRow]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def build(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    train_quotas = {
        ("unsafe", "necessary_abort"): 64,
        ("unsafe", "actionable_tradeoff"): 32,
        ("infeasible", "necessary_abort"): 16,
        ("infeasible", "actionable_tradeoff"): 32,
        ("missing_tool", "necessary_abort"): 64,
        ("missing_tool", "actionable_tradeoff"): 16,
    }
    validation_quotas = {
        ("unsafe", "necessary_abort"): 16,
        ("unsafe", "actionable_tradeoff"): 8,
        ("infeasible", "necessary_abort"): 4,
        ("infeasible", "actionable_tradeoff"): 4,
        ("missing_tool", "necessary_abort"): 4,
        ("missing_tool", "actionable_tradeoff"): 4,
    }
    source = {
        split: load_grpo_corpus(source_dir / f"{split}.jsonl")
        for split in ("train", "validation")
    }
    output = {
        "train": _select_split(
            source["train"],
            quotas=train_quotas,
            anchor_count=32,
            split="train",
        ),
        "validation": _select_split(
            source["validation"],
            quotas=validation_quotas,
            anchor_count=8,
            split="validation",
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows)

    all_rows = [row for rows in output.values() for row in rows]
    ids = [row.task.task_id for row in all_rows]
    fingerprints = [environment_fingerprint(row.task, row.snapshot) for row in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("focused curriculum contains duplicate task IDs")
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("focused curriculum contains duplicate environment states")

    preflight = preflight_grpo_corpus(
        output_dir,
        minimum_train_tasks=1,
        require_dependencies=False,
    )
    if not preflight.ready:
        raise ValueError("focused curriculum failed preflight")
    manifest = {
        "schema_version": "focused-decision-boundary-grpo.v1",
        "source_dir": str(source_dir),
        "scope": "train split selection only; validation remains evaluation-only",
        "counts": {split: len(rows) for split, rows in output.items()},
        "cell_counts": {
            split: dict(
                Counter(
                    (
                        f"{metadata['boundary_kind']}/{metadata['variant']}"
                        if (metadata := _metadata(row)) is not None
                        else "anchor"
                    )
                    for row in rows
                )
            )
            for split, rows in output.items()
        },
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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
