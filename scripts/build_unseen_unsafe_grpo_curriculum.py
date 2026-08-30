"""Build an unsafe-boundary GRPO curriculum unseen by the bridge SFT stage."""

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


def _stable(rows: list[GRPOCorpusRow], salt: str) -> list[GRPOCorpusRow]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{salt}:{row.task.task_id}".encode()).hexdigest(),
    )


def _sft_scenarios(path: Path) -> set[str]:
    scenarios: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("policy_name") == "VerifierLabelPolicy":
            scenarios.add(str(payload["scenario_id"]))
    return scenarios


def _write(path: Path, rows: list[GRPOCorpusRow]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def build(source_dir: Path, bridge_sft_dir: Path, output_dir: Path) -> dict[str, Any]:
    seen = _sft_scenarios(bridge_sft_dir / "train.jsonl")
    source_train = load_grpo_corpus(source_dir / "train.jsonl")
    unseen = [
        row
        for row in source_train
        if row.task.task_id not in seen
        and (metadata := _metadata(row)) is not None
        and metadata.get("boundary_kind") == "unsafe"
    ]
    by_variant = {
        variant: _stable(
            [
                row
                for row in unseen
                if _metadata(row).get("variant") == variant  # type: ignore[union-attr]
            ],
            f"train:{variant}",
        )
        for variant in ("actionable_tradeoff", "necessary_abort")
    }
    if not all(len(rows) >= 8 for rows in by_variant.values()):
        raise ValueError(
            "bridge SFT must leave at least eight unseen train scenarios per unsafe variant"
        )
    anchors = _stable(
        [row for row in source_train if _metadata(row) is None],
        "train:anchors",
    )[:8]
    train = _stable(
        [*by_variant["actionable_tradeoff"][:8], *by_variant["necessary_abort"][:8], *anchors],
        "train:output",
    )

    source_validation = load_grpo_corpus(source_dir / "validation.jsonl")
    validation: list[GRPOCorpusRow] = []
    for variant in ("actionable_tradeoff", "necessary_abort"):
        validation.extend(
            _stable(
                [
                    row
                    for row in source_validation
                    if (metadata := _metadata(row)) is not None
                    and metadata.get("boundary_kind") == "unsafe"
                    and metadata.get("variant") == variant
                ],
                f"validation:{variant}",
            )[:4]
        )
    validation.extend(
        _stable(
            [row for row in source_validation if _metadata(row) is None],
            "validation:anchors",
        )[:8]
    )
    validation = _stable(validation, "validation:output")

    all_rows = [*train, *validation]
    ids = [row.task.task_id for row in all_rows]
    fingerprints = [environment_fingerprint(row.task, row.snapshot) for row in all_rows]
    if len(ids) != len(set(ids)) or len(fingerprints) != len(set(fingerprints)):
        raise ValueError("unseen unsafe curriculum is not split-unique")

    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "train.jsonl", train)
    _write(output_dir / "validation.jsonl", validation)
    preflight = preflight_grpo_corpus(
        output_dir,
        minimum_train_tasks=1,
        require_dependencies=False,
    )
    if not preflight.ready:
        raise ValueError("unseen unsafe curriculum failed preflight")
    manifest = {
        "schema_version": "unseen-unsafe-grpo-curriculum.v1",
        "source_dir": str(source_dir),
        "bridge_sft_dir": str(bridge_sft_dir),
        "train_sft_scenario_overlap": sum(row.task.task_id in seen for row in train),
        "counts": {"train": len(train), "validation": len(validation)},
        "train_cells": dict(
            Counter(
                (
                    f"{metadata['boundary_kind']}/{metadata['variant']}"
                    if (metadata := _metadata(row)) is not None
                    else "anchor"
                )
                for row in train
            )
        ),
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
    parser.add_argument("--bridge-sft-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.source_dir, args.bridge_sft_dir, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
