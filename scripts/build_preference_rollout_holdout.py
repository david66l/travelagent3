"""Recover immutable environment rows for a frozen preference split."""

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

from agentic.distillation import TeacherPreferencePair  # noqa: E402
from agentic.grpo_training import GRPOCorpusRow  # noqa: E402


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build(
    preference_file: Path,
    source_files: list[Path],
    output_file: Path,
) -> dict[str, Any]:
    pairs = [TeacherPreferencePair(**row) for row in _read_jsonl(preference_file)]
    requested = {pair.task_id: pair.family for pair in pairs}
    rows: dict[str, GRPOCorpusRow] = {}
    source_counts = Counter()
    for source in source_files:
        for raw in _read_jsonl(source):
            row = GRPOCorpusRow(**raw)
            task_id = row.task.task_id
            if task_id not in requested:
                continue
            existing = rows.get(task_id)
            if existing is not None and existing != row:
                raise ValueError(f"conflicting environment row for task: {task_id}")
            rows[task_id] = row
            source_counts[str(source)] += 1
    missing = sorted(set(requested) - set(rows))
    if missing:
        raise ValueError(f"preference tasks missing immutable environment rows: {missing[:5]}")

    ordered = [rows[pair.task_id] for pair in pairs]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "\n".join(item.model_dump_json() for item in ordered) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(output_file.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "preference-rollout-holdout.v1",
        "preference_file": str(preference_file),
        "source_files": [str(path) for path in source_files],
        "source_match_counts": dict(source_counts),
        "cases": len(ordered),
        "unique_tasks": len(rows),
        "family_counts": dict(Counter(pair.family for pair in pairs)),
        "sha256": digest,
        "frozen_before_student_training": True,
    }
    (output_file.parent / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preference-file", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, action="append", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.preference_file, args.source_file, args.output_file),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
