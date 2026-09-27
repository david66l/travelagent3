"""Create a deterministic H-006 rollout subset without changing source rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402


def _target_action(row: GRPOCorpusRow) -> str:
    value = row.snapshot.hidden_test_facts.get("grpo_decision_state")
    return str(value.get("target_action") or "") if isinstance(value, dict) else "normal"


def select_rows(
    rows: list[GRPOCorpusRow],
    *,
    excluded_task_ids: set[str],
    excluded_actions: set[str],
    count: int,
) -> list[GRPOCorpusRow]:
    eligible = [
        row
        for row in rows
        if row.task.task_id not in excluded_task_ids
        and _target_action(row) not in excluded_actions
    ]
    eligible.sort(
        key=lambda row: (
            hashlib.sha256(row.task.task_id.encode()).hexdigest(),
            row.task.task_id,
        )
    )
    if len(eligible) < count:
        raise ValueError(f"only {len(eligible)} eligible rows for requested {count}")
    selected = eligible[:count]
    if len({row.task.task_id for row in selected}) != count:
        raise ValueError("selected rollout sources are not unique")
    return selected


def _read_task_ids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        result.update(row.task.task_id for row in load_grpo_corpus(path))
    return result


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.write_text(
        "".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--exclude-file", type=Path, action="append", default=[])
    parser.add_argument("--exclude-action", action="append", default=[])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    rows = load_grpo_corpus(args.input)[args.offset :]
    selected = select_rows(
        rows,
        excluded_task_ids=_read_task_ids(args.exclude_file),
        excluded_actions=set(args.exclude_action),
        count=args.count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output, selected)
    report = {
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "count": len(selected),
        "excluded_actions": sorted(set(args.exclude_action)),
        "action_counts": dict(sorted(Counter(_target_action(row) for row in selected).items())),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
