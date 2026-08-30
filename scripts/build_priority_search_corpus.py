"""Derive grounded single-keyword search tasks from official GRPO splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402


def is_eligible(row: GRPOCorpusRow) -> bool:
    if row.task.missing_slots:
        return False
    if row.task.feasibility_report.get("feasible", True) is False:
        return False
    interests = list(
        row.task.slots.get("interests") or row.task.profile.get("interests") or []
    )
    responses = row.snapshot.tool_responses.get("search_pois") or []
    return bool(
        len(interests) >= 2
        and responses
        and not responses[0].error_code
        and responses[0].data_source != "unavailable"
    )


def derive_priority_search(
    row: GRPOCorpusRow,
    *,
    target_position: str = "last",
) -> GRPOCorpusRow:
    """Require the first search to follow one explicit user priority."""
    if not is_eligible(row):
        raise ValueError("priority search requires a feasible two-interest search task")
    if target_position not in {"first", "last"}:
        raise ValueError("target_position must be 'first' or 'last'")
    interests = list(
        row.task.slots.get("interests") or row.task.profile.get("interests") or []
    )
    target = str(interests[0] if target_position == "first" else interests[-1])
    suffix = "priority-search" if target_position == "last" else "priority-search-first"
    derived = row.model_copy(deep=True)
    derived.task.task_id = f"{row.task.task_id}-{suffix}"
    derived.task.template_family = f"{row.task.template_family}-{suffix}"
    derived.task.difficulty = "L2"
    derived.task.user_request = (
        f"{row.task.user_request} 首轮搜索时请只使用“{target}”这一个关键词，"
        "不要混入其他兴趣。"
    )
    derived.snapshot.environment_version = "travel-priority-search.v1"
    derived.snapshot.snapshot_version = "travel-priority-search.v1"
    derived.snapshot.state_id = f"{row.snapshot.state_id}-priority-search"
    derived.snapshot.tool_responses["search_pois"][0].expected_arguments = {
        "keywords": [target]
    }
    derived.snapshot.hidden_test_facts["priority_search"] = {
        "target_keywords": [target],
        "target_position": target_position,
        "source_task_id": row.task.task_id,
    }
    return derived


def build(
    source_dir: Path,
    output_dir: Path,
    *,
    train_limit: int,
    validation_limit: int,
    target_position: str = "last",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    task_ids: dict[str, list[str]] = {}
    for split, limit in (("train", train_limit), ("validation", validation_limit)):
        source = load_grpo_corpus(source_dir / f"{split}.jsonl")
        derived = [
            derive_priority_search(row, target_position=target_position)
            for row in source
            if is_eligible(row)
        ][:limit]
        _write_jsonl(output_dir / f"{split}.jsonl", derived)
        counts[split] = len(derived)
        task_ids[split] = [row.task.task_id for row in derived]
    overlap = set(task_ids["train"]) & set(task_ids["validation"])
    if overlap:
        raise ValueError("derived train/validation task ids overlap")
    manifest = {
        "schema_version": "priority-search-corpus.v1",
        "source_dir": str(source_dir),
        "target_position": target_position,
        "derivation": (
            "user request names one grounded interest as the only first-search "
            "keyword; immutable snapshot enforces the same argument"
        ),
        "counts": counts,
        "task_ids": task_ids,
        "split_overlap": [],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_jsonl(path: Path, rows: list[GRPOCorpusRow]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                row.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for row in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--validation-limit", type=int, default=32)
    parser.add_argument(
        "--target-position",
        choices=("first", "last"),
        default="last",
        help="Visible interest position that must be used as the only first-search keyword.",
    )
    args = parser.parse_args()
    if args.train_limit < 1 or args.validation_limit < 1:
        parser.error("split limits must be positive")
    print(
        json.dumps(
            build(
                args.source_dir,
                args.output_dir,
                train_limit=args.train_limit,
                validation_limit=args.validation_limit,
                target_position=args.target_position,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
