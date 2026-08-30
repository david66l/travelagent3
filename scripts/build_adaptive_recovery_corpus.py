"""Derive evidence-conditioned recovery tasks from the official GRPO splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402


def derive_adaptive_recovery(row: GRPOCorpusRow) -> GRPOCorpusRow:
    responses = row.snapshot.tool_responses.get("search_pois") or []
    if len(responses) < 2 or not responses[0].retryable:
        raise ValueError("adaptive recovery requires a retryable two-response search")
    interests = list(row.task.slots.get("interests") or row.task.profile.get("interests") or [])
    if len(interests) < 2:
        raise ValueError("adaptive recovery requires at least two grounded interests")
    target = str(interests[-1])
    derived = row.model_copy(deep=True)
    derived.task.task_id = f"{row.task.task_id}-adaptive-recovery"
    derived.task.template_family = f"{row.task.template_family}-adaptive-recovery"
    derived.task.difficulty = "L3"
    derived.snapshot.environment_version = "travel-adaptive-recovery.v1"
    derived.snapshot.snapshot_version = "travel-adaptive-recovery.v1"
    derived.snapshot.state_id = f"{row.snapshot.state_id}-adaptive-recovery"
    first, second = derived.snapshot.tool_responses["search_pois"][:2]
    first.error_code = "QUERY_TOO_BROAD"
    first.fallback_reason = (
        f"查询范围过宽。请保留用户已提供的关键词“{target}”，并仅用该关键词重试。"
    )
    first.retryable = True
    second.expected_arguments = {"keywords": [target]}
    derived.snapshot.hidden_test_facts["adaptive_recovery"] = {
        "target_keywords": [target],
        "source_task_id": row.task.task_id,
    }
    return derived


def build(source_dir: Path, output_dir: Path, *, train_limit: int, validation_limit: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    task_ids: dict[str, list[str]] = {}
    for split, limit in (("train", train_limit), ("validation", validation_limit)):
        source = load_grpo_corpus(source_dir / f"{split}.jsonl")
        eligible = [
            row
            for row in source
            if len(row.snapshot.tool_responses.get("search_pois") or []) >= 2
            and (row.snapshot.tool_responses.get("search_pois") or [])[0].retryable
        ]
        derived = [derive_adaptive_recovery(row) for row in eligible[:limit]]
        _write_jsonl(output_dir / f"{split}.jsonl", derived)
        counts[split] = len(derived)
        task_ids[split] = [row.task.task_id for row in derived]
    overlap = set(task_ids["train"]) & set(task_ids["validation"])
    if overlap:
        raise ValueError("derived train/validation task ids overlap")
    manifest = {
        "schema_version": "adaptive-recovery-corpus.v1",
        "source_dir": str(source_dir),
        "derivation": (
            "retryable broad-query observation names one grounded interest; "
            "second snapshot response requires that exact narrowed keyword"
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
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
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
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
