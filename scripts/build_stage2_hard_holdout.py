"""Build a frozen, unseen hard holdout for policy decisions and recovery loops."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.corpus_generation import build_curriculum_case  # noqa: E402
from agentic.environment import SnapshotToolResponse, environment_fingerprint  # noqa: E402
from agentic.grpo_training import GRPOCorpusRow  # noqa: E402
from scripts.build_adaptive_recovery_corpus import (  # noqa: E402
    derive_adaptive_recovery,
)
from scripts.build_priority_search_corpus import (  # noqa: E402
    derive_priority_search,
    is_eligible as is_priority_eligible,
)


SCHEMA_VERSION = "stage2-hard-holdout.v2"


def _source_row(index: int) -> GRPOCorpusRow:
    task, snapshot = build_curriculum_case(index)
    return GRPOCorpusRow(task=task, snapshot=snapshot)


def _fresh_sources(
    start_index: int,
    count: int,
    predicate: Callable[[GRPOCorpusRow], bool],
) -> tuple[list[GRPOCorpusRow], int]:
    selected: list[GRPOCorpusRow] = []
    index = start_index
    while len(selected) < count:
        row = _source_row(index)
        if predicate(row):
            selected.append(row)
        index += 1
    return selected, index


def _is_retry(row: GRPOCorpusRow) -> bool:
    responses = row.snapshot.tool_responses.get("search_pois") or []
    return bool(len(responses) >= 2 and responses[0].retryable)


def derive_cross_tool_recovery(row: GRPOCorpusRow) -> GRPOCorpusRow:
    """Fail weather once and search once without exceeding per-task retries."""
    if not _is_retry(row):
        raise ValueError("cross-tool recovery requires a retryable search source")
    derived = derive_adaptive_recovery(row)
    derived.task.task_id = f"{row.task.task_id}-cross-tool-recovery"
    derived.task.template_family = f"{row.task.template_family}-cross-tool-recovery"
    derived.task.difficulty = "L4"
    derived.snapshot.environment_version = "travel-cross-tool-recovery.v1"
    derived.snapshot.snapshot_version = "travel-cross-tool-recovery.v1"
    derived.snapshot.state_id = f"{row.snapshot.state_id}-cross-tool-recovery"

    weather_success = derived.snapshot.tool_responses["get_weather"][0]
    weather_timeout = SnapshotToolResponse(
        data=None,
        data_source="unavailable",
        fallback_reason="天气上游暂时超时，请在该子任务的剩余预算内重试。",
        error_code="UPSTREAM_TIMEOUT",
        retryable=True,
    )
    derived.snapshot.tool_responses["get_weather"] = [
        weather_timeout,
        weather_success,
    ]
    derived.snapshot.hidden_test_facts["cross_tool_recovery"] = {
        "weather_failures": 1,
        "search_failures": 1,
        "target_keywords": derived.snapshot.hidden_test_facts["adaptive_recovery"][
            "target_keywords"
        ],
        "source_task_id": row.task.task_id,
    }
    return derived


def build(
    output_dir: Path,
    *,
    start_index: int = 30000,
    per_variant: int = 8,
) -> dict[str, Any]:
    if per_variant < 1:
        raise ValueError("per_variant must be positive")

    cursor = start_index
    priority_sources, cursor = _fresh_sources(
        cursor,
        per_variant,
        lambda row: is_priority_eligible(row) and row.task.slots.get("travel_days") == 3,
    )
    adaptive_sources, cursor = _fresh_sources(cursor, per_variant, _is_retry)
    cross_tool_sources, cursor = _fresh_sources(cursor, per_variant, _is_retry)

    priority = [
        derive_priority_search(
            row,
            target_position="first" if index % 2 == 0 else "last",
        )
        for index, row in enumerate(priority_sources)
    ]
    adaptive = [derive_adaptive_recovery(row) for row in adaptive_sources]
    cross_tool = [derive_cross_tool_recovery(row) for row in cross_tool_sources]
    # Interleave recovery variants so a small prefix screen covers both.
    rows: list[GRPOCorpusRow] = []
    for index in range(per_variant):
        rows.extend((priority[index], adaptive[index], cross_tool[index]))

    task_ids = [row.task.task_id for row in rows]
    fingerprints = [environment_fingerprint(row.task, row.snapshot) for row in rows]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("hard holdout contains duplicate task IDs")
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("hard holdout contains duplicate initial states")

    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir / "test.jsonl"
    serialized = [
        json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
        for row in rows
    ]
    test_path.write_text("\n".join(serialized) + "\n", encoding="utf-8")
    digest = hashlib.sha256(test_path.read_bytes()).hexdigest()
    variants = Counter(
        "cross_tool_recovery"
        if "cross-tool-recovery" in row.task.task_id
        else (
            "adaptive_recovery"
            if "adaptive-recovery" in row.task.task_id
            else "priority_search"
        )
        for row in rows
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": "frozen evaluation only; never use as a training split",
        "start_index": start_index,
        "next_unused_index": cursor,
        "rows": len(rows),
        "variant_counts": dict(sorted(variants.items())),
        "difficulty_counts": dict(Counter(row.task.difficulty for row in rows)),
        "task_ids": task_ids,
        "test_sha256": digest,
        "split_overlap": [],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=30000)
    parser.add_argument("--per-variant", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            build(
                args.output_dir,
                start_index=args.start_index,
                per_variant=args.per_variant,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
