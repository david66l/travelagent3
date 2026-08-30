"""Export authorized Shadow Agent episodes into immutable GRPO snapshots."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import episode_to_grpo_corpus_row  # noqa: E402
from core.database import async_session_maker  # noqa: E402
from models.agentic_evaluation import AgenticEvaluationRecord  # noqa: E402


def _split(partition: str) -> str:
    bucket = int(hashlib.sha256(partition.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 85 else "validation"


async def export(output_dir: Path, *, limit: int) -> dict[str, int]:
    async with async_session_maker() as db:
        result = await db.execute(
            select(AgenticEvaluationRecord)
            .where(
                AgenticEvaluationRecord.mode == "agent",
                AgenticEvaluationRecord.status == "completed",
                AgenticEvaluationRecord.metrics.is_not(None),
                AgenticEvaluationRecord.episode.is_not(None),
            )
            .order_by(AgenticEvaluationRecord.created_at.desc())
            .limit(limit)
        )
        records = list(result.scalars())

    rows: dict[str, list[str]] = {"train": [], "validation": []}
    rejected = 0
    seen: set[str] = set()
    for record in records:
        metrics = record.metrics or {}
        snapshot = record.input_snapshot or {}
        partition = str(snapshot.get("_training_partition") or "")
        if not (
            partition
            and metrics.get("hard_pass")
            and metrics.get("validated_draft")
            and float(metrics.get("task_completion_rate") or 0) >= 1
        ):
            rejected += 1
            continue
        try:
            row = episode_to_grpo_corpus_row(
                record.episode,
                task_id=f"shadow-{record.scenario_id}",
                template_family="shadow-city-trip",
                seed=int(hashlib.sha256(record.scenario_id.encode()).hexdigest()[:8], 16),
            )
        except ValueError:
            rejected += 1
            continue
        fingerprint = row.snapshot.hidden_test_facts["source_content_hash"]
        if fingerprint in seen:
            rejected += 1
            continue
        seen.add(fingerprint)
        rows[_split(partition)].append(row.model_dump_json())

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, values in rows.items():
        (output_dir / f"{split}.jsonl").write_text(
            "\n".join(values) + ("\n" if values else ""), encoding="utf-8"
        )
    report = {
        "completed_records": len(records),
        "train_tasks": len(rows["train"]),
        "validation_tasks": len(rows["validation"]),
        "rejected_quality_partition_or_integrity": rejected,
    }
    (output_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument(
        "--training-authorized",
        action="store_true",
        help="Explicit authorization is mandatory for production-derived trajectories.",
    )
    args = parser.parse_args()
    if not args.training_authorized:
        parser.error("--training-authorized is required for Shadow production data")
    report = asyncio.run(export(args.output_dir, limit=args.limit))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
