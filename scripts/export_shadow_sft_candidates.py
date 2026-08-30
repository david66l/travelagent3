"""Export completed, validated Shadow Agent episodes as audited SFT candidates."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from sqlalchemy import select

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.sft_dataset import EpisodeCandidate  # noqa: E402
from core.city_names import canonical_city_name  # noqa: E402
from core.database import async_session_maker  # noqa: E402
from models.agentic_evaluation import AgenticEvaluationRecord  # noqa: E402


def _family(snapshot: dict) -> str:
    slots = snapshot.get("slots") or {}
    profile = snapshot.get("profile") or {}
    interests = slots.get("interests") or profile.get("interests") or []
    return "normal-city-trip:" + "+".join(sorted(str(item).casefold() for item in interests))


async def export(path: Path, *, limit: int, training_authorized: bool) -> dict[str, int]:
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

    candidates: list[EpisodeCandidate] = []
    rejected_quality = 0
    for record in records:
        metrics = record.metrics or {}
        if not (
            metrics.get("hard_pass")
            and metrics.get("validated_draft")
            and float(metrics.get("task_completion_rate") or 0) >= 1
        ):
            rejected_quality += 1
            continue
        snapshot = record.input_snapshot or {}
        profile = snapshot.get("profile") or {}
        slots = snapshot.get("slots") or {}
        city = canonical_city_name(str(slots.get("destination") or profile.get("destination") or ""))
        partition = snapshot.get("_training_partition")
        if not city or not partition:
            rejected_quality += 1
            continue
        candidates.append(
            EpisodeCandidate(
                scenario_id=record.scenario_id,
                source="shadow",
                template_family=_family(snapshot),
                city=city,
                episode=record.episode,
                user_partition_key=str(partition),
                training_authorized=training_authorized,
                contains_production_data=True,
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(item.model_dump_json() + "\n" for item in candidates),
        encoding="utf-8",
    )
    return {
        "completed_records": len(records),
        "exported_candidates": len(candidates),
        "rejected_quality_or_partition": rejected_quality,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument(
        "--training-authorized",
        action="store_true",
        help="Required before production-derived examples can pass SFT review.",
    )
    args = parser.parse_args()
    report = asyncio.run(
        export(
            args.output,
            limit=args.limit,
            training_authorized=args.training_authorized,
        )
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

