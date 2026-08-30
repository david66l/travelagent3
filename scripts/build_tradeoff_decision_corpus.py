"""Derive explicit, verifier-gated tradeoff decisions from official GRPO rows."""

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
    return bool(
        not row.task.missing_slots
        and row.task.feasibility_report.get("feasible", True) is False
        and row.task.feasibility_report.get("reasons")
    )


def derive_tradeoff_decision(row: GRPOCorpusRow) -> GRPOCorpusRow:
    if not is_eligible(row):
        raise ValueError("tradeoff decision requires a grounded infeasible task")
    derived = row.model_copy(deep=True)
    derived.task.task_id = f"{row.task.task_id}-tradeoff-decision"
    derived.task.template_family = f"{row.task.template_family}-tradeoff-decision"
    derived.task.difficulty = "L3"
    derived.task.user_request = (
        f"{row.task.user_request} 如果这些约束无法同时满足，请明确说明冲突，"
        "并给出一到三个可选择的调整方案；不要继续生成看似可行的行程。"
    )
    derived.task.feasibility_report["actionable_alternatives"] = True
    derived.task.feasibility_report["alternatives"] = ["放宽当前约束", "调整行程要求"]
    derived.snapshot.environment_version = "travel-tradeoff-decision.v1"
    derived.snapshot.snapshot_version = "travel-tradeoff-decision.v1"
    derived.snapshot.state_id = f"{row.snapshot.state_id}-tradeoff-decision"
    derived.snapshot.hidden_test_facts["tradeoff_decision"] = {
        "expected_action": "propose_tradeoff",
        "source_task_id": row.task.task_id,
        "reasons": list(row.task.feasibility_report.get("reasons") or []),
    }
    return derived


def build(source_file: Path, output_dir: Path, *, limit: int) -> dict[str, Any]:
    rows = [
        derive_tradeoff_decision(row)
        for row in load_grpo_corpus(source_file)
        if is_eligible(row)
    ][:limit]
    if not rows:
        raise ValueError("source contains no eligible tradeoff tasks")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "train.jsonl"
    output_file.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"))
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    task_ids = [row.task.task_id for row in rows]
    manifest = {
        "schema_version": "tradeoff-decision-corpus.v1",
        "source_file": str(source_file),
        "counts": {"train": len(rows)},
        "unique_task_ids": len(set(task_ids)),
        "expected_action": "propose_tradeoff",
        "negative_contract": (
            "continuing an infeasible request must fail the capability termination gate"
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=512)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    print(json.dumps(build(args.source_file, args.output_dir, limit=args.limit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
