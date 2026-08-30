"""Generate verifier-approved SFT episodes and GRPO snapshot task splits."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.corpus_generation import (  # noqa: E402
    CurriculumTeacherPolicy,
    build_curriculum_case,
)
from agentic.environment import TravelAgentEnvironment  # noqa: E402
from agentic.policy import ControllerFirstPolicy  # noqa: E402
from agentic.sft_dataset import EpisodeCandidate  # noqa: E402
from agentic.trajectory import EpisodeReplayVerifier  # noqa: E402


def _split(task_id: str) -> str:
    bucket = int(hashlib.sha256(task_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 85 else "validation"


async def generate(
    count: int,
    concurrency: int,
    *,
    start_index: int = 0,
    execution_mode: str = "controller_first",
) -> tuple[list[EpisodeCandidate], dict]:
    semaphore = asyncio.Semaphore(concurrency)
    replay = EpisodeReplayVerifier()

    async def one(index: int):
        async with semaphore:
            task, snapshot = build_curriculum_case(index)
            teacher = CurriculumTeacherPolicy()
            policy = (
                teacher
                if execution_mode == "policy_driven"
                else ControllerFirstPolicy(teacher)
            )
            rollout = await TravelAgentEnvironment(task, snapshot).rollout(
                policy
            )
            errors = replay.verify(rollout.episode)
            if errors or rollout.reward.gate_status != "passed" or rollout.reward.episode_reward <= 0:
                raise ValueError(
                    f"{task.task_id} failed corpus gates: replay={errors}, "
                    f"reward={rollout.reward.model_dump(mode='json')}"
                )
            candidate = EpisodeCandidate(
                scenario_id=task.task_id,
                source="synthetic",
                template_family=task.template_family,
                city=str(task.slots["destination"]),
                episode=rollout.episode,
            )
            return candidate, task, snapshot, rollout.reward.episode_reward

    stop_index = start_index + count
    results = await asyncio.gather(
        *(one(index) for index in range(start_index, stop_index))
    )
    return [item[0] for item in results], {
        "rows": [(item[1], item[2]) for item in results],
        "start_index": start_index,
        "stop_index_exclusive": stop_index,
        "minimum_reward": min(item[3] for item in results),
        "maximum_reward": max(item[3] for item in results),
    }


def _write_jsonl(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First deterministic curriculum case index (inclusive)",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--execution-mode",
        choices=("policy_driven", "controller_first"),
        default="controller_first",
        help=(
            "policy_driven records every production DAG action as a model-owned "
            "SFT target; controller_first preserves the historical delegated-only corpus"
        ),
    )
    parser.add_argument("--sft-candidates", type=Path, required=True)
    parser.add_argument("--grpo-corpus-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.concurrency < 1 or args.start_index < 0:
        parser.error("count/concurrency must be positive and start-index non-negative")

    candidates, metadata = asyncio.run(
        generate(
            args.count,
            args.concurrency,
            start_index=args.start_index,
            execution_mode=args.execution_mode,
        )
    )
    _write_jsonl(
        args.sft_candidates,
        [candidate.model_dump_json() for candidate in candidates],
    )
    split_rows: dict[str, list[str]] = {"train": [], "validation": []}
    for task, snapshot in metadata["rows"]:
        split_rows[_split(task.task_id)].append(
            json.dumps(
                {
                    "task": task.model_dump(mode="json"),
                    "snapshot": snapshot.model_dump(mode="json"),
                },
                ensure_ascii=False,
            )
        )
    for split, rows in split_rows.items():
        _write_jsonl(args.grpo_corpus_dir / f"{split}.jsonl", rows)
    report = {
        "schema_version": "agentic-corpus-build.v1",
        "execution_mode": args.execution_mode,
        "policy_decision_scope": (
            "all_dag_actions"
            if args.execution_mode == "policy_driven"
            else "delegated_choice_actions_only"
        ),
        "start_index": metadata["start_index"],
        "stop_index_exclusive": metadata["stop_index_exclusive"],
        "executed_episodes": len(candidates),
        "sft_candidate_episodes": len(candidates),
        "grpo_train_tasks": len(split_rows["train"]),
        "grpo_validation_tasks": len(split_rows["validation"]),
        "minimum_reward": metadata["minimum_reward"],
        "maximum_reward": metadata["maximum_reward"],
    }
    args.grpo_corpus_dir.mkdir(parents=True, exist_ok=True)
    (args.grpo_corpus_dir / "generation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
