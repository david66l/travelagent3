"""Run one local checkpoint through the production snapshot Agent Loop."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.corpus_generation import build_curriculum_case  # noqa: E402
from agentic.environment import TravelAgentEnvironment  # noqa: E402
from agentic.local_policy import LocalCheckpointAgentPolicy  # noqa: E402
from agentic.policy import ControllerFirstPolicy  # noqa: E402


async def run(checkpoint: str, scenario: int, max_new_tokens: int) -> dict:
    policy = LocalCheckpointAgentPolicy(checkpoint, max_new_tokens=max_new_tokens)
    try:
        task, snapshot = build_curriculum_case(scenario)
        rollout = await TravelAgentEnvironment(task, snapshot).rollout(
            ControllerFirstPolicy(policy)
        )
        return {
            "checkpoint": checkpoint,
            "task_id": task.task_id,
            "status": rollout.episode.status,
            "termination_reason": rollout.episode.termination_reason,
            "actions": [step.action.action for step in rollout.episode.steps],
            "reward": rollout.reward.model_dump(mode="json"),
        }
    finally:
        policy.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(run(args.checkpoint, args.scenario, args.max_new_tokens)),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
