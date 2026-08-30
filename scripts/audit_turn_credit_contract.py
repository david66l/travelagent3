"""Run one deterministic production-loop episode and audit R1-v2 turn credit."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.corpus_generation import CurriculumTeacherPolicy, build_curriculum_case  # noqa: E402
from agentic.environment import TravelAgentEnvironment  # noqa: E402
from agentic.grpo import policy_turn_credit_records  # noqa: E402
from agentic.policy import ControllerFirstPolicy  # noqa: E402


async def audit(case_index: int, gamma: float) -> dict:
    task, snapshot = build_curriculum_case(case_index)
    rollout = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(CurriculumTeacherPolicy())
    )
    records = policy_turn_credit_records(
        rollout.reward,
        rollout.episode,
        gamma=gamma,
    )
    policy_steps = [
        step
        for step in rollout.episode.steps
        if step.action.decision_source != "controller"
    ]
    if len(policy_steps) != len(records):
        raise RuntimeError("policy steps and R1-v2 credit records do not align")
    rows = []
    for step, record in zip(policy_steps, records, strict=True):
        rows.append(
            {
                "step_index": step.step_index,
                "action": step.action.action,
                "arguments": step.action.arguments,
                "error_codes": [
                    item.error.code for item in step.observations if item.error
                ],
                **record.model_dump(mode="json"),
            }
        )
    invalid_positive = sum(
        item["validity"] == "invalid" and item["credit"] > 0 for item in rows
    )
    return {
        "schema_version": "turn-credit-contract-audit.v1",
        "task_id": task.task_id,
        "template_family": task.template_family,
        "episode_status": rollout.episode.status,
        "termination_reason": rollout.episode.termination_reason,
        "episode_reward": rollout.reward.episode_reward,
        "all_steps": len(rollout.episode.steps),
        "controller_steps": sum(
            step.action.decision_source == "controller"
            for step in rollout.episode.steps
        ),
        "policy_steps": len(policy_steps),
        "invalid_action_positive_credit_rate": (
            invalid_positive
            / max(1, sum(item["validity"] == "invalid" for item in rows))
        ),
        "records": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-index", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.gamma <= 1:
        parser.error("gamma must be in (0, 1]")
    result = asyncio.run(audit(args.case_index, args.gamma))
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
