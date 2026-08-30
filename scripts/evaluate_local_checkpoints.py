"""Evaluate Base, SFT and SFT+GRPO checkpoints on one fixed snapshot suite."""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import sys
import re
from collections import Counter
from pathlib import Path
from statistics import fmean

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.corpus_generation import build_curriculum_case  # noqa: E402
from agentic.environment import TravelAgentEnvironment  # noqa: E402
from agentic.local_policy import LocalCheckpointAgentPolicy  # noqa: E402
from agentic.policy import ControllerFirstPolicy  # noqa: E402
from evaluation.reward_ablation import build_reward_ablation_report  # noqa: E402


async def evaluate_arm(
    name: str,
    checkpoint: str,
    scenarios: list[int],
    max_new_tokens: int,
) -> tuple[dict, list]:
    policy = LocalCheckpointAgentPolicy(checkpoint, max_new_tokens=max_new_tokens)
    rows = []
    rewards = []
    try:
        for scenario in scenarios:
            task, snapshot = build_curriculum_case(scenario)
            rollout = await TravelAgentEnvironment(task, snapshot).rollout(
                ControllerFirstPolicy(policy)
            )
            reward = rollout.reward
            rewards.append(reward)
            rows.append(
                {
                    "scenario": scenario,
                    "task_id": task.task_id,
                    "difficulty": task.difficulty,
                    "status": rollout.episode.status,
                    "termination_reason": rollout.episode.termination_reason,
                    "gate_status": reward.gate_status,
                    "reward": reward.episode_reward,
                    "hard_pass": bool(reward.audit_metrics.get("hard_pass")),
                    "steps": len(rollout.episode.steps),
                    "tool_calls": sum(rollout.tool_call_counts.values()),
                    "tokens": sum(step.action.token_usage for step in rollout.episode.steps),
                }
            )
    finally:
        policy.close()
        gc.collect()

    summary = {
        "name": name,
        "checkpoint": checkpoint,
        "scenarios": len(rows),
        "gate_pass_rate": sum(row["gate_status"] == "passed" for row in rows) / len(rows),
        "hard_pass_rate": sum(row["hard_pass"] for row in rows) / len(rows),
        "mean_reward": round(fmean(row["reward"] for row in rows), 6),
        "mean_steps": round(fmean(row["steps"] for row in rows), 3),
        "mean_tool_calls": round(fmean(row["tool_calls"] for row in rows), 3),
        "mean_tokens": round(fmean(row["tokens"] for row in rows), 3),
        "termination_reasons": dict(Counter(row["termination_reason"] for row in rows)),
        "rows": rows,
    }
    return summary, rewards


async def run(args: argparse.Namespace) -> dict:
    if args.sft_test_file and args.grpo_validation_file:
        sft_ids = {
            json.loads(line)["scenario_id"]
            for line in args.sft_test_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        grpo_ids = {
            json.loads(line)["task"]["task_id"]
            for line in args.grpo_validation_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        held_out_ids = sorted(sft_ids & grpo_ids)
        scenarios = [_scenario_index(item) for item in held_out_ids[: args.scenario_count]]
        selection = "intersection_of_sft_test_and_grpo_validation"
    else:
        scenarios = list(range(args.scenario_start, args.scenario_start + args.scenario_count))
        selection = "explicit_range_smoke_only"
    if len(scenarios) < args.scenario_count:
        raise ValueError(
            f"fixed suite contains only {len(scenarios)} of {args.scenario_count} requested scenarios"
        )
    arms = []
    ablations = {}
    checkpoints = [("base", args.base), ("sft", args.sft)]
    if args.grpo:
        checkpoints.append(("sft_grpo_b0", args.grpo))
    for name, checkpoint in checkpoints:
        summary, rewards = await evaluate_arm(
            name, checkpoint, scenarios, args.max_new_tokens
        )
        arms.append(summary)
        ablations[name] = build_reward_ablation_report(rewards).model_dump(mode="json")
    return {
        "schema_version": "local-policy-fixed-eval.v1",
        "scope": "engineering smoke; checkpoints with bounded steps do not establish efficacy",
        "scenario_selection": selection,
        "scenario_ids": scenarios,
        "arms": arms,
        "reward_ablations": ablations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft", required=True)
    parser.add_argument("--grpo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario-start", type=int, default=0)
    parser.add_argument("--scenario-count", type=int, default=10)
    parser.add_argument("--sft-test-file", type=Path)
    parser.add_argument("--grpo-validation-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.scenario_count < 1:
        parser.error("scenario-count must be positive")
    if bool(args.sft_test_file) != bool(args.grpo_validation_file):
        parser.error("sft-test-file and grpo-validation-file must be supplied together")
    report = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _scenario_index(task_id: str) -> int:
    match = re.match(r"curriculum-(\d+)-", task_id)
    if not match:
        raise ValueError(f"unsupported curriculum task id: {task_id}")
    return int(match.group(1))


if __name__ == "__main__":
    raise SystemExit(main())
