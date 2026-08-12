"""Build auditable GRPO-B0 action data from isolated rollout groups."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic.environment import EnvironmentRollout
from agentic.grpo import GRPOGroupAuditor, GRPOGroupDecision
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT, policy_prompt_payload
from agentic.policy_actions import policy_action_schemas
from agentic.sft_dataset import SFTToolCall, SFTToolFunction


GRPO_DATASET_SCHEMA_VERSION = "agent-policy-grpo-b0.v2"


class GRPOGroupCandidate(BaseModel):
    group_id: str = Field(min_length=1)
    rollouts: list[EnvironmentRollout] = Field(min_length=2)


class GRPOActionExample(BaseModel):
    schema_version: str = GRPO_DATASET_SCHEMA_VERSION
    example_id: str
    group_id: str
    task_id: str
    trajectory_id: str
    step_index: int = Field(ge=0)
    initial_state_fingerprint: str
    prompt: list[dict[str, Any]]
    completion: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    episode_reward: float = Field(ge=-1, le=1)
    trajectory_advantage: float
    local_process_signal: float = Field(ge=-1, le=1)
    credit_mode: Literal["trajectory_b0"] = "trajectory_b0"
    environment_version: str
    snapshot_version: str
    reward_config_version: str


class GRPODatasetManifest(BaseModel):
    schema_version: str = GRPO_DATASET_SCHEMA_VERSION
    dataset_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    candidate_groups: int
    accepted_groups: int
    rejected_groups: int
    exported_examples: int
    routes: dict[str, int]
    rejection_codes: dict[str, int]
    reward_config_versions: list[str]
    environment_versions: list[str]
    snapshot_versions: list[str]
    credit_mode: Literal["trajectory_b0"] = "trajectory_b0"


class GRPODatasetResult(BaseModel):
    manifest: GRPODatasetManifest
    group_decisions: list[GRPOGroupDecision]
    examples: list[GRPOActionExample]


class GRPODatasetBuilder:
    """Export only valid, non-zero-variance, learnable rollout groups."""

    def __init__(self, auditor: GRPOGroupAuditor | None = None) -> None:
        self.auditor = auditor or GRPOGroupAuditor()

    def build(self, candidates: list[GRPOGroupCandidate | dict]) -> GRPODatasetResult:
        parsed = [
            item if isinstance(item, GRPOGroupCandidate) else GRPOGroupCandidate(**item)
            for item in candidates
        ]
        group_ids = [item.group_id for item in parsed]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("duplicate GRPO group_id")

        decisions: list[GRPOGroupDecision] = []
        examples: list[GRPOActionExample] = []
        for candidate in parsed:
            decision = self.auditor.evaluate(candidate.group_id, candidate.rollouts)
            decisions.append(decision)
            if not decision.eligible_for_update:
                continue
            advantages = {
                item.trajectory_id: item.standardized_advantage for item in decision.advantages
            }
            for rollout in candidate.rollouts:
                advantage = advantages[rollout.episode.trajectory_id]
                if len(rollout.reward.turn_rewards) != len(rollout.episode.steps):
                    raise ValueError(f"turn reward count mismatch: {rollout.episode.trajectory_id}")
                for step, turn_reward in zip(
                    rollout.episode.steps,
                    rollout.reward.turn_rewards,
                    strict=True,
                ):
                    context = json.dumps(
                        policy_prompt_payload(step.context),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    local_signal = (
                        turn_reward.format
                        + turn_reward.tool
                        + turn_reward.grounding
                        + turn_reward.efficiency
                    ) / 4
                    examples.append(
                        GRPOActionExample(
                            example_id=(
                                f"{candidate.group_id}:"
                                f"{rollout.episode.trajectory_id}:{step.step_index}"
                            ),
                            group_id=candidate.group_id,
                            task_id=rollout.task_id,
                            trajectory_id=rollout.episode.trajectory_id,
                            step_index=step.step_index,
                            initial_state_fingerprint=rollout.initial_state_fingerprint,
                            prompt=[
                                {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
                                {"role": "user", "content": context},
                            ],
                            completion=[
                                {
                                    "role": "assistant",
                                    "tool_calls": [
                                        SFTToolCall(
                                            function=SFTToolFunction(
                                                name=step.action.action,
                                                arguments=step.action.arguments,
                                            )
                                        ).model_dump()
                                    ],
                                }
                            ],
                            tools=policy_action_schemas(step.context.allowed_actions),
                            episode_reward=rollout.reward.episode_reward,
                            trajectory_advantage=advantage,
                            local_process_signal=local_signal,
                            environment_version=rollout.environment_version,
                            snapshot_version=rollout.snapshot_version,
                            reward_config_version=rollout.reward.reward_config_version,
                        )
                    )

        manifest = self._manifest(parsed, decisions, examples)
        return GRPODatasetResult(
            manifest=manifest,
            group_decisions=decisions,
            examples=examples,
        )

    @staticmethod
    def export(result: GRPODatasetResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(output_dir / "train.jsonl", result.examples)
        _write_jsonl(output_dir / "group_decisions.jsonl", result.group_decisions)
        (output_dir / "manifest.json").write_text(
            result.manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _manifest(
        candidates: list[GRPOGroupCandidate],
        decisions: list[GRPOGroupDecision],
        examples: list[GRPOActionExample],
    ) -> GRPODatasetManifest:
        version_input = {
            "groups": [
                {
                    "group_id": item.group_id,
                    "trajectories": [rollout.episode.content_hash for rollout in item.rollouts],
                }
                for item in candidates
            ],
            "examples": [item.example_id for item in examples],
        }
        canonical = json.dumps(
            version_input,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        dataset_version = "grpo-b0-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return GRPODatasetManifest(
            dataset_version=dataset_version,
            candidate_groups=len(candidates),
            accepted_groups=sum(item.eligible_for_update for item in decisions),
            rejected_groups=sum(not item.eligible_for_update for item in decisions),
            exported_examples=len(examples),
            routes=dict(Counter(item.route for item in decisions)),
            rejection_codes=dict(
                Counter(code for item in decisions for code in item.rejection_codes)
            ),
            reward_config_versions=sorted(
                {
                    rollout.reward.reward_config_version
                    for candidate in candidates
                    for rollout in candidate.rollouts
                }
            ),
            environment_versions=sorted(
                {
                    rollout.environment_version
                    for candidate in candidates
                    for rollout in candidate.rollouts
                }
            ),
            snapshot_versions=sorted(
                {
                    rollout.snapshot_version
                    for candidate in candidates
                    for rollout in candidate.rollouts
                }
            ),
        )


def _write_jsonl(path: Path, rows: Sequence[BaseModel]) -> None:
    content = "\n".join(item.model_dump_json() for item in rows)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")
