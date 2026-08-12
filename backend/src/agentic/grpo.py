"""GRPO-B0 group audit, relative advantages and curriculum routing."""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import Literal

from pydantic import BaseModel, Field

from agentic.environment import EnvironmentRollout


class GRPOGroupConfig(BaseModel):
    config_version: str = "grpo-group-b0.v1"
    minimum_group_size: int = Field(default=4, ge=2)
    zero_variance_epsilon: float = Field(default=1e-8, ge=0)
    preferred_success_rate_min: float = Field(default=0.20, ge=0, le=1)
    preferred_success_rate_max: float = Field(default=0.80, ge=0, le=1)


class RolloutAdvantage(BaseModel):
    trajectory_id: str
    reward: float
    standardized_advantage: float


class GRPOGroupDecision(BaseModel):
    group_id: str
    task_id: str
    initial_state_fingerprint: str
    group_size: int
    mean_reward: float
    reward_std: float
    success_rate: float
    zero_variance: bool
    eligible_for_update: bool
    curriculum_band: Literal["too_easy", "learnable", "too_hard", "invalid"]
    route: Literal["grpo_update", "evaluation", "sft_repair", "reject"]
    rejection_codes: list[str] = Field(default_factory=list)
    advantages: list[RolloutAdvantage] = Field(default_factory=list)


class GRPOGroupAuditor:
    def __init__(self, config: GRPOGroupConfig | None = None) -> None:
        self.config = config or GRPOGroupConfig()

    def evaluate(
        self,
        group_id: str,
        rollouts: list[EnvironmentRollout | dict],
    ) -> GRPOGroupDecision:
        parsed = [
            item if isinstance(item, EnvironmentRollout) else EnvironmentRollout(**item)
            for item in rollouts
        ]
        errors: list[str] = []
        if len(parsed) < self.config.minimum_group_size:
            errors.append("GROUP_TOO_SMALL")
        task_ids = {item.task_id for item in parsed}
        if len(task_ids) != 1:
            errors.append("TASK_ID_MISMATCH")
        fingerprints = {item.initial_state_fingerprint for item in parsed}
        if len(fingerprints) != 1:
            errors.append("INITIAL_STATE_MISMATCH")
        snapshot_versions = {item.snapshot_version for item in parsed}
        if len(snapshot_versions) != 1:
            errors.append("SNAPSHOT_VERSION_MISMATCH")
        environment_versions = {item.environment_version for item in parsed}
        if len(environment_versions) != 1:
            errors.append("ENVIRONMENT_VERSION_MISMATCH")
        trajectory_ids = [item.episode.trajectory_id for item in parsed]
        if len(trajectory_ids) != len(set(trajectory_ids)):
            errors.append("TRAJECTORY_ID_DUPLICATE")

        rewards = [item.reward.episode_reward for item in parsed]
        mean_reward = fmean(rewards) if rewards else 0.0
        reward_std = pstdev(rewards) if len(rewards) > 1 else 0.0
        successes = [
            item.reward.gate_status == "passed"
            and item.reward.components.task > 0
            and item.reward.components.constraint > 0
            for item in parsed
        ]
        success_rate = sum(successes) / len(successes) if successes else 0.0
        zero_variance = reward_std <= self.config.zero_variance_epsilon

        if errors:
            band: Literal["too_easy", "learnable", "too_hard", "invalid"] = "invalid"
            route: Literal["grpo_update", "evaluation", "sft_repair", "reject"] = "reject"
        elif success_rate > self.config.preferred_success_rate_max:
            band = "too_easy"
            route = "evaluation"
        elif success_rate < self.config.preferred_success_rate_min:
            band = "too_hard"
            route = "sft_repair"
        else:
            band = "learnable"
            route = "grpo_update"

        if zero_variance and not errors:
            if success_rate == 1.0:
                band = "too_easy"
                route = "evaluation"
            else:
                band = "too_hard"
                route = "sft_repair"

        eligible = not errors and not zero_variance and route == "grpo_update"
        advantages = [
            RolloutAdvantage(
                trajectory_id=item.episode.trajectory_id,
                reward=item.reward.episode_reward,
                standardized_advantage=(
                    round((item.reward.episode_reward - mean_reward) / reward_std, 8)
                    if reward_std > self.config.zero_variance_epsilon
                    else 0.0
                ),
            )
            for item in parsed
        ]
        return GRPOGroupDecision(
            group_id=group_id,
            task_id=next(iter(task_ids), "unknown"),
            initial_state_fingerprint=next(iter(fingerprints), "unknown"),
            group_size=len(parsed),
            mean_reward=round(mean_reward, 8),
            reward_std=round(reward_std, 8),
            success_rate=round(success_rate, 8),
            zero_variance=zero_variance,
            eligible_for_update=eligible,
            curriculum_band=band,
            route=route,
            rejection_codes=sorted(errors),
            advantages=advantages,
        )
