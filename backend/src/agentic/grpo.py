"""GRPO-B0 group audit, relative advantages and curriculum routing."""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import Literal

from pydantic import BaseModel, Field

from agentic.environment import EnvironmentRollout
from agentic.reward import EpisodeReward, TurnReward
from agentic.trajectory import AgentEpisode


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


class CurriculumSamplingDecision(BaseModel):
    task_id: str
    priority: float = Field(ge=0)
    reason: str


def model_aware_curriculum(
    decisions: list[GRPOGroupDecision | dict],
) -> list[CurriculumSamplingDecision]:
    """Rank tasks by learnability, uncertainty and useful group variance.

    Learnable mixed-outcome tasks receive the highest priority. All-success
    tasks remain evaluation anchors; all-failure tasks return to SFT repair and
    never consume GRPO updates merely because they look difficult.
    """
    parsed = [
        item if isinstance(item, GRPOGroupDecision) else GRPOGroupDecision(**item)
        for item in decisions
    ]
    ranked: list[CurriculumSamplingDecision] = []
    for item in parsed:
        uncertainty = 1.0 - abs(0.5 - item.success_rate) * 2
        if item.route == "grpo_update":
            priority = 1.0 + uncertainty + min(1.0, item.reward_std)
            reason = "learnable_nonzero_variance"
        elif item.route == "sft_repair":
            priority = 0.2
            reason = "route_to_sft_repair"
        elif item.route == "evaluation":
            priority = 0.1
            reason = "retain_as_evaluation_anchor"
        else:
            priority = 0.0
            reason = "invalid_group"
        ranked.append(
            CurriculumSamplingDecision(
                task_id=item.task_id,
                priority=round(priority, 6),
                reason=reason,
            )
        )
    return sorted(ranked, key=lambda item: (-item.priority, item.task_id))


def return_to_go_credit(
    reward: EpisodeReward | dict,
    *,
    gamma: float = 1.0,
) -> list[float]:
    """R1 research baseline: combine local signals with discounted terminal credit.

    This is exported as an explicit comparison signal; it is not mislabeled as
    the trajectory-level B0 objective used by the stock TRL trainer.
    """
    parsed = reward if isinstance(reward, EpisodeReward) else EpisodeReward(**reward)
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    total_steps = len(parsed.turn_rewards)
    credits: list[float] = []
    for index, turn in enumerate(parsed.turn_rewards):
        local = (turn.format + turn.tool + turn.grounding + turn.efficiency) / 4
        distance = total_steps - index - 1
        value = 0.5 * local + 0.5 * (gamma**distance) * parsed.episode_reward
        credits.append(round(max(-1.0, min(1.0, value)), 6))
    return credits


def policy_return_to_go_credit(
    reward: EpisodeReward | dict,
    episode: AgentEpisode | dict,
    *,
    gamma: float = 1.0,
) -> list[float]:
    """Return validity-gated credits over model decisions only.

    Discount distance is measured in policy decisions, never controller or
    tool-observation steps. Invalid model actions are fixed negative and legal
    calls that encounter an environment failure are neutral. Only valid turns
    may inherit verified terminal outcome credit.
    """
    return [item.credit for item in policy_turn_credit_records(reward, episode, gamma=gamma)]


class PolicyTurnCredit(BaseModel):
    step_index: int = Field(ge=0)
    validity: Literal["invalid", "external_failure", "valid"]
    future_credit_eligible: bool
    local_reward: float = Field(ge=-1, le=1)
    terminal_distance: int = Field(ge=0)
    credit: float = Field(ge=-1, le=1)


def policy_turn_credit_records(
    reward: EpisodeReward | dict,
    episode: AgentEpisode | dict,
    *,
    gamma: float = 1.0,
) -> list[PolicyTurnCredit]:
    """Build auditable R1-v2 credit facts for model-owned turns."""
    parsed_reward = reward if isinstance(reward, EpisodeReward) else EpisodeReward(**reward)
    parsed_episode = episode if isinstance(episode, AgentEpisode) else AgentEpisode(**episode)
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if len(parsed_reward.turn_rewards) != len(parsed_episode.steps):
        raise ValueError("turn rewards must align with episode steps")
    policy_turns = [
        turn
        for turn, step in zip(parsed_reward.turn_rewards, parsed_episode.steps, strict=True)
        if step.action.decision_source != "controller"
    ]
    records: list[PolicyTurnCredit] = []
    total_policy_turns = len(policy_turns)
    for policy_index, turn in enumerate(policy_turns):
        local = _local_turn_reward(turn)
        validity = _effective_turn_validity(turn)
        distance = total_policy_turns - policy_index - 1
        if validity == "invalid":
            value = -1.0
        elif validity == "external_failure":
            value = 0.0
        else:
            value = 0.5 * local + 0.5 * (gamma**distance) * parsed_reward.episode_reward
        records.append(
            PolicyTurnCredit(
                step_index=turn.step_index,
                validity=validity,
                future_credit_eligible=validity == "valid",
                local_reward=round(local, 6),
                terminal_distance=distance,
                credit=round(max(-1.0, min(1.0, value)), 6),
            )
        )
    return records


def _local_turn_reward(turn: TurnReward) -> float:
    return (turn.format + turn.tool + turn.grounding + turn.efficiency) / 4


def _effective_turn_validity(
    turn: TurnReward,
) -> Literal["invalid", "external_failure", "valid"]:
    # Old persisted reports predate explicit validity fields. Reconstruct the
    # conservative gate from their process facts so replay remains safe.
    if (
        turn.validity == "invalid"
        or turn.format < 0
        or turn.grounding < 0
        or "INVALID_MODEL_ACTION" in turn.signals
    ):
        return "invalid"
    if (
        turn.validity == "external_failure"
        or "EXTERNAL_FAILURE" in turn.signals
        or "TOOL_FAILED" in turn.signals
    ):
        return "external_failure"
    return "valid"
