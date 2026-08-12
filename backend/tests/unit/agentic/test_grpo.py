"""Tests for GRPO group validity, variance and curriculum routing."""

from copy import deepcopy

from agentic.environment import EnvironmentRollout
from agentic.grpo import GRPOGroupAuditor, GRPOGroupConfig
from agentic.reward import EpisodeReward, RewardComponents
from agentic.trajectory import AgentEpisode


def _rollout(index: int, reward: float, *, fingerprint: str = "same") -> EnvironmentRollout:
    status = "passed" if reward > 0 else "task_failed"
    episode = AgentEpisode(
        trajectory_id=f"trajectory-{index}",
        environment_version="env-v1",
        validator_version="validator-v1",
        policy_name="policy",
        policy_version="v1",
        initial_state={},
        final_state={},
        status="failed",
        termination_reason="partial_finish",
    )
    return EnvironmentRollout(
        task_id="task-1",
        seed=42,
        initial_state_fingerprint=fingerprint,
        environment_version="env-v1",
        snapshot_version="snapshot-v1",
        episode=episode,
        reward=EpisodeReward(
            trajectory_id=episode.trajectory_id,
            reward_config_version="v1",
            gate_status=status,
            components=RewardComponents(
                task=1 if reward > 0 else -1,
                constraint=1 if reward > 0 else -1,
                format=1,
                tool=1,
                grounding=1,
                efficiency=0,
            ),
            terminal_reward=reward,
            process_reward=0,
            quality_reward=0,
            episode_reward=reward,
            turn_rewards=[],
        ),
    )


def _auditor() -> GRPOGroupAuditor:
    return GRPOGroupAuditor(GRPOGroupConfig(minimum_group_size=4))


def test_mixed_success_group_is_eligible_and_advantages_are_centered():
    group = [_rollout(0, 0.8), _rollout(1, 0.4), _rollout(2, -0.3), _rollout(3, -0.6)]

    decision = _auditor().evaluate("group-1", group)

    assert decision.eligible_for_update is True
    assert decision.curriculum_band == "learnable"
    assert decision.route == "grpo_update"
    assert decision.success_rate == 0.5
    assert abs(sum(item.standardized_advantage for item in decision.advantages)) < 1e-7


def test_zero_variance_success_group_routes_to_evaluation():
    decision = _auditor().evaluate("group-easy", [_rollout(index, 0.8) for index in range(4)])

    assert decision.zero_variance is True
    assert decision.eligible_for_update is False
    assert decision.route == "evaluation"
    assert all(item.standardized_advantage == 0 for item in decision.advantages)


def test_zero_variance_failure_group_routes_to_sft_repair():
    decision = _auditor().evaluate("group-hard", [_rollout(index, -0.5) for index in range(4)])

    assert decision.zero_variance is True
    assert decision.curriculum_band == "too_hard"
    assert decision.route == "sft_repair"


def test_mismatched_initial_state_or_duplicate_trajectory_rejects_group():
    group = [_rollout(index, 0.8 - index * 0.2) for index in range(4)]
    group[3].initial_state_fingerprint = "different"
    group[2].episode.trajectory_id = group[1].episode.trajectory_id

    decision = _auditor().evaluate("group-invalid", group)

    assert decision.eligible_for_update is False
    assert decision.route == "reject"
    assert "INITIAL_STATE_MISMATCH" in decision.rejection_codes
    assert "TRAJECTORY_ID_DUPLICATE" in decision.rejection_codes


def test_input_rollouts_are_not_mutated():
    group = [_rollout(index, 0.8 - index * 0.2) for index in range(4)]
    original = deepcopy(group)

    _auditor().evaluate("group-pure", group)

    assert group == original
