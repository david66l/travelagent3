"""Tests for GRPO group validity, variance and curriculum routing."""

from copy import deepcopy

from agentic.environment import EnvironmentRollout
from agentic.grpo import (
    GRPOGroupAuditor,
    GRPOGroupConfig,
    model_aware_curriculum,
    policy_return_to_go_credit,
    policy_turn_credit_records,
    return_to_go_credit,
)
from agentic.loop import PolicyAction, PolicyContext
from agentic.reward import EpisodeReward, RewardComponents, TurnReward
from agentic.trajectory import AgentEpisode
from agentic.trajectory import TrajectoryStep


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


def test_verified_partial_credit_can_route_an_exact_failure_group_to_grpo():
    group = [_rollout(index, -0.5) for index in range(4)]
    partial_rewards = [-1.0, -0.333333, 0.333333, 0.333333]
    for rollout, reward in zip(group, partial_rewards, strict=True):
        rollout.reward.episode_reward = reward
        rollout.reward.terminal_reward = reward
        rollout.reward.audit_metrics["verified_partial_credit"] = True

    decision = _auditor().evaluate("group-partial", group)

    assert decision.success_rate == 0.0
    assert decision.reward_std > 0
    assert decision.verified_partial_credit is True
    assert decision.eligible_for_update is True
    assert decision.route == "grpo_update"


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


def test_model_aware_curriculum_prioritizes_learnable_groups():
    mixed = _auditor().evaluate(
        "mixed", [_rollout(0, 0.8), _rollout(1, 0.4), _rollout(2, -0.3), _rollout(3, -0.6)]
    )
    easy = _auditor().evaluate("easy", [_rollout(index + 4, 0.8) for index in range(4)])

    ranked = model_aware_curriculum([easy, mixed])

    assert ranked[0].task_id == mixed.task_id
    assert ranked[0].reason == "learnable_nonzero_variance"


def test_return_to_go_credit_keeps_turn_signals_distinct():
    reward = _rollout(0, 0.8).reward
    reward.turn_rewards = [
        TurnReward(
            step_index=0,
            action="search_pois",
            format=1,
            tool=-1,
            grounding=1,
            efficiency=-1,
        ),
        TurnReward(
            step_index=1,
            action="finish",
            format=1,
            tool=1,
            grounding=1,
            efficiency=1,
        ),
    ]

    credits = return_to_go_credit(reward, gamma=0.9)

    assert len(credits) == 2
    assert credits[1] > credits[0]


def test_policy_return_to_go_credit_excludes_controller_steps():
    rollout = _rollout(0, 0.8)
    context = PolicyContext(
        trajectory_id=rollout.episode.trajectory_id,
        goal_version=1,
        plan_version=1,
        original_request="plan",
        current_subtask={"task_id": "search"},
        hard_constraints={},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        failure_summary=[],
        remaining_tasks=1,
        remaining_steps=2,
        allowed_actions=["search_pois"],
    )
    rollout.episode.steps = [
        TrajectoryStep(
            step_index=0,
            task_id="controller",
            context=context,
            action=PolicyAction(action="get_weather", decision_source="controller"),
            state_before_hash="a",
            state_after_hash="b",
        ),
        TrajectoryStep(
            step_index=1,
            task_id="search",
            context=context,
            action=PolicyAction(action="search_pois", decision_source="policy"),
            state_before_hash="b",
            state_after_hash="c",
        ),
    ]
    rollout.reward.turn_rewards = [
        TurnReward(
            step_index=0,
            action="get_weather",
            format=-1,
            tool=-1,
            grounding=-1,
            efficiency=-1,
        ),
        TurnReward(
            step_index=1,
            action="search_pois",
            format=1,
            tool=1,
            grounding=1,
            efficiency=1,
        ),
    ]

    credits = policy_return_to_go_credit(rollout.reward, rollout.episode, gamma=1)

    assert credits == [0.9]


def test_policy_credit_discount_distance_counts_only_model_decisions():
    rollout = _rollout(0, 1.0)
    context = PolicyContext(
        trajectory_id=rollout.episode.trajectory_id,
        goal_version=1,
        plan_version=1,
        original_request="plan",
        current_subtask={"task_id": "search"},
        hard_constraints={},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        failure_summary=[],
        remaining_tasks=1,
        remaining_steps=4,
        allowed_actions=["search_pois", "finish"],
    )
    rollout.episode.steps = [
        TrajectoryStep(
            step_index=index,
            task_id="search",
            context=context,
            action=PolicyAction(action=action, decision_source=source),
            state_before_hash=f"before-{index}",
            state_after_hash=f"after-{index}",
        )
        for index, (action, source) in enumerate(
            [
                ("get_weather", "controller"),
                ("search_pois", "policy"),
                ("get_route_matrix", "controller"),
                ("finish", "policy"),
            ]
        )
    ]
    rollout.reward.episode_reward = 1.0
    rollout.reward.turn_rewards = [
        TurnReward(
            step_index=index,
            action=step.action.action,
            format=1,
            tool=1,
            grounding=1,
            efficiency=1,
        )
        for index, step in enumerate(rollout.episode.steps)
    ]

    records = policy_turn_credit_records(rollout.reward, rollout.episode, gamma=0.5)

    assert [item.terminal_distance for item in records] == [1, 0]
    assert [item.credit for item in records] == [0.75, 1.0]


def test_validity_gate_blocks_success_from_washing_invalid_actions():
    rollout = _rollout(0, 1.0)
    context = PolicyContext(
        trajectory_id=rollout.episode.trajectory_id,
        goal_version=1,
        plan_version=1,
        original_request="plan",
        current_subtask={"task_id": "search"},
        hard_constraints={},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        failure_summary=[],
        remaining_tasks=1,
        remaining_steps=3,
        allowed_actions=["search_pois"],
    )
    rollout.episode.steps = [
        TrajectoryStep(
            step_index=index,
            task_id="search",
            context=context,
            action=PolicyAction(action="search_pois", decision_source="policy"),
            state_before_hash=f"before-{index}",
            state_after_hash=f"after-{index}",
        )
        for index in range(3)
    ]
    rollout.reward.episode_reward = 1.0
    rollout.reward.turn_rewards = [
        TurnReward(
            step_index=0,
            action="search_pois",
            format=-1,
            tool=-1,
            grounding=-1,
            efficiency=-1,
            validity="invalid",
            future_credit_eligible=False,
        ),
        TurnReward(
            step_index=1,
            action="search_pois",
            format=1,
            tool=-0.5,
            grounding=1,
            efficiency=0,
            validity="external_failure",
            future_credit_eligible=False,
        ),
        TurnReward(
            step_index=2,
            action="search_pois",
            format=1,
            tool=1,
            grounding=1,
            efficiency=1,
        ),
    ]

    records = policy_turn_credit_records(rollout.reward, rollout.episode, gamma=0.9)

    assert [item.credit for item in records] == [-1.0, 0.0, 1.0]
    assert [item.future_credit_eligible for item in records] == [False, False, True]
