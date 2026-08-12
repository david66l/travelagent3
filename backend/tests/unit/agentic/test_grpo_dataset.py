"""Tests for auditable GRPO-B0 action dataset export."""

from agentic.environment import EnvironmentRollout
from agentic.grpo import GRPOGroupAuditor, GRPOGroupConfig
from agentic.grpo_dataset import GRPODatasetBuilder, GRPOGroupCandidate
from agentic.loop import PolicyAction, PolicyContext
from agentic.reward import EpisodeReward, RewardComponents, TurnReward
from agentic.trajectory import AgentEpisode, TrajectoryStep


def _rollout(index: int, value: float) -> EnvironmentRollout:
    trajectory_id = f"trajectory-{index}"
    context = PolicyContext(
        trajectory_id=trajectory_id,
        goal_version=1,
        plan_version=1,
        original_request="Plan Shanghai",
        current_subtask={"task_id": "weather"},
        hard_constraints={"destination": "Shanghai"},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        failure_summary=[],
        remaining_tasks=1,
        remaining_steps=4,
        allowed_actions=["get_weather"],
    )
    step = TrajectoryStep(
        step_index=0,
        task_id="weather",
        context=context,
        action=PolicyAction(action="get_weather"),
        observations=[],
        verification={},
        state_before_hash="before",
        state_after_hash="after",
    )
    episode = AgentEpisode(
        trajectory_id=trajectory_id,
        environment_version="env-v1",
        validator_version="validator-v1",
        policy_name="policy",
        policy_version="v1",
        initial_state={},
        steps=[step],
        final_state={},
        status="failed",
        termination_reason="partial_finish",
        content_hash=f"hash-{index}",
    )
    positive = value > 0
    return EnvironmentRollout(
        task_id="task-1",
        seed=42,
        initial_state_fingerprint="same",
        environment_version="env-v1",
        snapshot_version="snapshot-v1",
        episode=episode,
        reward=EpisodeReward(
            trajectory_id=trajectory_id,
            reward_config_version="reward-v1",
            gate_status="passed" if positive else "task_failed",
            components=RewardComponents(
                task=1 if positive else -1,
                constraint=1 if positive else -1,
                format=1,
                tool=1,
                grounding=1,
                efficiency=0,
            ),
            terminal_reward=value,
            process_reward=0,
            quality_reward=0,
            episode_reward=value,
            turn_rewards=[
                TurnReward(
                    step_index=0,
                    action="get_weather",
                    format=1,
                    tool=1 if positive else -0.5,
                    grounding=1,
                    efficiency=0,
                    information_gain=positive,
                )
            ],
        ),
    )


def _builder() -> GRPODatasetBuilder:
    auditor = GRPOGroupAuditor(GRPOGroupConfig(minimum_group_size=4))
    return GRPODatasetBuilder(auditor)


def test_only_eligible_group_exports_action_examples_with_group_advantage():
    eligible = GRPOGroupCandidate(
        group_id="mixed",
        rollouts=[
            _rollout(0, 0.8),
            _rollout(1, 0.4),
            _rollout(2, -0.3),
            _rollout(3, -0.6),
        ],
    )
    zero_variance = GRPOGroupCandidate(
        group_id="easy",
        rollouts=[_rollout(index + 10, 0.8) for index in range(4)],
    )

    result = _builder().build([eligible, zero_variance])

    assert result.manifest.accepted_groups == 1
    assert result.manifest.rejected_groups == 1
    assert result.manifest.exported_examples == 4
    assert {item.group_id for item in result.examples} == {"mixed"}
    assert all(item.credit_mode == "trajectory_b0" for item in result.examples)
    assert abs(sum(item.trajectory_advantage for item in result.examples)) < 1e-7
    assert result.manifest.routes == {"evaluation": 1, "grpo_update": 1}
    first = result.examples[0]
    assert first.completion[0]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert first.tools[0]["function"]["name"] == "get_weather"


def test_export_writes_training_rows_decisions_and_manifest(tmp_path):
    candidate = GRPOGroupCandidate(
        group_id="mixed",
        rollouts=[
            _rollout(0, 0.8),
            _rollout(1, 0.4),
            _rollout(2, -0.3),
            _rollout(3, -0.6),
        ],
    )
    result = _builder().build([candidate])

    _builder().export(result, tmp_path)

    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "group_decisions.jsonl").exists()
    assert (tmp_path / "manifest.json").exists()
