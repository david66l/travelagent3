"""Tests for hierarchical Agentic RL rewards and anti-hacking gates."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentic.loop import PolicyAction, PolicyContext
from agentic.observations import ObservationEnvelope
from agentic.reward import HierarchicalRewardEngine, RewardConfig, RewardSafetySignals
from agentic.trajectory import AgentEpisode, TrajectoryStep, episode_content_hash


def _context(trajectory_id: str, task_id: str, action: str) -> PolicyContext:
    return PolicyContext(
        trajectory_id=trajectory_id,
        goal_version=1,
        plan_version=1,
        original_request="Plan Shanghai",
        current_subtask={"task_id": task_id},
        hard_constraints={"destination": "Shanghai", "travel_days": 1},
        soft_preferences={},
        relevant_fact_refs=[],
        relevant_artifact_refs=[],
        relevant_facts=[],
        relevant_artifacts=[],
        failure_summary=[],
        remaining_tasks=1,
        remaining_steps=8,
        allowed_actions=[action],
    )


def _episode(*, hard_pass: bool = True, termination: str = "awaiting_user") -> AgentEpisode:
    trajectory_id = "reward-trajectory"
    actions = ["solve_itinerary", "validate_itinerary", "compose_draft"]
    steps = []
    for index, action in enumerate(actions):
        is_tool = action in {"solve_itinerary", "validate_itinerary"}
        steps.append(
            TrajectoryStep(
                step_index=index,
                task_id=action,
                context=_context(trajectory_id, action, action),
                action=PolicyAction(action_id=f"call-{index}", action=action),
                observations=(
                    [
                        ObservationEnvelope(
                            ok=True,
                            tool=action,
                            data={"ok": True},
                            source="built_in",
                            confidence=1,
                            tool_call_id=f"call-{index}",
                        )
                    ]
                    if is_tool
                    else []
                ),
                verification={"passed": True},
                state_before_hash=f"before-{index}",
                state_after_hash=f"after-{index}",
            )
        )
    if termination == "validated_finish":
        steps.append(
            TrajectoryStep(
                step_index=3,
                task_id="await_confirmation",
                context=_context(trajectory_id, "await_confirmation", "finish"),
                action=PolicyAction(action="finish"),
                observations=[],
                verification={},
                state_before_hash="before-finish",
                state_after_hash="after-finish",
            )
        )
    episode = AgentEpisode(
        trajectory_id=trajectory_id,
        environment_version="env-v1",
        validator_version="validator-v1",
        policy_name="policy",
        policy_version="v1",
        initial_state={"goal": {"missing_information": []}},
        steps=steps,
        final_state={
            "goal": {"missing_information": []},
            "task_graph": {
                "tasks": [
                    {
                        "required": True,
                        "status": "succeeded",
                        "allowed_actions": [action],
                    }
                    for action in actions
                ]
            },
            "artifacts": {
                "solver": {"artifact_type": "solver_result", "payload": {"days": [{}]}},
                "validation": {
                    "artifact_type": "validation_report",
                    "payload": {
                        "hard_pass": hard_pass,
                        "hard_violations": ([] if hard_pass else [{"code": "BUDGET_EXCEEDED"}]),
                        "metrics": {"budget_error_rate": 0 if hard_pass else 0.2},
                    },
                },
            },
        },
        status="finished" if termination == "validated_finish" else "interrupted",
        termination_reason=termination,
        completed_at=datetime.now(UTC),
    )
    episode.content_hash = episode_content_hash(episode)
    return episode


def _rehash(episode: AgentEpisode) -> AgentEpisode:
    episode.content_hash = episode_content_hash(episode)
    return episode


def test_validated_episode_gets_positive_outcome_first_reward():
    reward = HierarchicalRewardEngine().score(_episode(), quality_score=1.0)

    assert reward.gate_status == "passed"
    assert reward.components.task == 1
    assert reward.components.constraint == 1
    assert reward.episode_reward > 0.7
    assert reward.quality_reward == 0
    assert reward.audit_metrics["quality_drives_training"] is False
    assert len(reward.turn_rewards) == 3


def test_security_or_forgery_cannot_be_offset_by_other_rewards():
    reward = HierarchicalRewardEngine().score(
        _episode(),
        safety=RewardSafetySignals(security_violation=True),
        quality_score=1.0,
    )

    assert reward.gate_status == "unsafe"
    assert reward.episode_reward == -1
    assert "SECURITY_VIOLATION" in reward.gate_reasons


def test_finish_with_failed_hard_constraints_is_capped_negative():
    reward = HierarchicalRewardEngine().score(
        _episode(hard_pass=False, termination="validated_finish")
    )

    assert reward.gate_status == "hard_constraint_failed"
    assert reward.episode_reward <= -0.25
    assert reward.components.constraint < 0


def test_fast_failure_does_not_receive_efficiency_reward():
    episode = _episode(termination="partial_finish")
    reward = HierarchicalRewardEngine().score(episode)

    assert reward.components.task == -1
    assert reward.components.efficiency == 0
    assert reward.episode_reward < 0


def test_duplicate_successful_tool_call_is_penalized_without_double_gain():
    episode = _episode()
    duplicate = episode.steps[0].model_copy(deep=True)
    duplicate.step_index = 1
    episode.steps.insert(1, duplicate)
    for index, step in enumerate(episode.steps):
        step.step_index = index
    _rehash(episode)

    reward = HierarchicalRewardEngine().score(episode)

    assert reward.audit_metrics["duplicate_calls"] == 1
    assert reward.turn_rewards[1].efficiency == -1
    assert reward.turn_rewards[1].tool < reward.turn_rewards[0].tool


def test_protected_validator_payload_is_treated_as_fact_forgery():
    episode = _episode()
    episode.steps[1].action.arguments = {"itinerary": [], "constraints": {}}
    _rehash(episode)

    reward = HierarchicalRewardEngine().score(episode)

    assert reward.gate_status == "unsafe"
    assert reward.episode_reward == -1
    assert "FORGED_FACT" in reward.gate_reasons


def test_reward_config_enforces_terminal_dominance_and_process_cap():
    with pytest.raises(ValidationError, match="terminal reward"):
        RewardConfig(task_weight=0.2, constraint_weight=0.2)
    with pytest.raises(ValidationError, match="process weights"):
        RewardConfig(format_weight=0.1, tool_weight=0.1, grounding_weight=0.1)
