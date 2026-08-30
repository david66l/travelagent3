"""Verifier-guided teacher selection and preference extraction tests."""

import pytest

from agentic.corpus_generation import CurriculumTeacherPolicy, build_curriculum_case
from agentic.distillation import (
    build_preference_pair,
    build_teacher_candidate,
    select_teacher_group,
)
from agentic.environment import TravelAgentEnvironment
from agentic.policy import ControllerFirstPolicy
from agentic.trajectory import episode_content_hash


async def _rollout():
    task, snapshot = build_curriculum_case(0)
    rollout = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(CurriculumTeacherPolicy())
    )
    return task, rollout


def _failed_variant(rollout):
    failed = rollout.model_copy(deep=True)
    failed.episode.trajectory_id = "rejected-trajectory"
    failed.reward.trajectory_id = "rejected-trajectory"
    for step in failed.episode.steps:
        step.context.trajectory_id = "rejected-trajectory"
    policy_step = next(
        step for step in failed.episode.steps if step.action.decision_source != "controller"
    )
    keywords = list(policy_step.action.arguments.get("keywords") or [])
    policy_step.action.arguments["keywords"] = keywords[:1]
    failed.reward.gate_status = "task_failed"
    failed.reward.components.task = -1
    failed.reward.components.constraint = -1
    failed.reward.episode_reward = -0.25
    failed.reward.audit_metrics["hard_pass"] = False
    failed.reward.audit_metrics["invalid_model_steps"] = 1
    failed.episode.content_hash = episode_content_hash(failed.episode)
    return failed


@pytest.mark.asyncio
async def test_teacher_selection_prefers_verified_success_and_builds_pair():
    task, successful = await _rollout()
    failed = _failed_variant(successful)
    chosen = build_teacher_candidate(
        successful,
        family="search",
        sample_index=0,
    )
    rejected = build_teacher_candidate(
        failed,
        family="search",
        sample_index=1,
    )

    result = select_teacher_group([rejected, chosen])

    assert result.task_id == task.task_id
    assert result.chosen.score.successful is True
    assert result.rejected[0].score.successful is False
    assert len(result.preference_pairs) == 1
    pair = result.preference_pairs[0]
    assert "VERIFIER_SUCCESS_OVER_FAILURE" in pair.reason_codes
    assert pair.chosen != pair.rejected
    assert pair.tools


@pytest.mark.asyncio
async def test_identical_action_has_no_trainable_preference_pair():
    _, successful = await _rollout()
    failed = _failed_variant(successful)
    successful_step = next(
        step for step in successful.episode.steps if step.action.decision_source != "controller"
    )
    failed_step = next(
        step for step in failed.episode.steps if step.action.decision_source != "controller"
    )
    failed_step.action = successful_step.action.model_copy(deep=True)
    failed.episode.content_hash = episode_content_hash(failed.episode)
    chosen = build_teacher_candidate(successful, family="search", sample_index=0)
    rejected = build_teacher_candidate(failed, family="search", sample_index=1)

    assert build_preference_pair(chosen, rejected) is None


@pytest.mark.asyncio
async def test_teacher_selection_rejects_mixed_tasks():
    _, first = await _rollout()
    task, snapshot = build_curriculum_case(1)
    second = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(CurriculumTeacherPolicy())
    )

    with pytest.raises(ValueError, match="task_id"):
        select_teacher_group(
            [
                build_teacher_candidate(first, family="search", sample_index=0),
                build_teacher_candidate(second, family="search", sample_index=1),
            ]
        )
