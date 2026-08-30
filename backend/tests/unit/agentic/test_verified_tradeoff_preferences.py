import pytest

from agentic.corpus_generation import CurriculumTeacherPolicy, build_curriculum_case
from agentic.distillation import build_teacher_candidate, select_teacher_group
from agentic.environment import TravelAgentEnvironment
from agentic.grpo_training import GRPOCorpusRow
from agentic.policy import ControllerFirstPolicy
from scripts.build_tradeoff_decision_corpus import derive_tradeoff_decision
from scripts.build_verified_tradeoff_preferences import (
    FakeContinuationPolicy,
    _valid_teacher_tradeoff,
)


@pytest.mark.asyncio
async def test_fake_continuation_is_verifier_failed_against_real_tradeoff():
    task, snapshot = build_curriculum_case(8)
    row = derive_tradeoff_decision(GRPOCorpusRow(task=task, snapshot=snapshot))

    chosen_rollout = await TravelAgentEnvironment(row.task, row.snapshot).rollout(
        ControllerFirstPolicy(CurriculumTeacherPolicy())
    )
    rejected_rollout = await TravelAgentEnvironment(row.task, row.snapshot).rollout(
        ControllerFirstPolicy(FakeContinuationPolicy())
    )
    chosen = build_teacher_candidate(chosen_rollout, family="tradeoff", sample_index=0)
    rejected = build_teacher_candidate(rejected_rollout, family="tradeoff", sample_index=1)

    assert _valid_teacher_tradeoff(chosen) is True
    assert chosen.score.successful is True
    assert rejected.score.successful is False
    assert rejected.rollout.reward.audit_metrics["capability_termination_mismatch"] is True
    selection = select_teacher_group([chosen, rejected])
    assert len(selection.preference_pairs) == 1
    assert "VERIFIER_SUCCESS_OVER_FAILURE" in selection.preference_pairs[0].reason_codes
