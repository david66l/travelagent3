from copy import deepcopy

import pytest

from agentic.cascade_distillation import (
    CascadeTeacherCandidate,
    TeacherProvenance,
    contribution_summary,
    select_cascade_group,
)
from agentic.corpus_generation import CurriculumTeacherPolicy, build_curriculum_case
from agentic.distillation import build_teacher_candidate
from agentic.environment import TravelAgentEnvironment
from agentic.policy import ControllerFirstPolicy


async def _candidate(*, teacher: str, tier: str, family: str, sample: int):
    task, snapshot = build_curriculum_case(0)
    rollout = await TravelAgentEnvironment(task, snapshot).rollout(
        ControllerFirstPolicy(CurriculumTeacherPolicy())
    )
    candidate = build_teacher_candidate(
        rollout,
        family=family,
        sample_index=sample,
    )
    # Arbitration tests isolate cascade behavior from curriculum reward changes.
    candidate.score.successful = True
    candidate.score.gate_status = "passed"
    candidate.score.hard_pass = True
    return CascadeTeacherCandidate(
        provenance=TeacherProvenance(
            teacher_id=teacher,
            model=teacher,
            checkpoint=f"checkpoint/{teacher}",
            tier=tier,
            run_id=f"run-{teacher}",
        ),
        candidate=candidate,
    )


@pytest.mark.asyncio
async def test_easy_group_prefers_verified_student_teacher():
    student = await _candidate(
        teacher="qwen3-4b", tier="student_teacher", family="search", sample=0
    )
    complex_teacher = await _candidate(
        teacher="qwen3-8b", tier="complex_teacher", family="search", sample=0
    )

    selected = select_cascade_group([student, complex_teacher])

    assert selected.difficulty == "easy"
    assert selected.chosen.provenance.teacher_id == "qwen3-4b"
    assert selected.arbitration_reason == "EASY_VERIFIED_STUDENT_TEACHER"


@pytest.mark.asyncio
async def test_complex_family_prefers_verified_complex_teacher():
    student = await _candidate(
        teacher="qwen3-4b", tier="student_teacher", family="tradeoff", sample=0
    )
    complex_teacher = await _candidate(
        teacher="qwen3-8b", tier="complex_teacher", family="tradeoff", sample=0
    )

    selected = select_cascade_group([student, complex_teacher])

    assert selected.difficulty == "hard"
    assert selected.chosen.provenance.teacher_id == "qwen3-8b"
    assert selected.arbitration_reason == "COMPLEX_FAMILY_COMPLEX_TEACHER"


@pytest.mark.asyncio
async def test_verifier_failure_escalates_and_contribution_is_auditable():
    student = await _candidate(
        teacher="qwen3-4b", tier="student_teacher", family="search", sample=0
    )
    failed_student = deepcopy(student)
    failed_student.candidate.rollout.episode.trajectory_id += "-failed"
    failed_student.candidate.score.trajectory_id += "-failed"
    failed_student.candidate.score.successful = False
    failed_student.candidate.score.gate_status = "failed"
    complex_teacher = await _candidate(
        teacher="qwen3-8b", tier="complex_teacher", family="search", sample=0
    )

    selected = select_cascade_group([failed_student, complex_teacher])
    summary = contribution_summary([failed_student, complex_teacher], [selected])

    assert selected.chosen.provenance.teacher_id == "qwen3-8b"
    assert selected.arbitration_reason == "STUDENT_TEACHER_FAILED_ESCALATED"
    assert {row.teacher_id: row.chosen for row in summary} == {
        "qwen3-4b": 0,
        "qwen3-8b": 1,
    }


@pytest.mark.asyncio
async def test_rejects_mixed_initial_states():
    first = await _candidate(teacher="qwen3-4b", tier="student_teacher", family="search", sample=0)
    second = await _candidate(teacher="qwen3-8b", tier="complex_teacher", family="search", sample=0)
    second.candidate.rollout.initial_state_fingerprint = "different"

    with pytest.raises(ValueError, match="immutable initial state"):
        select_cascade_group([first, second])
