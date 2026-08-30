"""Verifier-first arbitration for multi-teacher policy distillation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agentic.distillation import (
    TeacherCandidateRecord,
    TeacherPreferencePair,
    build_preference_pair,
)


CASCADE_DISTILLATION_SCHEMA_VERSION = "cascade-distillation.v1"
TeacherTier = Literal["student_teacher", "complex_teacher"]

COMPLEX_FAMILIES = {
    "tradeoff",
    "abort",
    "necessary_abort",
    "long_context_replan",
}


class TeacherProvenance(BaseModel):
    """Identity required for every independently generated teacher rollout."""

    teacher_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    checkpoint: str = Field(min_length=1)
    tier: TeacherTier
    run_id: str = Field(min_length=1)


class CascadeTeacherCandidate(BaseModel):
    schema_version: str = CASCADE_DISTILLATION_SCHEMA_VERSION
    provenance: TeacherProvenance
    candidate: TeacherCandidateRecord


class CascadeTeacherContribution(BaseModel):
    teacher_id: str
    candidates: int = Field(ge=0)
    successful: int = Field(ge=0)
    chosen: int = Field(ge=0)


class CascadeSelection(BaseModel):
    schema_version: str = CASCADE_DISTILLATION_SCHEMA_VERSION
    task_id: str
    family: str
    difficulty: Literal["easy", "hard"]
    arbitration_reason: str
    chosen: CascadeTeacherCandidate
    rejected: list[CascadeTeacherCandidate]
    preference_pairs: list[TeacherPreferencePair]
    successful_teacher_ids: list[str]
    teacher_action_agreement: bool

    @model_validator(mode="after")
    def validate_chosen(self) -> "CascadeSelection":
        if not self.chosen.candidate.score.successful:
            raise ValueError("cascade chosen candidate must pass the verifier")
        return self


def select_cascade_group(
    candidates: list[CascadeTeacherCandidate | dict[str, Any]],
    *,
    chosen_trajectory_ids: set[str] | None = None,
) -> CascadeSelection:
    """Select one verified behavior without treating a larger model as ground truth.

    Easy, internally consistent tasks prefer the 4B-style ``student_teacher``.
    Complex tasks, 4B failures and 4B action disagreement prefer a successful
    ``complex_teacher``. Verifier outcome always precedes teacher identity.
    """

    parsed = [
        item if isinstance(item, CascadeTeacherCandidate) else CascadeTeacherCandidate(**item)
        for item in candidates
    ]
    if len(parsed) < 2:
        raise ValueError("cascade selection requires at least two candidates")
    task_ids = {item.candidate.task_id for item in parsed}
    families = {item.candidate.family for item in parsed}
    fingerprints = {item.candidate.rollout.initial_state_fingerprint for item in parsed}
    trajectories = {item.candidate.rollout.episode.trajectory_id for item in parsed}
    if len(task_ids) != 1:
        raise ValueError("cascade candidates must share one task_id")
    if len(families) != 1:
        raise ValueError("cascade candidates must share one family")
    if len(fingerprints) != 1:
        raise ValueError("cascade candidates must share one immutable initial state")
    if len(trajectories) != len(parsed):
        raise ValueError("cascade trajectory IDs must be unique")

    successful = [
        item
        for item in parsed
        if item.candidate.score.successful
        and (
            chosen_trajectory_ids is None
            or item.candidate.score.trajectory_id in chosen_trajectory_ids
        )
    ]
    if not successful:
        raise ValueError("cascade group has no verifier-approved trainable successful candidate")

    student_candidates = [item for item in successful if item.provenance.tier == "student_teacher"]
    successful_student = [item for item in student_candidates if item.candidate.score.successful]
    complex_success = [item for item in successful if item.provenance.tier == "complex_teacher"]
    student_signatures = {
        signature
        for item in successful_student
        if (signature := first_policy_action_signature(item.candidate)) is not None
    }
    family = next(iter(families))
    student_disagrees = len(student_signatures) > 1
    difficult = family in COMPLEX_FAMILIES or not successful_student or student_disagrees

    if difficult and complex_success:
        eligible = complex_success
        reason = (
            "COMPLEX_FAMILY_COMPLEX_TEACHER"
            if family in COMPLEX_FAMILIES
            else "STUDENT_TEACHER_DISAGREEMENT_ESCALATED"
            if student_disagrees
            else "STUDENT_TEACHER_FAILED_ESCALATED"
        )
    elif successful_student:
        eligible = successful_student
        reason = (
            "COMPLEX_TEACHER_UNAVAILABLE_VERIFIED_STUDENT_TEACHER"
            if difficult
            else "EASY_VERIFIED_STUDENT_TEACHER"
        )
    else:
        eligible = successful
        reason = "VERIFIED_FALLBACK_TEACHER"

    chosen = min(eligible, key=_cascade_rank)
    rejected = sorted((item for item in parsed if item is not chosen), key=_cascade_rank)
    preference_by_id: dict[str, TeacherPreferencePair] = {}
    for item in rejected:
        pair = build_preference_pair(chosen.candidate, item.candidate)
        if pair is None or "VERIFIER_SUCCESS_OVER_FAILURE" not in pair.reason_codes:
            continue
        preference_by_id.setdefault(pair.pair_id, pair)

    successful_signatures = {
        signature
        for item in successful
        if (signature := first_policy_action_signature(item.candidate)) is not None
    }
    return CascadeSelection(
        task_id=chosen.candidate.task_id,
        family=family,
        difficulty="hard" if difficult else "easy",
        arbitration_reason=reason,
        chosen=chosen,
        rejected=rejected,
        preference_pairs=list(preference_by_id.values()),
        successful_teacher_ids=sorted({item.provenance.teacher_id for item in successful}),
        teacher_action_agreement=len(successful_signatures) <= 1,
    )


def first_policy_action_signature(candidate: TeacherCandidateRecord) -> str | None:
    """Return a stable action-level signature without exposing hidden gold labels."""

    for step in candidate.rollout.episode.steps:
        if step.action.decision_source == "controller":
            continue
        arguments = step.action.arguments
        normalized = _canonical_value(arguments)
        return f"{step.action.action}:{normalized}"
    return None


def contribution_summary(
    candidates: list[CascadeTeacherCandidate],
    selections: list[CascadeSelection],
) -> list[CascadeTeacherContribution]:
    candidate_counts = Counter(item.provenance.teacher_id for item in candidates)
    success_counts = Counter(
        item.provenance.teacher_id for item in candidates if item.candidate.score.successful
    )
    chosen_counts = Counter(item.chosen.provenance.teacher_id for item in selections)
    return [
        CascadeTeacherContribution(
            teacher_id=teacher_id,
            candidates=candidate_counts[teacher_id],
            successful=success_counts[teacher_id],
            chosen=chosen_counts[teacher_id],
        )
        for teacher_id in sorted(candidate_counts)
    ]


def _cascade_rank(item: CascadeTeacherCandidate) -> tuple[Any, ...]:
    score = item.candidate.score
    return (
        not score.successful,
        not score.hard_pass,
        score.invalid_model_steps,
        score.duplicate_calls,
        score.policy_steps,
        score.tool_calls,
        score.completion_tokens,
        -score.episode_reward,
        score.trajectory_id,
    )


def _canonical_value(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CASCADE_DISTILLATION_SCHEMA_VERSION",
    "CascadeSelection",
    "CascadeTeacherCandidate",
    "CascadeTeacherContribution",
    "TeacherProvenance",
    "contribution_summary",
    "first_policy_action_signature",
    "select_cascade_group",
]
