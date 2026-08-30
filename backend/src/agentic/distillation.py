"""Verifier-guided teacher candidate ranking and preference extraction."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field, model_validator

from agentic.environment import EnvironmentRollout
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT, policy_prompt_payload
from agentic.policy_actions import policy_action_schemas


DISTILLATION_SCHEMA_VERSION = "teacher-distillation.v1"


class TeacherCandidateScore(BaseModel):
    trajectory_id: str
    successful: bool
    gate_status: str
    episode_reward: float
    hard_pass: bool
    episode_steps: int = Field(ge=0)
    policy_steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    duplicate_calls: int = Field(ge=0)
    invalid_model_steps: int = Field(ge=0)
    request_latency_ms: float = Field(ge=0)


class TeacherCandidateRecord(BaseModel):
    schema_version: str = DISTILLATION_SCHEMA_VERSION
    task_id: str
    family: str
    sample_index: int = Field(ge=0)
    score: TeacherCandidateScore
    rollout: EnvironmentRollout

    @model_validator(mode="after")
    def validate_identity(self) -> "TeacherCandidateRecord":
        if self.task_id != self.rollout.task_id:
            raise ValueError("candidate task_id must match rollout task_id")
        if self.score.trajectory_id != self.rollout.episode.trajectory_id:
            raise ValueError("candidate score must match rollout trajectory_id")
        return self


class TeacherPreferencePair(BaseModel):
    schema_version: str = "teacher-preference-pair.v1"
    pair_id: str
    task_id: str
    family: str
    context_hash: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    chosen: dict[str, Any]
    rejected: dict[str, Any]
    chosen_trajectory_id: str
    rejected_trajectory_id: str
    reason_codes: list[str]
    reward_margin: float


class TeacherGroupSelection(BaseModel):
    schema_version: str = DISTILLATION_SCHEMA_VERSION
    task_id: str
    chosen: TeacherCandidateRecord
    rejected: list[TeacherCandidateRecord]
    preference_pairs: list[TeacherPreferencePair]


def score_teacher_rollout(rollout: EnvironmentRollout) -> TeacherCandidateScore:
    """Reduce one verified rollout to stable outcome and efficiency evidence."""
    reward = rollout.reward
    policy_steps = [
        step for step in rollout.episode.steps if step.action.decision_source != "controller"
    ]
    completion_tokens = sum(
        (
            step.action.inference_metrics.completion_tokens
            if step.action.inference_metrics is not None
            else step.action.token_usage
        )
        for step in policy_steps
    )
    request_latency_ms = sum(
        step.action.inference_metrics.request_latency_ms
        for step in policy_steps
        if step.action.inference_metrics is not None
    )
    hard_pass = bool(reward.audit_metrics.get("hard_pass"))
    successful = bool(
        reward.gate_status == "passed"
        and reward.components.task > 0
        and reward.components.constraint > 0
    )
    return TeacherCandidateScore(
        trajectory_id=rollout.episode.trajectory_id,
        successful=successful,
        gate_status=reward.gate_status,
        episode_reward=reward.episode_reward,
        hard_pass=hard_pass,
        episode_steps=len(rollout.episode.steps),
        policy_steps=len(policy_steps),
        tool_calls=sum(rollout.tool_call_counts.values()),
        completion_tokens=completion_tokens,
        duplicate_calls=int(reward.audit_metrics.get("duplicate_calls") or 0),
        invalid_model_steps=int(reward.audit_metrics.get("invalid_model_steps") or 0),
        request_latency_ms=round(request_latency_ms, 3),
    )


def build_teacher_candidate(
    rollout: EnvironmentRollout,
    *,
    family: str,
    sample_index: int,
) -> TeacherCandidateRecord:
    return TeacherCandidateRecord(
        task_id=rollout.task_id,
        family=family,
        sample_index=sample_index,
        score=score_teacher_rollout(rollout),
        rollout=rollout,
    )


def select_teacher_group(
    candidates: list[TeacherCandidateRecord | dict[str, Any]],
) -> TeacherGroupSelection:
    """Choose the successful shortest candidate and retain auditable negatives."""
    parsed = [
        item if isinstance(item, TeacherCandidateRecord) else TeacherCandidateRecord(**item)
        for item in candidates
    ]
    if len(parsed) < 2:
        raise ValueError("teacher selection requires at least two candidates")
    task_ids = {item.task_id for item in parsed}
    fingerprints = {item.rollout.initial_state_fingerprint for item in parsed}
    trajectories = {item.rollout.episode.trajectory_id for item in parsed}
    if len(task_ids) != 1:
        raise ValueError("teacher candidates must share one task_id")
    if len(fingerprints) != 1:
        raise ValueError("teacher candidates must share one immutable initial state")
    if len(trajectories) != len(parsed):
        raise ValueError("teacher candidate trajectory IDs must be unique")

    ordered = sorted(parsed, key=_candidate_rank)
    chosen = ordered[0]
    if not chosen.score.successful:
        raise ValueError("teacher group has no verifier-approved successful candidate")
    rejected = ordered[1:]
    pairs = [
        pair
        for candidate in rejected
        if (pair := build_preference_pair(chosen, candidate)) is not None
    ]
    return TeacherGroupSelection(
        task_id=chosen.task_id,
        chosen=chosen,
        rejected=rejected,
        preference_pairs=pairs,
    )


def build_preference_pair(
    chosen: TeacherCandidateRecord,
    rejected: TeacherCandidateRecord,
) -> TeacherPreferencePair | None:
    """Extract the first different action under an identical model-visible context."""
    if chosen.task_id != rejected.task_id:
        raise ValueError("preference candidates must share one task_id")
    reasons = _preference_reasons(chosen.score, rejected.score)
    if not reasons:
        return None

    rejected_steps: dict[str, Any] = {}
    for step in rejected.rollout.episode.steps:
        if step.action.decision_source == "controller":
            continue
        rejected_steps.setdefault(_context_hash(step), step)
    for chosen_step in chosen.rollout.episode.steps:
        if chosen_step.action.decision_source == "controller":
            continue
        context_hash = _context_hash(chosen_step)
        rejected_step = rejected_steps.get(context_hash)
        if rejected_step is None:
            continue
        chosen_response = _assistant_response(chosen_step)
        rejected_response = _assistant_response(rejected_step)
        if chosen_response == rejected_response:
            continue
        context_json = json.dumps(
            policy_prompt_payload(chosen_step.context),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        pair_payload = {
            "task_id": chosen.task_id,
            "context_hash": context_hash,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }
        return TeacherPreferencePair(
            pair_id="pref-" + _canonical_hash(pair_payload)[:20],
            task_id=chosen.task_id,
            family=chosen.family,
            context_hash=context_hash,
            messages=[
                {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": context_json},
            ],
            tools=policy_action_schemas(chosen_step.context.allowed_actions),
            chosen=chosen_response,
            rejected=rejected_response,
            chosen_trajectory_id=chosen.rollout.episode.trajectory_id,
            rejected_trajectory_id=rejected.rollout.episode.trajectory_id,
            reason_codes=reasons,
            reward_margin=round(
                chosen.score.episode_reward - rejected.score.episode_reward,
                6,
            ),
        )
    return None


def _candidate_rank(candidate: TeacherCandidateRecord) -> tuple[Any, ...]:
    score = candidate.score
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


def _preference_reasons(
    chosen: TeacherCandidateScore,
    rejected: TeacherCandidateScore,
) -> list[str]:
    reasons: list[str] = []
    if chosen.successful and not rejected.successful:
        reasons.append("VERIFIER_SUCCESS_OVER_FAILURE")
    if chosen.hard_pass and not rejected.hard_pass:
        reasons.append("HARD_CONSTRAINT_PASS_OVER_FAILURE")
    if chosen.invalid_model_steps < rejected.invalid_model_steps:
        reasons.append("FEWER_INVALID_MODEL_STEPS")
    if chosen.duplicate_calls < rejected.duplicate_calls:
        reasons.append("FEWER_DUPLICATE_CALLS")
    if chosen.policy_steps < rejected.policy_steps:
        reasons.append("FEWER_POLICY_STEPS")
    if chosen.tool_calls < rejected.tool_calls:
        reasons.append("FEWER_TOOL_CALLS")
    if chosen.completion_tokens < rejected.completion_tokens:
        reasons.append("FEWER_COMPLETION_TOKENS")
    if chosen.episode_reward > rejected.episode_reward:
        reasons.append("HIGHER_VERIFIED_REWARD")
    return reasons


def _context_hash(step: Any) -> str:
    payload = {
        "context": policy_prompt_payload(step.context),
        "tools": policy_action_schemas(step.context.allowed_actions),
    }
    return _canonical_hash(payload)


def _assistant_response(step: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": step.action.action,
                    "arguments": step.action.arguments,
                },
            }
        ],
    }


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "DISTILLATION_SCHEMA_VERSION",
    "TeacherCandidateRecord",
    "TeacherCandidateScore",
    "TeacherGroupSelection",
    "TeacherPreferencePair",
    "build_preference_pair",
    "build_teacher_candidate",
    "score_teacher_rollout",
    "select_teacher_group",
]
