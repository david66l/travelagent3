"""Versioned, replayable and privacy-safe Agent Loop episode records."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from agentic.loop import AgentLoopEvent, AgentLoopResult, PolicyAction, PolicyContext
from agentic.observations import ObservationEnvelope
from agentic.state import AgentLedgerState


EPISODE_SCHEMA_VERSION = "agent-episode.v1"
_PHONE = re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")


def _now() -> datetime:
    return datetime.now(UTC)


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


class TrajectoryStep(BaseModel):
    step_index: int = Field(ge=0)
    task_id: str
    context: PolicyContext
    action: PolicyAction
    observations: list[ObservationEnvelope] = Field(default_factory=list)
    verification: dict[str, Any] = Field(default_factory=dict)
    state_before_hash: str
    state_after_hash: str
    started_at: datetime = Field(default_factory=_now)
    finished_at: datetime = Field(default_factory=_now)


class AgentEpisode(BaseModel):
    schema_version: str = EPISODE_SCHEMA_VERSION
    trajectory_id: str
    environment_version: str
    validator_version: str
    observation_schema_version: str = "observation.v1"
    policy_name: str
    policy_version: str
    initial_state: dict[str, Any]
    steps: list[TrajectoryStep] = Field(default_factory=list)
    events: list[AgentLoopEvent] = Field(default_factory=list)
    final_state: dict[str, Any] | None = None
    status: Literal["running", "finished", "interrupted", "failed"] = "running"
    termination_reason: str | None = None
    created_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None
    content_hash: str | None = None

    @model_validator(mode="after")
    def validate_step_sequence(self) -> AgentEpisode:
        indexes = [step.step_index for step in self.steps]
        if indexes != list(range(len(indexes))):
            raise ValueError("trajectory step indexes must be contiguous from zero")
        return self


def episode_content_hash(episode: AgentEpisode) -> str:
    """Return the canonical integrity hash used by finalized episodes."""
    return _canonical_hash(episode.model_dump(mode="json", exclude={"content_hash"}))


class EpisodeRecorder:
    """Append-only recorder with deterministic state hashes."""

    def __init__(
        self,
        initial_state: AgentLedgerState,
        *,
        environment_version: str,
        validator_version: str,
        policy_name: str,
        policy_version: str,
    ) -> None:
        redacted = redact_pii(initial_state.model_dump(mode="json"))
        self.episode = AgentEpisode(
            trajectory_id=initial_state.trajectory_id,
            environment_version=environment_version,
            validator_version=validator_version,
            policy_name=policy_name,
            policy_version=policy_version,
            initial_state=redacted,
        )

    def record_step(
        self,
        *,
        task_id: str,
        context: PolicyContext,
        action: PolicyAction,
        observations: list[ObservationEnvelope],
        verification: dict[str, Any],
        state_before: AgentLedgerState,
        state_after: AgentLedgerState,
    ) -> None:
        before = redact_pii(state_before.model_dump(mode="json"))
        after = redact_pii(state_after.model_dump(mode="json"))
        safe_context = PolicyContext(**redact_pii(context.model_dump(mode="json")))
        safe_action = PolicyAction(**redact_pii(action.model_dump(mode="json")))
        safe_observations = [
            ObservationEnvelope(**redact_pii(item.model_dump(mode="json"))) for item in observations
        ]
        self.episode.steps.append(
            TrajectoryStep(
                step_index=len(self.episode.steps),
                task_id=task_id,
                context=safe_context,
                action=safe_action,
                observations=safe_observations,
                verification=redact_pii(verification),
                state_before_hash=_canonical_hash(before),
                state_after_hash=_canonical_hash(after),
            )
        )

    def finalize(self, result: AgentLoopResult) -> AgentEpisode:
        self.episode.events = result.events
        self.episode.final_state = redact_pii(result.ledger.model_dump(mode="json"))
        self.episode.status = result.status
        self.episode.termination_reason = result.termination_reason
        self.episode.completed_at = _now()
        self.episode.content_hash = episode_content_hash(self.episode)
        return self.episode


def redact_pii(value: Any, *, field_name: str | None = None) -> Any:
    """Recursively redact direct identifiers before an episode is persisted."""
    sensitive_fields = {
        "user_id",
        "email",
        "phone",
        "phone_number",
        "id_card",
        "passport_number",
        "real_name",
        "full_name",
        "address",
    }
    if field_name and field_name.lower() in sensitive_fields:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {key: redact_pii(item, field_name=str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [redact_pii(item) for item in value]
    if isinstance(value, tuple):
        return [redact_pii(item) for item in value]
    if isinstance(value, str):
        value = _EMAIL.sub("[REDACTED_EMAIL]", value)
        value = _PHONE.sub("[REDACTED_PHONE]", value)
        return _ID_CARD.sub("[REDACTED_ID]", value)
    return value


class EpisodeReplayVerifier:
    """Verify record integrity before it is used for eval, SFT or RL."""

    def verify(self, episode: AgentEpisode | dict[str, Any]) -> list[str]:
        parsed = episode if isinstance(episode, AgentEpisode) else AgentEpisode(**episode)
        errors: list[str] = []
        if parsed.status != "running" and parsed.final_state is None:
            errors.append("FINAL_STATE_MISSING")
        for step in parsed.steps:
            if step.context.trajectory_id != parsed.trajectory_id:
                errors.append(f"TRAJECTORY_ID_MISMATCH:{step.step_index}")
            if step.context.current_subtask.get("task_id") != step.task_id:
                errors.append(f"TASK_ID_MISMATCH:{step.step_index}")
            if step.action.action not in step.context.allowed_actions:
                errors.append(f"ACTION_NOT_ALLOWED:{step.step_index}")
        if parsed.content_hash is not None:
            actual = episode_content_hash(parsed)
            if actual != parsed.content_hash:
                errors.append("CONTENT_HASH_MISMATCH")
        return errors
