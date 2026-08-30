"""Versioned snapshot environment for isolated Agentic RL rollouts."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic.action_executor import TravelActionExecutor
from agentic.guard import GuardContext, ToolGuard
from agentic.loop import AgentPolicy, BoundedAgentLoop
from agentic.observations import ObservationEnvelope
from agentic.reward import EpisodeReward, HierarchicalRewardEngine, RewardSafetySignals
from agentic.runtime import initialize_agent_ledger
from agentic.state import AgentLedgerState
from agentic.trajectory import AgentEpisode, EpisodeRecorder
from evaluation.validator import VALIDATOR_VERSION
from schemas import ToolResult


ENVIRONMENT_SCHEMA_VERSION = "travel-rl-environment.v1"
Difficulty = Literal["L0", "L1", "L2", "L3", "L4"]


class EnvironmentTask(BaseModel):
    task_id: str = Field(min_length=1)
    template_family: str = Field(min_length=1)
    difficulty: Difficulty
    seed: int
    user_request: str = Field(min_length=1)
    slots: dict[str, Any] = Field(default_factory=dict)
    profile: dict[str, Any] = Field(default_factory=dict)
    missing_slots: list[str] = Field(default_factory=list)
    feasibility_report: dict[str, Any] = Field(default_factory=dict)


class SnapshotToolResponse(BaseModel):
    data: Any = None
    data_source: Literal["api", "built_in", "fallback", "unavailable"] = "built_in"
    confidence: float = Field(default=1.0, ge=0, le=1)
    is_fallback: bool = False
    fallback_reason: str | None = None
    latency_ms: int = Field(default=0, ge=0)
    expected_arguments: dict[str, Any] = Field(default_factory=dict)
    argument_match_mode: Literal["exact", "context_tolerant_keywords"] = "exact"
    ignored_keyword_values: list[str] = Field(default_factory=list)
    error_code: str | None = None
    retryable: bool = False


class EnvironmentSnapshot(BaseModel):
    environment_version: str = Field(min_length=1)
    snapshot_version: str = Field(min_length=1)
    state_id: str = Field(min_length=1)
    tool_responses: dict[str, list[SnapshotToolResponse]] = Field(default_factory=dict)
    hidden_test_facts: dict[str, Any] = Field(default_factory=dict)


class EnvironmentRollout(BaseModel):
    environment_schema_version: str = ENVIRONMENT_SCHEMA_VERSION
    task_id: str
    seed: int
    initial_state_fingerprint: str
    environment_version: str
    snapshot_version: str
    episode: AgentEpisode
    reward: EpisodeReward
    tool_call_counts: dict[str, int] = Field(default_factory=dict)


class SnapshotToolExecutor:
    """ToolExecutor-compatible backend that never reaches live providers."""

    def __init__(self, snapshot: EnvironmentSnapshot) -> None:
        self.snapshot = snapshot.model_copy(deep=True)
        self.call_counts: dict[str, int] = {}
        # Contracted responses are selected by arguments rather than snapshot
        # position.  Separate consumed indices preserve retry sequences when
        # several responses intentionally share the same arguments.
        self._consumed_response_indices: dict[str, set[int]] = {}
        self.guard = ToolGuard(mode="enforce", max_calls=64)

    async def execute(
        self,
        tool_calls: list[dict[str, Any]],
        guard_context: GuardContext | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        decisions = self.guard.evaluate_batch(tool_calls, guard_context)
        for call, decision in zip(tool_calls, decisions, strict=True):
            tool_call_id = str(call.get("id") or "")
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            raw_arguments = function.get("arguments") or "{}"
            try:
                arguments = (
                    json.loads(raw_arguments)
                    if isinstance(raw_arguments, str)
                    else dict(raw_arguments)
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                arguments = {}

            if not decision.allowed:
                violation = decision.violations[0]
                observation = ObservationEnvelope.failure(
                    tool=name,
                    code=violation.code,
                    message=violation.message,
                    retryable=False,
                    tool_call_id=tool_call_id,
                )
                result = ToolResult(
                    data=None,
                    data_source="unavailable",
                    fallback_reason=violation.message,
                )
            else:
                result, observation = self._next(name, arguments, tool_call_id)
            records.append(
                {
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "result": result.model_dump(mode="json"),
                    "observation": observation.model_dump(mode="json"),
                    "guard": decision.model_dump(mode="json"),
                }
            )
        return records

    def _next(
        self,
        name: str,
        arguments: dict[str, Any],
        tool_call_id: str,
    ) -> tuple[ToolResult, ObservationEnvelope]:
        call_index = self.call_counts.get(name, 0)
        self.call_counts[name] = call_index + 1
        responses = self.snapshot.tool_responses.get(name) or []
        consumed = self._consumed_response_indices.setdefault(name, set())
        available = [index for index in range(len(responses)) if index not in consumed]
        if not available:
            return self._failure(
                name,
                tool_call_id,
                "SNAPSHOT_RESPONSE_EXHAUSTED",
                f"no snapshot response {call_index} for {name}",
                retryable=False,
            )

        matched_index = next(
            (
                index
                for index in available
                if responses[index].expected_arguments
                and self._arguments_match(arguments, responses[index], responses)
            ),
            None,
        )
        if matched_index is None:
            # Backward compatibility for old snapshots that did not declare an
            # argument contract: they keep their original sequential meaning.
            matched_index = next(
                (index for index in available if not responses[index].expected_arguments),
                None,
            )
        if matched_index is None:
            expected = [responses[index].expected_arguments for index in available]
            return self._failure(
                name,
                tool_call_id,
                "SNAPSHOT_ARGUMENT_MISMATCH",
                "tool arguments do not match any unused snapshot response",
                retryable=False,
                details={"actual": arguments, "expected_any": expected},
            )

        consumed.add(matched_index)
        response = responses[matched_index]
        mismatches = (
            {}
            if self._arguments_match(arguments, response, responses)
            else {
                key: {"expected": value, "actual": arguments.get(key)}
                for key, value in response.expected_arguments.items()
                if arguments.get(key) != value
            }
        )
        if mismatches:
            return self._failure(
                name,
                tool_call_id,
                "SNAPSHOT_ARGUMENT_MISMATCH",
                "tool arguments do not match the immutable snapshot",
                retryable=False,
                details=mismatches,
            )
        result = ToolResult(
            data=deepcopy(response.data),
            data_source=response.data_source,
            confidence=response.confidence,
            is_fallback=response.is_fallback,
            fallback_reason=response.fallback_reason,
            latency_ms=response.latency_ms,
        )
        if response.error_code or response.data_source == "unavailable":
            observation = ObservationEnvelope.failure(
                tool=name,
                code=response.error_code or "TOOL_UNAVAILABLE",
                message=response.fallback_reason or f"{name} unavailable in snapshot",
                retryable=response.retryable,
                tool_call_id=tool_call_id,
                latency_ms=response.latency_ms,
            )
        else:
            observation = ObservationEnvelope.from_tool_result(
                tool=name,
                result=result,
                tool_call_id=tool_call_id,
                snapshot_version=self.snapshot.snapshot_version,
                environment_version=self.snapshot.environment_version,
            )
        return result, observation

    @staticmethod
    def _arguments_match(
        arguments: dict[str, Any],
        response: SnapshotToolResponse,
        responses: list[SnapshotToolResponse],
    ) -> bool:
        expected = response.expected_arguments
        if response.argument_match_mode == "exact":
            return all(arguments.get(key) == value for key, value in expected.items())

        # A frozen search snapshot cannot enumerate every harmless context
        # token a real provider accepts, such as the trusted destination. Only
        # explicitly declared context values may be ignored; arbitrary extra
        # keywords remain a mismatch.
        expected_keywords = expected.get("keywords")
        actual_keywords = arguments.get("keywords")
        if not isinstance(expected_keywords, list) or not isinstance(actual_keywords, list):
            return False
        ignored_keywords = {
            keyword
            for candidate in responses
            if candidate.argument_match_mode == "context_tolerant_keywords"
            for keyword in candidate.ignored_keyword_values
        }
        projected = [keyword for keyword in actual_keywords if str(keyword) not in ignored_keywords]
        if projected != expected_keywords:
            return False
        return all(
            arguments.get(key) == value for key, value in expected.items() if key != "keywords"
        )

    @staticmethod
    def _failure(
        name: str,
        tool_call_id: str,
        code: str,
        message: str,
        *,
        retryable: bool,
        details: dict[str, Any] | None = None,
    ) -> tuple[ToolResult, ObservationEnvelope]:
        return (
            ToolResult(
                data=None,
                data_source="unavailable",
                fallback_reason=message,
            ),
            ObservationEnvelope.failure(
                tool=name,
                code=code,
                message=message,
                retryable=retryable,
                tool_call_id=tool_call_id,
                details=details,
            ),
        )


class TravelAgentEnvironment:
    """One immutable task/snapshot pair; each rollout gets fresh mutable state."""

    def __init__(
        self,
        task: EnvironmentTask,
        snapshot: EnvironmentSnapshot,
        *,
        reward_engine: HierarchicalRewardEngine | None = None,
    ) -> None:
        self.task = task.model_copy(deep=True)
        self.snapshot = snapshot.model_copy(deep=True)
        self.reward_engine = reward_engine or HierarchicalRewardEngine()
        self.initial_state_fingerprint = environment_fingerprint(self.task, self.snapshot)

    async def rollout(
        self,
        policy: AgentPolicy,
        *,
        safety: RewardSafetySignals | None = None,
        quality_score: float | None = None,
    ) -> EnvironmentRollout:
        state = {
            "user_input": self.task.user_request,
            "slots": deepcopy(self.task.slots),
            "profile": deepcopy(self.task.profile),
            "missing_slots": list(self.task.missing_slots),
            "feasibility_report": deepcopy(self.task.feasibility_report),
        }
        initialized = initialize_agent_ledger(state, mode="agent")
        ledger = AgentLedgerState(**initialized["agent_ledger"])
        recorder = EpisodeRecorder(
            ledger,
            environment_version=self.snapshot.environment_version,
            validator_version=VALIDATOR_VERSION,
            policy_name=type(policy).__name__,
            policy_version="rollout",
        )
        backend = SnapshotToolExecutor(self.snapshot)
        result = await BoundedAgentLoop().run(
            ledger,
            policy=policy,
            executor=TravelActionExecutor(backend),  # type: ignore[arg-type]
            recorder=recorder,
        )
        episode = recorder.episode
        if result.status == "failed" and episode.status == "running":
            raise RuntimeError("episode recorder did not finalize failed rollout")
        reward = self.reward_engine.score(
            episode,
            safety=safety,
            quality_score=quality_score,
        )
        return EnvironmentRollout(
            task_id=self.task.task_id,
            seed=self.task.seed,
            initial_state_fingerprint=self.initial_state_fingerprint,
            environment_version=self.snapshot.environment_version,
            snapshot_version=self.snapshot.snapshot_version,
            episode=episode,
            reward=reward,
            tool_call_counts=dict(backend.call_counts),
        )

    def isolated_copy(self) -> "TravelAgentEnvironment":
        return TravelAgentEnvironment(
            self.task,
            self.snapshot,
            reward_engine=HierarchicalRewardEngine(self.reward_engine.config),
        )


def create_rollout_group(
    task: EnvironmentTask,
    snapshot: EnvironmentSnapshot,
    group_size: int,
) -> list[TravelAgentEnvironment]:
    if group_size < 2:
        raise ValueError("GRPO rollout group_size must be at least 2")
    return [TravelAgentEnvironment(task, snapshot) for _ in range(group_size)]


def environment_fingerprint(task: EnvironmentTask, snapshot: EnvironmentSnapshot) -> str:
    payload = {
        "environment_schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "task": task.model_dump(mode="json"),
        "snapshot": snapshot.model_dump(mode="json"),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
