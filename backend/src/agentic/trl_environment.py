"""TRL environment adapter backed by the production interactive Agent Loop."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import threading
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal

from agentic.action_executor import TravelActionExecutor
from datetime import UTC, datetime

from agentic.clock import frozen_reference_time
from agentic.environment import (
    EnvironmentRollout,
    EnvironmentSnapshot,
    EnvironmentTask,
    SnapshotToolExecutor,
    environment_fingerprint,
)
from agentic.interactive import InteractiveAgentSession, InteractiveTransition
from agentic.loop import AgentLoopResult, PolicyAction
from agentic.policy import constrain_policy_context, controller_policy_action, policy_prompt_payload
from agentic.policy_actions import (
    PolicyArgumentValidationError,
    controller_override_attempt,
    controller_tradeoff_options,
    model_visible_policy_actions,
    policy_action_schema,
    policy_action_schemas_for_state,
    validate_policy_arguments,
)
from agentic.reward import EpisodeReward, HierarchicalRewardEngine
from agentic.reason_quality import verifier_reason_quality_checks
from agentic.grpo import policy_return_to_go_credit, policy_turn_credit_records
from agentic.grpo_training import (
    AUTHORITY_PAYLOAD_ENCODING,
    FRESH_LEDGER_ROLLOUT_CONTRACT,
    VERIFIER_REPAIR_ACTIONS_BY_ROUTE,
    VERIFIER_REPAIR_DECISION_SCHEMA_VERSION,
    VERIFIED_DECISION_STATE_REPLAY_CONTRACT,
    decode_authority_payload,
)
from agentic.runtime import initialize_agent_ledger
from agentic.state import AgentLedgerState
from evaluation.validator import VALIDATOR_VERSION


_AUDIT_LOCK = threading.Lock()
GRPOExecutionMode = Literal["policy_driven", "controller_first", "react"]


def _bounded_audit_arguments(arguments: Any) -> tuple[Any, bool, list[str]]:
    """Bound model-authored diagnostics and redact obvious credential-like fields."""
    truncated = False
    redacted: list[str] = []

    def sanitize(value: Any, *, path: str, depth: int = 0) -> Any:
        nonlocal truncated
        if depth >= 4:
            truncated = True
            return "[MAX_DEPTH]"
        if isinstance(value, str):
            if len(value) > 1000:
                truncated = True
                return value[:1000] + "...[TRUNCATED]"
            return value
        if isinstance(value, list):
            if len(value) > 20:
                truncated = True
            return [
                sanitize(item, path=f"{path}[{index}]", depth=depth + 1)
                for index, item in enumerate(value[:20])
            ]
        if isinstance(value, dict):
            if len(value) > 32:
                truncated = True
            output = {}
            for key, item in list(value.items())[:32]:
                item_path = f"{path}.{key}" if path else str(key)
                lowered = str(key).casefold()
                if any(marker in lowered for marker in ("password", "secret", "token", "api_key")):
                    redacted.append(item_path)
                    output[str(key)] = "[REDACTED]"
                else:
                    output[str(key)] = sanitize(item, path=item_path, depth=depth + 1)
            return output
        if value is None or isinstance(value, (bool, int, float)):
            return value
        truncated = True
        return repr(value)[:200]

    return sanitize(arguments, path=""), truncated, redacted


def _tolerate_copied_schema_annotations(method: Callable[..., str]) -> Callable[..., str]:
    """Validate TRL kwargs with the production contract without changing its signature."""

    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> str:
        self._policy_call_attempt_count += 1
        parameter_names = [
            name
            for name in method.__annotations__
            if name not in {"return", "self"}
        ]
        if len(args) > len(parameter_names):
            return method(self, *args, **kwargs)
        raw_arguments = dict(kwargs)
        for name, value in zip(parameter_names, args, strict=False):
            if name in raw_arguments:
                raise TypeError(f"{method.__name__} received duplicate argument {name!r}")
            raw_arguments[name] = value
        if (
            getattr(self, "_single_decision_tool_contract", False)
            and self._policy_call_attempt_count > 1
        ):
            self._policy_call_rejection_count += 1
            audit_arguments, truncated, redacted = _bounded_audit_arguments(raw_arguments)
            self._audit(
                "policy_call_rejected",
                submitted={"action": method.__name__, "arguments": audit_arguments},
                rejection={
                    "parsed_tool_call": {
                        "type": "function",
                        "function": {
                            "name": method.__name__,
                            "arguments": audit_arguments,
                        },
                    },
                    "raw_arguments_truncated": truncated,
                    "redacted_argument_paths": redacted,
                    "parse_valid": True,
                    "schema_valid": None,
                    "rejection_code": "TOOL_CALL_CARDINALITY_INVALID",
                    "validation_errors": [],
                    "execution_attempted": False,
                    "execution_valid": None,
                    "execution_status": "not_attempted",
                },
            )
            raise ValueError(
                "TOOL_CALL_CARDINALITY_INVALID: verifier decision turns require "
                "exactly one tool call"
            )
        arguments_to_validate = dict(raw_arguments)
        if method.__name__ == "propose_tradeoff" and "options" not in parameter_names:
            arguments_to_validate.pop("options", None)
        try:
            validated = validate_policy_arguments(method.__name__, arguments_to_validate)
        except PolicyArgumentValidationError as exc:
            self._policy_call_rejection_count += 1
            self._policy_argument_rejection_count += 1
            audit_arguments, truncated, redacted = _bounded_audit_arguments(raw_arguments)
            self._audit(
                "policy_argument_rejected",
                submitted={"action": method.__name__, "arguments": audit_arguments},
                rejection={
                    "parsed_tool_call": {
                        "type": "function",
                        "function": {
                            "name": method.__name__,
                            "arguments": audit_arguments,
                        },
                    },
                    "raw_arguments_truncated": truncated,
                    "redacted_argument_paths": redacted,
                    "parse_valid": True,
                    "schema_valid": False,
                    "rejection_code": exc.rejection_code,
                    "validation_errors": exc.validation_errors,
                    "execution_attempted": False,
                    "execution_valid": None,
                    "execution_status": "not_attempted",
                },
            )
            raise
        # The production action model may retain legacy/defaulted fields that a
        # route-specific TRL method intentionally no longer exposes.  Only pass
        # arguments present in the actual callable signature; the Agent Loop
        # will hydrate controller-owned fields at its authorization boundary.
        validated = {
            name: value for name, value in validated.items() if name in parameter_names
        }
        if getattr(self, "_pending_model_action_audit", None) is None:
            self._set_pending_model_action_audit(
                PolicyAction(
                    action=method.__name__,
                    arguments=validated,
                    model_arguments=raw_arguments,
                    controller_override_attempt=controller_override_attempt(
                        method.__name__, raw_arguments
                    ),
                    model_contract_compliant=not controller_override_attempt(
                        method.__name__, raw_arguments
                    ),
                )
            )
        return method(self, **validated)

    return wrapped


def canonical_trl_tool_schemas(
    tools: list[Any],
    *,
    action_order: tuple[str, ...] | None = None,
    capability: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Render TRL environment methods with the production Pydantic schemas."""
    tools_by_name: dict[str, Any] = {}
    for tool in tools:
        if isinstance(tool, dict):
            name = (tool.get("function") or {}).get("name")
        else:
            name = getattr(tool, "__name__", None)
        if not isinstance(name, str):
            raise TypeError("TRL environment tools must be callables or schema dictionaries")
        if name in tools_by_name:
            raise ValueError(f"duplicate TRL environment tool: {name}")
        tools_by_name[name] = tool
    ordered_names = action_order or tuple(tools_by_name)
    actual_names = tuple(tools_by_name)
    if capability is not None:
        actual_names = tuple(
            model_visible_policy_actions(actual_names, capability=capability)
        )
    if set(ordered_names) != set(actual_names):
        raise ValueError(
            "TRL environment tool surface differs from the declared action order: "
            f"declared={list(ordered_names)}, actual={list(actual_names)}"
        )
    if capability is None:
        return [policy_action_schema(name) for name in ordered_names]
    return policy_action_schemas_for_state(ordered_names, capability=capability)


class _TRLTravelEnvironmentBase:
    """Stateful, snapshot-only environment compatible with TRL ``environment_factory``.

    TRL exposes every public method except ``reset`` and ``get_reward`` as a
    model-callable tool. The production task graph remains authoritative: a
    method call that is not allowed for the current state is recorded as a
    failed action and cannot mutate trusted facts or artifacts.
    """

    def __init__(
        self,
        *,
        audit_enabled: bool = True,
        execution_mode: GRPOExecutionMode = "policy_driven",
    ) -> None:
        if execution_mode not in {"policy_driven", "controller_first", "react"}:
            raise ValueError(f"unsupported GRPO execution mode: {execution_mode}")
        self._session: InteractiveAgentSession | None = None
        self._runner: _SessionLoopThread | None = None
        self._transition: InteractiveTransition | None = None
        self._reward: EpisodeReward | None = None
        self._reward_engine = HierarchicalRewardEngine()
        self._task_id: str | None = None
        self._task: EnvironmentTask | None = None
        self._snapshot: EnvironmentSnapshot | None = None
        self._backend: SnapshotToolExecutor | None = None
        self._audit_enabled = audit_enabled
        self.execution_mode = execution_mode
        self._rollout_contract = FRESH_LEDGER_ROLLOUT_CONTRACT
        self._policy_call_attempt_count = 0
        self._policy_call_rejection_count = 0
        self._policy_argument_rejection_count = 0
        self._pending_model_action_audit: dict[str, Any] | None = None

    def reset(
        self,
        *,
        task: dict[str, Any] | str,
        snapshot: dict[str, Any] | str,
        prompt: list[dict[str, Any]] | None = None,
        authority_payload_encoding: str | None = None,
        **_: Any,
    ) -> str:
        """Reset one rollout from an immutable task and tool-response snapshot."""
        if self._session is not None and self._session.recorder.episode.status == "running":
            raise RuntimeError("previous rollout was not scored before environment reuse")
        if self._runner is not None:
            self._runner.close()
            self._runner = None
        self._policy_call_attempt_count = 0
        self._policy_call_rejection_count = 0
        self._policy_argument_rejection_count = 0
        self._pending_model_action_audit = None
        task_is_scalar = isinstance(task, str)
        snapshot_is_scalar = isinstance(snapshot, str)
        legacy_payload = False
        if task_is_scalar and snapshot_is_scalar:
            if authority_payload_encoding != AUTHORITY_PAYLOAD_ENCODING:
                raise ValueError(
                    "canonical authority payloads require "
                    f"authority_payload_encoding={AUTHORITY_PAYLOAD_ENCODING!r}"
                )
        elif isinstance(task, dict) and isinstance(snapshot, dict):
            if authority_payload_encoding is not None:
                raise ValueError("legacy dict authority payloads must not declare an encoding")
            legacy_payload = True
        else:
            raise ValueError(
                "task and snapshot authority payloads must use the same representation"
            )
        decoded_task = decode_authority_payload(task, field="task")
        decoded_snapshot = decode_authority_payload(snapshot, field="snapshot")
        parsed_task = EnvironmentTask(**decoded_task)
        # Retain compatibility for trusted internal dict callers created before
        # authority payloads were moved to scalar JSON at the Dataset boundary.
        normalized_snapshot = dict(decoded_snapshot)
        if legacy_payload:
            normalized_snapshot["tool_responses"] = {
                name: responses or []
                for name, responses in (decoded_snapshot.get("tool_responses") or {}).items()
            }
        parsed_snapshot = EnvironmentSnapshot(**normalized_snapshot)
        self._validate_initial_prompt(prompt, user_request=parsed_task.user_request)
        self._rollout_contract = (
            VERIFIED_DECISION_STATE_REPLAY_CONTRACT
            if isinstance(prompt, list) and len(prompt) > 2
            else FRESH_LEDGER_ROLLOUT_CONTRACT
        )
        self._task = parsed_task
        self._snapshot = parsed_snapshot
        self._task_id = parsed_task.task_id
        initialized = initialize_agent_ledger(
            {
                "user_input": parsed_task.user_request,
                "slots": parsed_task.slots,
                "profile": parsed_task.profile,
                "missing_slots": parsed_task.missing_slots,
                "feasibility_report": parsed_task.feasibility_report,
            },
            mode="agent",
            task_graph_mode="react" if self.execution_mode == "react" else "configured",
        )
        ledger = AgentLedgerState(**initialized["agent_ledger"])
        backend = SnapshotToolExecutor(parsed_snapshot)
        self._backend = backend
        self._session = InteractiveAgentSession(
            ledger,
            executor=TravelActionExecutor(backend),  # type: ignore[arg-type]
            environment_version=parsed_snapshot.environment_version,
            validator_version=VALIDATOR_VERSION,
            policy_name=f"trl-grpo-{self.execution_mode}",
            policy_version="online-rollout",
            automatic_action=(
                controller_policy_action
                if self.execution_mode in {"controller_first", "react"}
                else None
            ),
        )
        self._runner = _SessionLoopThread()
        # H-002: resolve the frozen authoring moment BEFORE the session
        # starts, because the task graph (declared research requirements,
        # including the ten-day weather window) is materialized during
        # session.start() on the session thread.
        self._frozen_moment: datetime | None = self._frozen_reference_moment()
        start = self._session.start()
        if self._frozen_moment is not None:
            start = self._run_under_frozen_clock(start)
        self._transition = self._runner.run(start)
        if self._transition.done or self._transition.next_context is None:
            raise RuntimeError("environment task has no policy-owned decision")
        self._reward = None
        self._audit("reset", transition=self._transition)
        return self._render_context(self._transition.next_context)

    def _reject_policy_call_batch(
        self,
        tool_calls: list[dict[str, Any]],
        *,
        rejection_code: str,
    ) -> None:
        """Record a Trainer-side whole-message rejection before any tool executes."""
        attempted = max(1, len(tool_calls))
        self._policy_call_attempt_count += attempted
        self._policy_call_rejection_count += attempted
        bounded_calls, truncated, redacted = _bounded_audit_arguments(tool_calls)
        self._audit(
            "policy_call_batch_rejected",
            submitted={"tool_calls": bounded_calls},
            rejection={
                "parse_valid": True,
                "schema_valid": None,
                "rejection_code": rejection_code,
                "raw_arguments_truncated": truncated,
                "redacted_argument_paths": redacted,
                "execution_attempted": False,
                "execution_valid": None,
                "execution_status": "not_attempted",
            },
        )

    def _trl_tool_loop_done(self) -> bool:
        """Tell the pinned Trainer whether another model/tool turn is meaningful."""
        return False

    @staticmethod
    def _validate_initial_prompt(
        prompt: list[dict[str, Any]] | None,
        *,
        user_request: str,
    ) -> None:
        """Reject teacher-forced history before an online GRPO rollout starts."""
        if prompt is None:
            return
        roles = [str(message.get("role") or "") for message in prompt]
        if roles != ["system", "user"]:
            raise ValueError(
                "GRPO rollout prompt must contain exactly system + user messages; "
                "assistant/tool trajectory prefixes are forbidden"
            )
        if any(message.get("tool_calls") for message in prompt):
            raise ValueError("GRPO rollout prompt cannot contain teacher tool calls")
        if str(prompt[-1].get("content") or "") != user_request:
            raise ValueError("GRPO rollout user prompt must match the immutable task request")

    def get_reward(self) -> float:
        """Return the gated six-component trajectory reward."""
        session = self._require_session()
        episode = session.recorder.episode
        if episode.status == "running":
            runner = self._require_runner()
            runner.run(session.aclose())
            session.recorder.finalize(
                AgentLoopResult(
                    ledger=session.ledger,
                    status="failed",
                    termination_reason="rollout_truncated",
                    events=[],
                )
            )
        self._reward = self._reward_engine.score(session.recorder.episode)
        self._audit("reward", reward=self._reward)
        if self._runner is not None:
            self._runner.close()
            self._runner = None
        return self._reward.episode_reward

    @property
    def reward_record(self) -> EpisodeReward | None:
        """Expose the auditable breakdown to logging callbacks, never to the model."""
        return self._reward

    @property
    def rollout_record(self) -> EnvironmentRollout | None:
        """Return the scored rollout for offline curriculum auditing."""
        if (
            self._reward is None
            or self._session is None
            or self._task is None
            or self._snapshot is None
        ):
            return None
        return EnvironmentRollout(
            task_id=self._task.task_id,
            seed=self._task.seed,
            initial_state_fingerprint=environment_fingerprint(self._task, self._snapshot),
            environment_version=self._snapshot.environment_version,
            snapshot_version=self._snapshot.snapshot_version,
            episode=self._session.recorder.episode,
            reward=self._reward,
            tool_call_counts=(dict(self._backend.call_counts) if self._backend is not None else {}),
        )

    def _frozen_reference_moment(self) -> datetime | None:
        """Best-effort authoring time of the frozen snapshot, if recorded."""
        latest: datetime | None = None
        try:
            payload = self._snapshot.model_dump(mode="json") if self._snapshot else None
        except (AttributeError, TypeError, ValueError):
            payload = None
        if not isinstance(payload, dict):
            return None

        def visit(node: Any) -> None:
            nonlocal latest
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in {"queried_at", "retrieved_at", "created_at", "generated_at"} and isinstance(value, str):
                        try:
                            moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
                        except ValueError:
                            continue
                        if moment.tzinfo is None:
                            moment = moment.replace(tzinfo=UTC)
                        if latest is None or moment > latest:
                            latest = moment
                    else:
                        visit(value)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(payload)
        return latest

    async def _run_under_frozen_clock(self, step: Any) -> Any:
        """Execute one session step inside the frozen reference clock.

        The session loop lives on its own thread and ``run_coroutine_threadsafe``
        does not propagate the caller's contextvars, so the frozen moment is
        carried on the instance and entered here on the session thread.
        """
        with frozen_reference_time(self._frozen_moment):
            return await step

    def _act(self, action: str, arguments: dict[str, Any]) -> str:
        session = self._require_session()
        validated = validate_policy_arguments(action, arguments)
        pending = self._pending_model_action_audit
        self._pending_model_action_audit = None
        if pending is not None and pending.get("action") != action:
            raise RuntimeError("pending model action audit does not match submitted action")
        model_arguments = (
            dict(pending.get("model_arguments") or {})
            if pending is not None
            else dict(arguments)
        )
        runner = self._require_runner()
        step = session.submit(
            PolicyAction(
                action=action,
                arguments=validated,
                model_arguments=model_arguments,
                controller_override_attempt=bool(
                    pending and pending.get("controller_override_attempt")
                ),
                model_contract_compliant=(
                    bool(pending.get("model_contract_compliant"))
                    if pending is not None
                    else True
                ),
            )
        )
        if self._frozen_moment is not None:
            step = self._run_under_frozen_clock(step)
        transition = runner.run(step)
        self._transition = transition
        self._audit(
            "action",
            transition=transition,
            submitted={"action": action, "arguments": validated},
        )
        return self._render_transition(transition)

    def _set_pending_model_action_audit(self, action: PolicyAction) -> None:
        """Attach out-of-band policy authorship to the next public TRL tool call."""
        if self._pending_model_action_audit is not None:
            raise RuntimeError("pending model action audit was not consumed")
        self._pending_model_action_audit = {
            "action": action.action,
            "model_arguments": dict(
                action.model_arguments
                if action.model_arguments is not None
                else action.arguments
            ),
            "controller_override_attempt": action.controller_override_attempt,
            "model_contract_compliant": action.model_contract_compliant,
        }

    def _policy_turn_credits(self, gamma: float) -> list[float]:
        """Expose model-owned credits to the trainer, never as a callable tool."""
        if self._reward is None or self._session is None:
            return []
        return policy_return_to_go_credit(
            self._reward,
            self._session.recorder.episode,
            gamma=gamma,
        )

    def _policy_turn_credit_records(self, gamma: float) -> list[dict[str, Any]]:
        """Expose auditable validity metadata to the trainer, never the model."""
        if self._reward is None or self._session is None:
            return []
        return [
            item.model_dump(mode="json")
            for item in policy_turn_credit_records(
                self._reward,
                self._session.recorder.episode,
                gamma=gamma,
            )
        ]

    def _audit(
        self,
        event: str,
        *,
        transition: InteractiveTransition | None = None,
        submitted: dict[str, Any] | None = None,
        reward: EpisodeReward | None = None,
        rejection: dict[str, Any] | None = None,
    ) -> None:
        """Optionally persist a minimal rollout audit for smoke diagnosis."""
        raw_path = os.environ.get("AGENTIC_GRPO_AUDIT_PATH")
        if not self._audit_enabled or not raw_path:
            return
        session = self._session
        episode = session.recorder.episode if session is not None else None
        payload = {
            "event": event,
            "environment": type(self).__name__,
            "execution_mode": self.execution_mode,
            "rollout_contract": self._rollout_contract,
            "credited_step_start": int(getattr(self, "_decision_step_start", 0)),
            "task_id": self._task_id,
            "initial_state_fingerprint": (
                environment_fingerprint(self._task, self._snapshot)
                if self._task is not None and self._snapshot is not None
                else None
            ),
            "trajectory_id": episode.trajectory_id if episode else None,
            "episode_status": episode.status if episode else None,
            "termination_reason": episode.termination_reason if episode else None,
            "submitted": submitted,
            "rejection": rejection,
            "transition": (
                {
                    "done": transition.done,
                    "status": transition.status,
                    "termination_reason": transition.termination_reason,
                    "next_allowed_actions": (
                        transition.next_context.allowed_actions
                        if transition.next_context is not None
                        else None
                    ),
                }
                if transition is not None
                else None
            ),
            "steps": (
                [
                    {
                        "index": step.step_index,
                        "task_id": step.task_id,
                        "action": step.action.action,
                        "arguments": step.action.arguments,
                        "decision_source": step.action.decision_source,
                        "allowed_actions": step.context.allowed_actions,
                        "decision_cardinality": len(step.context.allowed_actions),
                        "verification": step.verification,
                        "observations": [
                            {
                                "tool": item.tool,
                                "ok": item.ok,
                                "error_code": item.error.code if item.error else None,
                                "is_fallback": item.is_fallback,
                            }
                            for item in step.observations
                        ],
                        "turn_reward": (
                            reward.turn_rewards[step.step_index].model_dump(mode="json")
                            if reward is not None and step.step_index < len(reward.turn_rewards)
                            else None
                        ),
                    }
                    for step in episode.steps
                ]
                if episode is not None
                else []
            ),
            "reward": reward.model_dump(mode="json") if reward is not None else None,
        }
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOCK, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    def _require_session(self) -> InteractiveAgentSession:
        if self._session is None:
            raise RuntimeError("environment must be reset before use")
        return self._session

    def _require_runner(self) -> _SessionLoopThread:
        if self._runner is None:
            raise RuntimeError("environment rollout loop is not running")
        return self._runner

    @staticmethod
    def _render_context(context: Any) -> str:
        context = constrain_policy_context(context)
        return json.dumps(
            {"policy_state": policy_prompt_payload(context)},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _render_transition(cls, transition: InteractiveTransition) -> str:
        payload: dict[str, Any] = {"done": transition.done}
        if transition.committed_step is not None:
            payload["last_transition"] = {
                "action": transition.committed_step.action.action,
                "observations": [
                    {
                        "ok": item.ok,
                        "tool": item.tool,
                        "error_code": item.error.code if item.error else None,
                        "is_fallback": item.is_fallback,
                    }
                    for item in transition.committed_step.observations
                ],
                "verification": transition.committed_step.verification,
            }
        if transition.next_context is not None:
            payload["policy_state"] = policy_prompt_payload(
                constrain_policy_context(transition.next_context)
            )
        if transition.done:
            payload["status"] = transition.status
            payload["termination_reason"] = transition.termination_reason
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class _SessionLoopThread:
    """Keep one interactive production loop alive behind TRL's synchronous reset."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit(self, coroutine: Any) -> concurrent.futures.Future[Any]:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop)

    def run(self, coroutine: Any) -> Any:
        return self.submit(coroutine).result()

    def close(self) -> None:
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive() and threading.current_thread() is not self.thread:
            self.thread.join(timeout=2)
        if not self.loop.is_closed():
            self.loop.close()


class TRLSearchEnvironment(_TRLTravelEnvironmentBase):
    """Controller-first baseline exposing only the delegated search decision."""

    def __init__(self, *, audit_enabled: bool = True) -> None:
        super().__init__(audit_enabled=audit_enabled, execution_mode="controller_first")

    def search_pois(
        self,
        keywords: list[str] | None = None,
    ) -> str:
        """Search POIs using grounded preferences.

        Args:
            keywords: Grounded preference keywords.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("search_pois", {"keywords": keywords or []})


class TRLClarificationEnvironment(_TRLTravelEnvironmentBase):
    """Controller-first baseline exposing only the delegated clarification."""

    def __init__(self, *, audit_enabled: bool = True) -> None:
        super().__init__(audit_enabled=audit_enabled, execution_mode="controller_first")

    def ask_user(self, question: str) -> str:
        """Ask for information only the user can provide.

        Args:
            question: One concise, grounded question.

        Returns:
            The terminal clarification transition.
        """
        return self._act("ask_user", {"question": question})


class TRLTradeoffEnvironment(_TRLTravelEnvironmentBase):
    """Controller-first baseline exposing delegated terminal trade-offs."""

    def __init__(self, *, audit_enabled: bool = True) -> None:
        super().__init__(audit_enabled=audit_enabled, execution_mode="controller_first")

    def propose_tradeoff(self, reason: str, options: list[str] | None = None) -> str:
        """Offer grounded alternatives for the current conflict.

        Args:
            reason: The grounded constraint conflict.
            options: Up to three grounded alternatives.

        Returns:
            The terminal tradeoff transition.
        """
        return self._act("propose_tradeoff", {"reason": reason, "options": options or []})

    def abort(self, reason: str) -> str:
        """Stop when no safe or feasible alternative exists.

        Args:
            reason: The grounded reason the task cannot continue.

        Returns:
            The terminal transition.
        """
        return self._act("abort", {"reason": reason})


class TRLPolicyDrivenEnvironment(_TRLTravelEnvironmentBase):
    """Expose the complete production action contract for every model turn.

    TRL currently discovers a static tool schema from public environment
    methods. The live state still supplies the authoritative ``allowed_actions``
    subset and the production loop rejects any out-of-state call.
    """

    def abort(self, reason: str) -> str:
        """Stop when the task cannot continue safely or feasibly.

        Args:
            reason: Grounded reason the task cannot continue.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("abort", {"reason": reason})

    def accept_candidates(self) -> str:
        """Accept the grounded candidate set after observing search results."""
        return self._act("accept_candidates", {})

    def accept_itinerary(self) -> str:
        """Accept the itinerary only after the latest hard-pass report."""
        return self._act("accept_itinerary", {})

    def ask_user(self, question: str) -> str:
        """Ask one grounded question for information only the user can provide.

        Args:
            question: One concise question for the missing information.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("ask_user", {"question": question})

    def capability_check(self) -> str:
        """Commit the controller-computed capability assessment.

        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("capability_check", {})

    def compose_draft(self) -> str:
        """Compose a draft from the verified solver artifact.

        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("compose_draft", {})

    def finish(self) -> str:
        """Present the verified draft and wait for user confirmation.

        Returns:
            The terminal confirmation transition.
        """
        return self._act("finish", {})

    def propose_tradeoff(
        self,
        reason: str,
        options: list[str] | None = None,
    ) -> str:
        """Offer up to three grounded alternatives for a constraint conflict.

        Args:
            reason: Grounded constraint conflict.
            options: Up to three grounded alternatives.
        Returns:
            The terminal trade-off transition.
        """
        return self._act("propose_tradeoff", {"reason": reason, "options": options or []})

    def retry_solve(self, reason: str) -> str:
        """Request one controller-bounded retry with a verifier-grounded reason.

        Args:
            reason: Verifier-grounded reason that the previous solve must be retried.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("retry_solve", {"reason": reason})

    def get_weather(self, date: str | None = None) -> str:
        """Read the trusted destination weather snapshot for an optional date.

        Args:
            date: Grounded date in YYYY-MM-DD format.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("get_weather", {"date": date} if date else {})

    def search_pois(self, keywords: list[str] | None = None) -> str:
        """Search POIs using grounded preference keywords.

        Args:
            keywords: Up to eight grounded preference keywords.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("search_pois", {"keywords": keywords or []})

    def retrieve_city_knowledge(self, topic: str | None = None) -> str:
        """Read stable destination facts from the local knowledge base.

        Args:
            topic: Optional grounded topic to narrow the lookup.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("retrieve_city_knowledge", {"topic": topic} if topic else {})

    def search_current_info(
        self,
        query: str,
        info_type: Literal[
            "event", "opening_hours", "restaurant", "seasonal_activity", "closure", "general"
        ] = "general",
        date: str | None = None,
    ) -> str:
        """Search source-backed current facts through the generic live-search tool.

        Args:
            query: Grounded event, opening-hours, restaurant, closure, or seasonal query.
            info_type: Type of current information being requested.
            date: Optional grounded date in YYYY-MM-DD format.
        Returns:
            The verified transition and next policy state, if any.
        """
        arguments: dict[str, Any] = {"query": query, "info_type": info_type}
        if date:
            arguments["date"] = date
        return self._act("search_current_info", arguments)

    def search_transport(
        self,
        mode: Literal["flight", "train", "both"] = "both",
        date: str | None = None,
    ) -> str:
        """Search source-backed current flight or train schedule evidence.

        Args:
            mode: Transport mode to search.
            date: Optional grounded departure date in YYYY-MM-DD format.
        Returns:
            The verified transition and next policy state, if any.
        """
        arguments: dict[str, Any] = {"mode": mode}
        if date:
            arguments["date"] = date
        return self._act("search_transport", arguments)

    def finalize_research(self) -> str:
        """Ask the evidence verifier to confirm that research is sufficient."""
        return self._act("finalize_research", {})

    def get_poi_detail(self) -> str:
        """Collect details for controller-selected POI candidates.

        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("get_poi_detail", {})

    def get_route_matrix(self) -> str:
        """Build a route matrix from trusted candidate artifacts.

        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("get_route_matrix", {})

    def solve_itinerary(
        self,
        strategy: Literal["auto", "cpsat", "greedy"] = "auto",
    ) -> str:
        """Run the deterministic itinerary solver with a bounded strategy.

        Args:
            strategy: Auto, CP-SAT, or greedy solver selection.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("solve_itinerary", {"strategy": strategy})

    def validate_itinerary(self) -> str:
        """Run the programmatic hard-constraint validator.

        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("validate_itinerary", {})


class TRLReactEnvironment(_TRLTravelEnvironmentBase):
    """Train the same hybrid ReAct decision boundary used in production.

    The model sees the complete static tool schema but only chooses actions at
    genuine research, recovery, clarification, and trade-off branches. Mandatory
    solver, verifier, composition, and completion transitions are advanced by
    the shared production controller.
    """

    def __init__(self, *, audit_enabled: bool = True) -> None:
        super().__init__(
            audit_enabled=audit_enabled,
            execution_mode="react",
        )

    # Only actions owned by the production policy are exposed. Solver,
    # verifier, composition, completion, and the finalize-research gate remain
    # controller-owned and therefore cannot be sampled by GRPO. The review
    # actions below are intentionally included because production delegates a
    # failed-verifier repair/trade-off choice back to the model.
    abort = _tolerate_copied_schema_annotations(TRLPolicyDrivenEnvironment.abort)
    accept_itinerary = _tolerate_copied_schema_annotations(
        TRLPolicyDrivenEnvironment.accept_itinerary
    )
    ask_user = _tolerate_copied_schema_annotations(TRLPolicyDrivenEnvironment.ask_user)
    propose_tradeoff = _tolerate_copied_schema_annotations(
        TRLPolicyDrivenEnvironment.propose_tradeoff
    )
    retry_solve = _tolerate_copied_schema_annotations(
        TRLPolicyDrivenEnvironment.retry_solve
    )
    get_weather = _tolerate_copied_schema_annotations(TRLPolicyDrivenEnvironment.get_weather)
    search_pois = _tolerate_copied_schema_annotations(TRLPolicyDrivenEnvironment.search_pois)
    retrieve_city_knowledge = _tolerate_copied_schema_annotations(
        TRLPolicyDrivenEnvironment.retrieve_city_knowledge
    )
    get_poi_detail = _tolerate_copied_schema_annotations(
        TRLPolicyDrivenEnvironment.get_poi_detail
    )
    get_route_matrix = _tolerate_copied_schema_annotations(
        TRLPolicyDrivenEnvironment.get_route_matrix
    )


class TRLReactCurrentInfoEnvironment(TRLReactEnvironment):
    """ReAct research environment that additionally permits live current facts."""

    search_current_info = _tolerate_copied_schema_annotations(
        TRLPolicyDrivenEnvironment.search_current_info
    )


class TRLReactTransportEnvironment(TRLReactEnvironment):
    """ReAct research environment that additionally permits live transport facts."""

    search_transport = _tolerate_copied_schema_annotations(
        TRLPolicyDrivenEnvironment.search_transport
    )


class TRLReactGetPoiDetailDecisionEnvironment(_TRLTravelEnvironmentBase):
    """One verified production decision state for argument-level GRPO.

    TRL 1.9 keeps one static tool schema for a complete tool loop, while the
    production ReAct scheduler narrows that schema after every transition. This
    environment replays a hidden, verified prefix and exposes only the current
    decision. It therefore trains the same state-scoped contract used online
    without teacher-forcing any tokens into the model prompt.
    """

    def __init__(self, *, audit_enabled: bool = True) -> None:
        super().__init__(audit_enabled=audit_enabled, execution_mode="react")
        self._decision_step_start = 0

    def reset(self, **kwargs: Any) -> str:
        rendered = super().reset(**kwargs)
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("decision-state snapshot was not initialized")
        decision_state = snapshot.hidden_test_facts.get("grpo_decision_state")
        if not isinstance(decision_state, dict):
            raise ValueError("decision-state metadata is missing")
        prompt = kwargs.get("prompt")
        authoritative_prompt = decision_state.get("prompt_messages")
        if isinstance(prompt, list) and len(prompt) > 2:
            if not isinstance(authoritative_prompt, list) or prompt != authoritative_prompt:
                raise ValueError("decision-state replay prompt is not authoritative")
        if decision_state.get("target_action") != "get_poi_detail":
            raise ValueError("decision-state target does not match environment")
        for item in decision_state.get("prefix_actions") or []:
            if not isinstance(item, dict):
                raise ValueError("decision-state prefix action must be an object")
            raw_arguments = item.get("arguments")
            if not isinstance(raw_arguments, dict):
                raise ValueError("decision-state prefix arguments must be an object")
            if any(value is None for value in raw_arguments.values()):
                raise ValueError("decision-state prefix arguments must not contain null")
            prefix_arguments = dict(raw_arguments)
            rendered = self._act(
                str(item.get("action") or ""),
                prefix_arguments,
            )
            if json.loads(rendered).get("done") is True:
                raise ValueError("decision-state prefix terminated before the target")
        session = self._require_session()
        self._decision_step_start = len(session.recorder.episode.steps)
        transition = json.loads(rendered)
        allowed = list((transition.get("policy_state") or {}).get("allowed_actions") or [])
        if "get_poi_detail" not in allowed:
            raise ValueError("replayed decision state does not allow get_poi_detail")
        rendered = json.dumps(
            transition,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._audit("decision_replay_ready", transition=self._transition)
        # A verified replay prompt already ends with the exact transition above.
        # TRL appends reset() output to the final message, so return an empty
        # observation to avoid duplicating the state in that contract.
        if isinstance(prompt, list) and len(prompt) > 2:
            if str(prompt[-1].get("content") or "") != rendered:
                raise ValueError("decision-state replay prompt does not match replayed state")
            return ""
        return rendered

    @staticmethod
    def _validate_initial_prompt(
        prompt: list[dict[str, Any]] | None,
        *,
        user_request: str,
    ) -> None:
        if prompt is None or len(prompt) == 2:
            _TRLTravelEnvironmentBase._validate_initial_prompt(
                prompt,
                user_request=user_request,
            )
            return
        roles = [str(message.get("role") or "") for message in prompt]
        if roles[:2] != ["system", "user"] or roles[-1] != "tool":
            raise ValueError("decision-state replay prompt has an invalid role sequence")
        if any(role not in {"system", "user", "assistant", "tool"} for role in roles):
            raise ValueError("decision-state replay prompt contains an unsupported role")
        for index in range(2, len(roles), 2):
            if roles[index : index + 2] != ["assistant", "tool"]:
                raise ValueError("decision-state replay prompt must alternate assistant/tool")

    def _complete_decision(self, action: str, arguments: dict[str, Any]) -> str:
        rendered = json.loads(self._act(action, arguments))
        rendered.pop("policy_state", None)
        rendered.update({"done": True, "decision_complete": True})
        return json.dumps(rendered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def retrieve_city_knowledge(self, topic: str | None = None) -> str:
        """Read stable destination facts from the local knowledge base.

        Args:
            topic: Optional grounded topic to narrow the lookup.
        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision(
            "retrieve_city_knowledge",
            {"topic": topic} if topic else {},
        )

    def get_poi_detail(self) -> str:
        """Collect details for the controller-selected POI candidates.

        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision("get_poi_detail", {})

    def get_route_matrix(self) -> str:
        """Build a route matrix from trusted candidate artifacts.

        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision("get_route_matrix", {})

    def get_reward(self) -> float:
        """Reward the verified target decision, independent of unfinished downstream work."""
        session = self._require_session()
        decision_steps = session.recorder.episode.steps[self._decision_step_start :]
        valid = any(
            step.action.decision_source != "controller"
            and step.action.action == "get_poi_detail"
            and all(observation.ok for observation in step.observations)
            and not step.verification.get("error_code")
            for step in decision_steps
        )
        super().get_reward()
        score = 1.0 if valid else -1.0
        if self._reward is None:
            raise RuntimeError("decision-state reward was not initialized")
        self._reward = self._reward.model_copy(
            update={
                "episode_reward": score,
                # The generic full-episode scorer correctly marks the replayed
                # prefix as unfinished.  For this declared one-decision
                # contract, however, a verified target action is the task and
                # constraint success.  Keep these components aligned with the
                # custom gate so curriculum routing does not send useful mixed
                # groups back to SFT merely because the full trip is not done.
                "components": self._reward.components.model_copy(
                    update={"task": score, "constraint": score}
                ),
                "gate_status": "passed" if valid else "task_failed",
                "gate_reasons": [] if valid else ["DECISION_ACTION_INVALID_OR_MISSING"],
                "audit_metrics": {
                    **self._reward.audit_metrics,
                    "decision_state_training": True,
                    "decision_step_valid": valid,
                },
            }
        )
        self._audit("decision_reward", reward=self._reward)
        return score


class _TRLReactVerifierRepairDecisionEnvironmentBase(_TRLTravelEnvironmentBase):
    """Replay a verified prefix and train one production review decision.

    The hidden prefix is executed through the same ReAct controller, snapshot
    executor, CP-SAT boundary, and verifier transition used online. Only the
    final model-owned review decision receives GRPO credit. This avoids
    spending most rollout tokens relearning deterministic setup actions while
    preserving the exact production state shown to the policy.
    """

    _SUPPORTED_TARGETS = {"retry_solve", "propose_tradeoff", "abort"}
    _single_decision_tool_contract = True

    def __init__(self, *, audit_enabled: bool = True) -> None:
        super().__init__(audit_enabled=audit_enabled, execution_mode="react")
        self._decision_step_start = 0
        self._decision_contract: dict[str, Any] = {}


    def reset(self, **kwargs: Any) -> str:
        rendered = super().reset(**kwargs)
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("decision-state snapshot was not initialized")
        decision_state = snapshot.hidden_test_facts.get("grpo_decision_state")
        if not isinstance(decision_state, dict):
            raise ValueError("decision-state metadata is missing")
        schema_version = str(decision_state.get("schema_version") or "")
        if schema_version != VERIFIER_REPAIR_DECISION_SCHEMA_VERSION:
            raise ValueError(
                "unsupported verifier-repair decision-state schema: "
                f"{schema_version or 'missing'}"
            )
        prompt = kwargs.get("prompt")
        authoritative_prompt = decision_state.get("prompt_messages")
        if isinstance(prompt, list) and len(prompt) > 2:
            if not isinstance(authoritative_prompt, list) or prompt != authoritative_prompt:
                raise ValueError("decision-state replay prompt is not authoritative")
        target = str(decision_state.get("target_action") or "")
        if target not in self._SUPPORTED_TARGETS:
            raise ValueError("verifier-repair target does not match environment")
        expected_target = getattr(self, "_EXPECTED_TARGET", None)
        if expected_target is not None and target != expected_target:
            raise ValueError(
                f"verifier-repair target {target!r} does not match route {expected_target!r}"
            )
        expected_actions = getattr(self, "_EXPECTED_ACTIONS", None)
        if not isinstance(expected_actions, tuple) or not expected_actions:
            raise RuntimeError("verifier-repair route action contract is not configured")
        review_allowed_actions = decision_state.get("review_allowed_actions")
        if not isinstance(review_allowed_actions, list) or tuple(
            str(action) for action in review_allowed_actions
        ) != expected_actions:
            raise ValueError(
                "verifier-repair decision metadata does not match route action contract"
            )
        # H-002: the state is frozen at authoring time.  Date-relative
        # evidence requirements (the ten-day weather window) and freshness
        # TTLs must be evaluated against that moment, not the wall clock, or
        # the recorded prefix silently stops reaching itinerary review once
        # the calendar crosses a requirement boundary.
        for item in decision_state.get("prefix_actions") or []:
            if not isinstance(item, dict):
                raise ValueError("decision-state prefix action must be an object")
            raw_arguments = item.get("arguments")
            if not isinstance(raw_arguments, dict):
                raise ValueError("decision-state prefix arguments must be an object")
            if any(value is None for value in raw_arguments.values()):
                raise ValueError("decision-state prefix arguments must not contain null")
            prefix_arguments = dict(raw_arguments)
            rendered = self._act(str(item.get("action") or ""), prefix_arguments)
            if json.loads(rendered).get("done") is True:
                raise ValueError("decision-state prefix terminated before the target")
        session = self._require_session()
        self._decision_step_start = len(session.recorder.episode.steps)
        self._decision_contract = decision_state
        transition = json.loads(rendered)
        policy_state = transition.get("policy_state") or {}
        if (policy_state.get("current_subtask") or {}).get("task_id") != "review_itinerary":
            raise ValueError("replayed decision state did not reach itinerary review")
        allowed = list(policy_state.get("allowed_actions") or [])
        if tuple(str(action) for action in allowed) != expected_actions:
            raise ValueError(
                "replayed decision state does not match route action contract"
            )
        expected_arguments = decision_state.get("expected_arguments")
        expected_controller_arguments = decision_state.get("controller_arguments")
        expected_argument_keys: set[str] = set()
        if (
            not isinstance(expected_arguments, dict)
            or set(expected_arguments) != expected_argument_keys
            or "options" in expected_arguments
        ):
            raise ValueError("decision-state model-owned argument contract is invalid")
        expected_controller_keys = {"strategy"} if target == "retry_solve" else set()
        if (
            not isinstance(expected_controller_arguments, dict)
            or set(expected_controller_arguments) != expected_controller_keys
            or (
                target == "retry_solve"
                and expected_controller_arguments.get("strategy") != "greedy"
            )
        ):
            raise ValueError("decision-state controller argument contract is invalid")
        supervised_options = decision_state.get("supervised_options")
        if target == "propose_tradeoff":
            authorized_options = controller_tradeoff_options(
                dict(policy_state.get("capability") or {})
            )
            if (
                decision_state.get("require_options") is not True
                or not isinstance(supervised_options, list)
                or supervised_options != authorized_options
            ):
                raise ValueError(
                    "decision-state controller options do not match replayed authority"
                )
        elif decision_state.get("require_options") is not False or supervised_options not in (
            [],
            None,
        ):
            raise ValueError("non-tradeoff decision-state contains controller options")
        rendered = json.dumps(
            transition,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self._audit("decision_replay_ready", transition=self._transition)
        if isinstance(prompt, list) and len(prompt) > 2:
            if str(prompt[-1].get("content") or "") != rendered:
                raise ValueError("decision-state replay prompt does not match replayed state")
            return ""
        return rendered

    _validate_initial_prompt = staticmethod(
        TRLReactGetPoiDetailDecisionEnvironment._validate_initial_prompt
    )

    def _complete_decision(self, action: str, arguments: dict[str, Any]) -> str:
        rendered = json.loads(self._act(action, arguments))
        rendered.pop("policy_state", None)
        rendered.update({"done": True, "decision_complete": True})
        return json.dumps(rendered, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def _trl_tool_loop_done(self) -> bool:
        # A verifier-repair replay owns exactly one model decision. Invalid
        # attempts also terminate the rollout; there is no within-turn repair.
        return self._policy_call_attempt_count >= 1

    @_tolerate_copied_schema_annotations
    def accept_itinerary(self) -> str:
        """Attempt to accept the current itinerary.

        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision("accept_itinerary", {})

    @_tolerate_copied_schema_annotations
    def ask_user(self, question: str) -> str:
        """Ask for a user-owned choice after verifier review.

        Args:
            question: One concise grounded question in the user's language.
        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision("ask_user", {"question": question})

    @_tolerate_copied_schema_annotations
    def retry_solve(self, reason: str) -> str:
        """Request a controller-bounded retry after a repairable verifier failure.

        Args:
            reason: Verifier-grounded reason for the retry.
        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision("retry_solve", {"reason": reason})

    @_tolerate_copied_schema_annotations
    def propose_tradeoff(self, reason: str) -> str:
        """Explain a conflict whose exact options are controller-owned.

        Args:
            reason: Verifier-grounded conflict.
        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision(
            "propose_tradeoff",
            {"reason": reason},
        )

    @_tolerate_copied_schema_annotations
    def abort(self, reason: str) -> str:
        """Stop when verifier evidence has no safe or feasible repair.

        Args:
            reason: Verifier-grounded reason the task cannot continue.
        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision("abort", {"reason": reason})

    @_tolerate_copied_schema_annotations
    def get_weather(self, date: str | None = None) -> str:
        """Choose weather research when it is the best review repair.

        Args:
            date: Optional grounded date in YYYY-MM-DD format.
        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision("get_weather", {"date": date} if date else {})

    @_tolerate_copied_schema_annotations
    def search_pois(self, keywords: list[str] | None = None) -> str:
        """Choose another POI search when candidates caused verifier failure.

        Args:
            keywords: Up to eight grounded preference keywords.
        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision("search_pois", {"keywords": keywords or []})

    @_tolerate_copied_schema_annotations
    def retrieve_city_knowledge(self, topic: str | None = None) -> str:
        """Choose stable city research when review evidence is incomplete.

        Args:
            topic: Optional grounded topic to narrow the lookup.
        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision(
            "retrieve_city_knowledge",
            {"topic": topic} if topic else {},
        )

    @_tolerate_copied_schema_annotations
    def get_poi_detail(self) -> str:
        """Choose another trusted POI-detail lookup after review.

        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision("get_poi_detail", {})

    @_tolerate_copied_schema_annotations
    def get_route_matrix(self) -> str:
        """Choose route-matrix refresh when review evidence requires it.

        Returns:
            A terminal observation for this one-decision training episode.
        """
        return self._complete_decision("get_route_matrix", {})

    @_tolerate_copied_schema_annotations
    def search_current_info(
        self,
        query: str,
        info_type: Literal[
            "event", "opening_hours", "restaurant", "seasonal_activity", "closure", "general"
        ] = "general",
        date: str | None = None,
    ) -> str:
        """Choose live information research when review evidence is stale.

        Args:
            query: Grounded current-information query.
            info_type: Type of current information being requested.
            date: Optional grounded date in YYYY-MM-DD format.
        Returns:
            A terminal observation for this one-decision training episode.
        """
        arguments: dict[str, Any] = {"query": query, "info_type": info_type}
        if date:
            arguments["date"] = date
        return self._complete_decision("search_current_info", arguments)

    @_tolerate_copied_schema_annotations
    def search_transport(
        self,
        mode: Literal["flight", "train", "both"] = "both",
        date: str | None = None,
    ) -> str:
        """Choose a transport refresh when review evidence requires it.

        Args:
            mode: Transport mode to search.
            date: Optional grounded departure date in YYYY-MM-DD format.
        Returns:
            A terminal observation for this one-decision training episode.
        """
        arguments: dict[str, Any] = {"mode": mode}
        if date:
            arguments["date"] = date
        return self._complete_decision("search_transport", arguments)

    def get_reward(self) -> float:
        """Score model-owned success plus verifier-decomposed partial progress.

        The controller is allowed to sanitize a bad payload before execution so
        serving stays safe.  That safety property must not become a training
        credit: a completion which attempts to supply controller-owned fields
        is a model-contract failure even when the authorized action succeeds.
        Keep the two facts separately in the audit record.
        """
        session = self._require_session()
        policy_steps = [
            step
            for step in session.recorder.episode.steps[self._decision_step_start :]
            if step.action.decision_source != "controller"
        ]
        target = str(self._decision_contract.get("target_action") or "")
        system_structural_checks: dict[str, bool] = {
            "single_policy_call": self._policy_call_attempt_count == 1,
            "no_policy_call_rejections": self._policy_call_rejection_count == 0,
        }
        model_structural_checks: dict[str, bool] = {
            "model_contract_compliant": False,
            "no_controller_override_attempt": False,
            # Tradeoff options and retry strategy are controller-injected.
            # Other routes correctly have no controller hydration to audit.
            "controller_hydration_exact": target not in {"propose_tradeoff", "retry_solve"},
        }
        checks: dict[str, bool] = {"action_match": False}
        if len(policy_steps) == 1:
            step = policy_steps[0]
            model_structural_checks["model_contract_compliant"] = bool(
                step.action.model_contract_compliant
            )
            model_structural_checks["no_controller_override_attempt"] = not bool(
                step.action.controller_override_attempt
            )
            if target in {"propose_tradeoff", "retry_solve"}:
                model_structural_checks["controller_hydration_exact"] = bool(
                    step.action.controller_hydration_exact
                ) and step.action.controller_hydrated_fields == (
                    ["options"] if target == "propose_tradeoff" else ["strategy"]
                )
            checks["action_match"] = step.action.action == target
            if checks["action_match"]:
                verification_error = step.verification.get("error_code")
                expected_abort = target == "abort" and verification_error == "POLICY_ABORT"
                checks["execution_valid"] = all(
                    observation.ok for observation in step.observations
                ) and (not verification_error or expected_abort)
                checks.update(self._argument_contract_checks(step.action.arguments))
        system_valid = all(system_structural_checks.values()) and all(checks.values())
        valid = system_valid and all(model_structural_checks.values())
        super().get_reward()
        output_contract_valid = all(system_structural_checks.values()) and all(
            model_structural_checks.values()
        )
        # Controller-argument equality is a serving-safety invariant, not a
        # model skill.  It must be required for a passing rollout through
        # ``system_valid``, but it must not inflate partial model credit merely
        # because the controller injected its own fallback strategy.
        model_scored_checks = {
            name: value
            for name, value in checks.items()
            if name != "controller_arguments_match"
        }
        score = (
            round(2 * sum(model_scored_checks.values()) / len(model_scored_checks) - 1, 6)
            if output_contract_valid and checks.get("action_match")
            else -1.0
        )
        if self._reward is None:
            raise RuntimeError("decision-state reward was not initialized")
        self._reward = self._reward.model_copy(
            update={
                "episode_reward": score,
                "terminal_reward": score,
                "components": self._reward.components.model_copy(
                    update={
                        "task": 1.0 if checks.get("action_match") else -1.0,
                        "constraint": 1.0 if valid else score,
                    }
                ),
                "gate_status": "passed" if valid else "task_failed",
                "gate_reasons": [] if valid else ["VERIFIER_REPAIR_DECISION_INVALID"],
                "audit_metrics": {
                    **self._reward.audit_metrics,
                    "decision_state_training": True,
                    # ``decision_system_step_valid`` is the serving-safety
                    # result after controller authorization.  It is useful for
                    # incident analysis but must never be used as a model
                    # reward or selector success criterion.
                    "decision_system_step_valid": system_valid,
                    "decision_step_valid": valid,
                    "decision_target_action": target,
                    "decision_policy_call_attempt_count": self._policy_call_attempt_count,
                    "decision_policy_call_rejection_count": self._policy_call_rejection_count,
                    "decision_policy_argument_rejection_count": (
                        self._policy_argument_rejection_count
                    ),
                    "verified_partial_credit": True,
                    "decision_partial_reward": score,
                    **{
                        f"decision_{name}": value
                        for name, value in system_structural_checks.items()
                    },
                    **{
                        f"decision_{name}": value
                        for name, value in model_structural_checks.items()
                    },
                    **{f"decision_{name}": value for name, value in checks.items()},
                },
            }
        )
        self._audit("decision_reward", reward=self._reward)
        return score

    def _arguments_match_contract(self, arguments: dict[str, Any]) -> bool:
        return all(self._argument_contract_checks(arguments).values())

    def _argument_contract_checks(self, arguments: dict[str, Any]) -> dict[str, bool]:
        checks: dict[str, bool] = {}
        expected = self._decision_contract.get("expected_arguments")
        if not isinstance(expected, dict):
            checks["expected_arguments_match"] = False
        else:
            checks["expected_arguments_match"] = all(
                arguments.get(key) == value for key, value in expected.items()
            )
        expected_controller = self._decision_contract.get("controller_arguments")
        if not isinstance(expected_controller, dict):
            checks["controller_arguments_match"] = False
        else:
            checks["controller_arguments_match"] = all(
                arguments.get(key) == value for key, value in expected_controller.items()
            )
        phrases = [
            str(item)
            for item in self._decision_contract.get("grounding_phrases") or []
            if str(item).strip()
        ]
        if phrases:
            # One copied substring is not a complete user-facing explanation.
            # Score all visible evidence phrases, concrete day/value anchors,
            # language alignment, public wording, and action rationale.
            checks.update(
                verifier_reason_quality_checks(
                    reason=arguments.get("reason", ""),
                    target_action=str(
                        self._decision_contract.get("target_action") or ""
                    ),
                    grounding_phrases=phrases,
                    evidence=(
                        _decision_visible_violation(self._decision_contract)
                        or "；".join(phrases)
                    ),
                )
            )
        if self._decision_contract.get("require_options") is True:
            options = arguments.get("options")
            submitted_options = (
                [str(item) for item in options if str(item).strip()]
                if isinstance(options, list)
                else []
            )
            supported_options = [
                str(item)
                for item in self._decision_contract.get("supervised_options") or []
                if str(item).strip()
            ]
            checks["options_present"] = bool(submitted_options)
            checks["option_contract_present"] = bool(supported_options)
            checks["options_supported"] = bool(submitted_options) and all(
                any(_decision_option_matches(item, expected) for expected in supported_options)
                for item in submitted_options
            )
            checks["option_contract_covered"] = bool(supported_options) and all(
                any(_decision_option_matches(item, expected) for item in submitted_options)
                for expected in supported_options
            )
        return checks


class TRLReactVerifierRetryDecisionEnvironment(
    _TRLReactVerifierRepairDecisionEnvironmentBase
):
    """Expose exactly the production retry-state review action surface."""

    _EXPECTED_TARGET = "retry_solve"
    _EXPECTED_ACTIONS = VERIFIER_REPAIR_ACTIONS_BY_ROUTE[
        "decision_verifier_repair_retry"
    ]
    accept_itinerary = None


class TRLReactVerifierTradeoffDecisionEnvironment(
    _TRLReactVerifierRepairDecisionEnvironmentBase
):
    """Expose only the two actions allowed by actionable infeasibility."""

    _EXPECTED_TARGET = "propose_tradeoff"
    _EXPECTED_ACTIONS = VERIFIER_REPAIR_ACTIONS_BY_ROUTE[
        "decision_verifier_repair_tradeoff"
    ]
    accept_itinerary = None
    ask_user = None
    get_poi_detail = None
    get_route_matrix = None
    get_weather = None
    retrieve_city_knowledge = None
    retry_solve = None
    search_current_info = None
    search_pois = None
    search_transport = None


class TRLReactVerifierAbortDecisionEnvironment(
    _TRLReactVerifierRepairDecisionEnvironmentBase
):
    """Expose only fail-closed abort for non-actionable infeasibility."""

    _EXPECTED_TARGET = "abort"
    _EXPECTED_ACTIONS = VERIFIER_REPAIR_ACTIONS_BY_ROUTE[
        "decision_verifier_repair_abort"
    ]
    accept_itinerary = None
    ask_user = None
    get_poi_detail = None
    get_route_matrix = None
    get_weather = None
    propose_tradeoff = None
    retrieve_city_knowledge = None
    retry_solve = None
    search_current_info = None
    search_pois = None
    search_transport = None


def _decision_text(value: Any) -> str:
    """Normalize multilingual evidence without exposing hidden labels to the model."""
    rendered = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return "".join(character.casefold() for character in rendered if character.isalnum())


def _decision_visible_violation(contract: dict[str, Any]) -> str:
    """Read the verifier message from the same prompt state shown to the model."""
    for message in reversed(list(contract.get("prompt_messages") or [])):
        try:
            payload = json.loads(str(message.get("content") or "{}"))
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            continue
        state = payload.get("policy_state")
        if not isinstance(state, dict):
            continue
        reports = [
            item
            for item in state.get("relevant_artifacts") or []
            if isinstance(item, dict) and item.get("artifact_type") == "validation_report"
        ]
        violations = reports[-1].get("violations") or [] if reports else []
        if violations and isinstance(violations[0], dict):
            return str(violations[0].get("message") or "").strip()
    return ""


def _decision_option_matches(submitted: str, expected: str) -> bool:
    """Match the explicit normalized option contract without substring loopholes."""
    submitted_text = _decision_text(submitted)
    expected_text = _decision_text(expected)
    if not submitted_text or not expected_text:
        return False
    return submitted_text == expected_text


def build_trl_environment_factories(
    execution_mode: GRPOExecutionMode = "policy_driven",
) -> dict[str, Callable[..., _TRLTravelEnvironmentBase]]:
    """Build route-compatible factories for a declared train/serve contract."""
    if execution_mode == "policy_driven":
        return {
            "search": TRLPolicyDrivenEnvironment,
            "search_current": TRLPolicyDrivenEnvironment,
            "search_transport": TRLPolicyDrivenEnvironment,
            "clarification": TRLPolicyDrivenEnvironment,
            "tradeoff": TRLPolicyDrivenEnvironment,
        }
    if execution_mode == "controller_first":
        return {
            "search": TRLSearchEnvironment,
            "search_current": TRLSearchEnvironment,
            "search_transport": TRLSearchEnvironment,
            "clarification": TRLClarificationEnvironment,
            "tradeoff": TRLTradeoffEnvironment,
        }
    if execution_mode == "react":
        return {
            "search": TRLReactEnvironment,
            "search_current": TRLReactCurrentInfoEnvironment,
            "search_transport": TRLReactTransportEnvironment,
            "decision_get_poi_detail": TRLReactGetPoiDetailDecisionEnvironment,
            "decision_verifier_repair_retry": TRLReactVerifierRetryDecisionEnvironment,
            "decision_verifier_repair_tradeoff": (
                TRLReactVerifierTradeoffDecisionEnvironment
            ),
            "decision_verifier_repair_abort": TRLReactVerifierAbortDecisionEnvironment,
            "clarification": TRLClarificationEnvironment,
            "tradeoff": TRLTradeoffEnvironment,
        }
    raise ValueError(f"unsupported GRPO execution mode: {execution_mode}")


# Production-aligned default. Narrow controller-first classes remain available
# only for an explicit cost/latency baseline.
TRLTravelEnvironment = TRLPolicyDrivenEnvironment
TRL_ENVIRONMENT_FACTORIES = build_trl_environment_factories("policy_driven")


__all__ = [
    "FRESH_LEDGER_ROLLOUT_CONTRACT",
    "VERIFIED_DECISION_STATE_REPLAY_CONTRACT",
    "VERIFIER_REPAIR_ACTIONS_BY_ROUTE",
    "TRLClarificationEnvironment",
    "TRLPolicyDrivenEnvironment",
    "TRLReactCurrentInfoEnvironment",
    "TRLReactEnvironment",
    "TRLReactGetPoiDetailDecisionEnvironment",
    "TRLReactVerifierAbortDecisionEnvironment",
    "TRLReactVerifierRetryDecisionEnvironment",
    "TRLReactVerifierTradeoffDecisionEnvironment",
    "TRLReactTransportEnvironment",
    "TRLSearchEnvironment",
    "TRLTradeoffEnvironment",
    "TRLTravelEnvironment",
    "TRL_ENVIRONMENT_FACTORIES",
    "build_trl_environment_factories",
    "canonical_trl_tool_schemas",
]
