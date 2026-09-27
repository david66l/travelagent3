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
from agentic.policy import constrain_policy_context, policy_prompt_payload
from agentic.policy_actions import (
    PolicyArgumentValidationError,
    controller_override_attempt,
    model_visible_policy_actions,
    policy_action_schema,
    policy_action_schemas_for_state,
    validate_policy_arguments,
)
from agentic.reward import EpisodeReward, HierarchicalRewardEngine
from agentic.grpo import policy_return_to_go_credit, policy_turn_credit_records
from agentic.grpo_training import (
    AUTHORITY_PAYLOAD_ENCODING,
    FRESH_LEDGER_ROLLOUT_CONTRACT,
    decode_authority_payload,
)
from agentic.runtime import initialize_agent_ledger
from agentic.state import AgentLedgerState
from evaluation.validator import VALIDATOR_VERSION


_AUDIT_LOCK = threading.Lock()
GRPOExecutionMode = Literal["react"]


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
            name for name in method.__annotations__ if name not in {"return", "self"}
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
        validated = {name: value for name, value in validated.items() if name in parameter_names}
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
        actual_names = tuple(model_visible_policy_actions(actual_names, capability=capability))
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
        execution_mode: GRPOExecutionMode = "react",
    ) -> None:
        # Retired modes (policy_driven / controller_first) are archived in
        # agentic.legacy; live wiring only ever passes "react". Validation
        # lives at the factory and composition root so archived classes can
        # still be constructed verbatim for ablation reproduction.
        if execution_mode != "react":
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
        if parsed_snapshot.hidden_test_facts.get("grpo_decision_state"):
            raise ValueError("Legacy decision replay is retired; use fresh full episodes")
        self._validate_initial_prompt(prompt, user_request=parsed_task.user_request)
        self._rollout_contract = FRESH_LEDGER_ROLLOUT_CONTRACT
        self._task = parsed_task
        self._snapshot = parsed_snapshot
        self._task_id = parsed_task.task_id
        # H-002: pin before the ledger initializes — the react task graph
        # materializes declared research requirements (weather window) here.
        # The ledger initializes on the caller thread, so the frozen context
        # must be entered directly (contextvars do not cross threads).
        self._frozen_moment: datetime | None = self._frozen_reference_moment()
        with frozen_reference_time(self._frozen_moment):
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
        )
        self._runner = _SessionLoopThread()
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
                    if key in {
                        "queried_at",
                        "retrieved_at",
                        "created_at",
                        "generated_at",
                    } and isinstance(value, str):
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
            dict(pending.get("model_arguments") or {}) if pending is not None else dict(arguments)
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
                    bool(pending.get("model_contract_compliant")) if pending is not None else True
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
                action.model_arguments if action.model_arguments is not None else action.arguments
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


class TRLAgentEnvironment(_TRLTravelEnvironmentBase):
    """Full-episode adapter for the exact same loop used online; no automatic decisions."""

    def abort(self, reason: str) -> str:
        """Stop when the task cannot continue safely or feasibly.

        Args:
            reason: Grounded reason the task cannot continue.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("abort", {"reason": reason})

    def ask_user(self, question: str) -> str:
        """Ask one grounded question for information only the user can provide.

        Args:
            question: One concise question for the missing information.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("ask_user", {"question": question})

    def finish(self) -> str:
        """Present the verified draft and wait for user confirmation.

        Returns:
            The terminal confirmation transition.
        """
        return self._act("finish", {})

    def propose_tradeoff(self, reason: str) -> str:
        """Explain a constraint conflict and offer the authorized alternatives.

        Args:
            reason: Grounded constraint conflict in the user's language.
        Returns:
            The transition awaiting the user's choice.
        """
        return self._act("propose_tradeoff", {"reason": reason})

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
        strategy: Literal["auto", "cpsat", "greedy"] = "cpsat",
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


# Use the same strict argument validator and raw-output audit for every exposed method.
from agentic.policy_actions import POLICY_ACTION_MODELS

for _action_name in POLICY_ACTION_MODELS:
    setattr(
        TRLAgentEnvironment,
        _action_name,
        _tolerate_copied_schema_annotations(getattr(TRLAgentEnvironment, _action_name)),
    )


def build_trl_environment_factories(execution_mode: GRPOExecutionMode = "react"):
    if execution_mode != "react":
        raise ValueError("Only agent-harness-v1 is supported; regenerate legacy training corpora")
    return {"travel": TRLAgentEnvironment}


__all__ = ["TRLAgentEnvironment", "build_trl_environment_factories", "canonical_trl_tool_schemas"]
