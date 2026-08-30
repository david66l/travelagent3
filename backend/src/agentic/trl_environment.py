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
from agentic.policy_actions import strip_policy_schema_artifacts, validate_policy_arguments
from agentic.reward import EpisodeReward, HierarchicalRewardEngine
from agentic.grpo import policy_return_to_go_credit, policy_turn_credit_records
from agentic.runtime import initialize_agent_ledger
from agentic.state import AgentLedgerState
from evaluation.validator import VALIDATOR_VERSION


_AUDIT_LOCK = threading.Lock()
GRPOExecutionMode = Literal["policy_driven", "controller_first", "react"]
FRESH_LEDGER_ROLLOUT_CONTRACT = "fresh_ledger_no_teacher_prefix.v1"


def _tolerate_copied_schema_annotations(method: Callable[..., str]) -> Callable[..., str]:
    """Accept harmless schema annotations without changing the advertised signature."""

    @wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> str:
        sanitized = strip_policy_schema_artifacts(method.__name__, kwargs)
        return method(self, *args, **sanitized)

    return wrapped


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

    def reset(
        self,
        *,
        task: dict[str, Any],
        snapshot: dict[str, Any],
        prompt: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> str:
        """Reset one rollout from an immutable task and tool-response snapshot."""
        if self._session is not None and self._session.recorder.episode.status == "running":
            raise RuntimeError("previous rollout was not scored before environment reuse")
        if self._runner is not None:
            self._runner.close()
            self._runner = None
        parsed_task = EnvironmentTask(**task)
        # Arrow/Hugging Face Dataset materializes heterogeneous nested structs
        # with the union of all keys and inserts ``None`` for a tool absent from
        # one row. Normalize that columnar representation back to the JSON
        # contract before Pydantic validation.
        normalized_snapshot = dict(snapshot)
        normalized_snapshot["tool_responses"] = {
            name: responses or []
            for name, responses in (snapshot.get("tool_responses") or {}).items()
        }
        parsed_snapshot = EnvironmentSnapshot(**normalized_snapshot)
        self._validate_initial_prompt(prompt, user_request=parsed_task.user_request)
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
        self._transition = self._runner.run(self._session.start())
        if self._transition.done or self._transition.next_context is None:
            raise RuntimeError("environment task has no policy-owned decision")
        self._reward = None
        self._audit("reset", transition=self._transition)
        return self._render_context(self._transition.next_context)

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

    def _act(self, action: str, arguments: dict[str, Any]) -> str:
        session = self._require_session()
        validated = validate_policy_arguments(action, arguments)
        runner = self._require_runner()
        transition = runner.run(session.submit(PolicyAction(action=action, arguments=validated)))
        self._transition = transition
        self._audit(
            "action",
            transition=transition,
            submitted={"action": action, "arguments": validated},
        )
        return self._render_transition(transition)

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
            "rollout_contract": FRESH_LEDGER_ROLLOUT_CONTRACT,
            "task_id": self._task_id,
            "trajectory_id": episode.trajectory_id if episode else None,
            "episode_status": episode.status if episode else None,
            "termination_reason": episode.termination_reason if episode else None,
            "submitted": submitted,
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

    def retry_solve(self, strategy: Literal["cpsat", "greedy"], reason: str) -> str:
        """Retry the solver with a verifier-grounded alternate strategy.

        Args:
            strategy: Bounded solver strategy selected for the retry.
            reason: Verifier-grounded reason that the previous solve must be retried.
        Returns:
            The verified transition and next policy state, if any.
        """
        return self._act("retry_solve", {"strategy": strategy, "reason": reason})

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

    # Only actions owned by the production policy on the ReAct research node
    # are exposed. Solver, verifier, composition, completion, and the
    # finalize-research gate remain controller-owned and therefore cannot be
    # sampled by GRPO.
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
        if decision_state.get("target_action") != "get_poi_detail":
            raise ValueError("decision-state target does not match environment")
        for item in decision_state.get("prefix_actions") or []:
            if not isinstance(item, dict):
                raise ValueError("decision-state prefix action must be an object")
            # Hugging Face Dataset widens heterogeneous nested argument structs
            # and inserts None for keys owned by another action. Restore the
            # original sparse JSON object before replaying the verified prefix.
            prefix_arguments = {
                key: value
                for key, value in dict(item.get("arguments") or {}).items()
                if value is not None
            }
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
        prompt = kwargs.get("prompt")
        # A verified replay prompt already ends with the exact transition above.
        # TRL appends reset() output to the final message, so return an empty
        # observation to avoid duplicating the state in that contract.
        if isinstance(prompt, list) and len(prompt) > 2:
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
    "TRLClarificationEnvironment",
    "TRLPolicyDrivenEnvironment",
    "TRLReactCurrentInfoEnvironment",
    "TRLReactEnvironment",
    "TRLReactGetPoiDetailDecisionEnvironment",
    "TRLReactTransportEnvironment",
    "TRLSearchEnvironment",
    "TRLTradeoffEnvironment",
    "TRLTravelEnvironment",
    "TRL_ENVIRONMENT_FACTORIES",
    "build_trl_environment_factories",
]
