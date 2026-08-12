"""TRL environment adapter backed by the production interactive Agent Loop."""

from __future__ import annotations

import json
from typing import Any, Literal

from agentic.action_executor import TravelActionExecutor
from agentic.environment import EnvironmentSnapshot, EnvironmentTask, SnapshotToolExecutor
from agentic.interactive import InteractiveAgentSession, InteractiveTransition
from agentic.loop import AgentLoopResult, BoundedAgentLoop, PolicyAction
from agentic.policy import policy_prompt_payload
from agentic.policy_actions import validate_policy_arguments
from agentic.reward import EpisodeReward, HierarchicalRewardEngine
from agentic.runtime import initialize_agent_ledger
from agentic.scheduler import TaskScheduler
from agentic.state import AgentLedgerState
from evaluation.validator import VALIDATOR_VERSION


class TRLTravelEnvironment:
    """Stateful, snapshot-only environment compatible with TRL ``environment_factory``.

    TRL exposes every public method except ``reset`` and ``get_reward`` as a
    model-callable tool. The production task graph remains authoritative: a
    method call that is not allowed for the current state is recorded as a
    failed action and cannot mutate trusted facts or artifacts.
    """

    def __init__(self) -> None:
        self._session: InteractiveAgentSession | None = None
        self._initial_context = None
        self._started = False
        self._transition: InteractiveTransition | None = None
        self._reward: EpisodeReward | None = None
        self._reward_engine = HierarchicalRewardEngine()

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
        parsed_task = EnvironmentTask(**task)
        parsed_snapshot = EnvironmentSnapshot(**snapshot)
        initialized = initialize_agent_ledger(
            {
                "user_input": parsed_task.user_request,
                "slots": parsed_task.slots,
                "profile": parsed_task.profile,
                "missing_slots": parsed_task.missing_slots,
                "feasibility_report": parsed_task.feasibility_report,
            },
            mode="agent",
        )
        ledger = AgentLedgerState(**initialized["agent_ledger"])
        backend = SnapshotToolExecutor(parsed_snapshot)
        self._session = InteractiveAgentSession(
            ledger,
            executor=TravelActionExecutor(backend),  # type: ignore[arg-type]
            environment_version=parsed_snapshot.environment_version,
            validator_version=VALIDATOR_VERSION,
            policy_name="trl-grpo-policy",
            policy_version="online-rollout",
        )
        graph, batch = TaskScheduler(max_parallel_tasks=1).select(ledger.task_graph)
        if batch is None:
            raise RuntimeError("environment task has no initial policy decision")
        ledger.task_graph = graph
        self._initial_context = BoundedAgentLoop._policy_context(  # noqa: SLF001
            ledger, ledger.task_graph.get(batch.task_ids[0])
        )
        self._started = False
        self._transition = None
        self._reward = None
        return self._render_context(self._initial_context)

    async def capability_check(self) -> str:
        """Record the controller-computed capability assessment.

        Returns:
            The verified transition and next bounded policy state.
        """
        return await self._act("capability_check", {})

    async def ask_user(self, question: str) -> str:
        """Ask for information or confirmation only the user can provide.

        Args:
            question: One concise, grounded question.

        Returns:
            The terminal clarification transition.
        """
        return await self._act("ask_user", {"question": question})

    async def propose_tradeoff(self, reason: str, options: list[str] | None = None) -> str:
        """Offer grounded alternatives when constraints conflict.

        Args:
            reason: The grounded constraint conflict or unavailable capability.
            options: Up to three grounded alternatives.

        Returns:
            The terminal tradeoff transition.
        """
        return await self._act("propose_tradeoff", {"reason": reason, "options": options or []})

    async def abort(self, reason: str) -> str:
        """Stop an unsafe, unsupported, or infeasible task.

        Args:
            reason: The grounded reason the task cannot continue.

        Returns:
            The failed terminal transition.
        """
        return await self._act("abort", {"reason": reason})

    async def get_weather(self, date: str | None = None) -> str:
        """Read weather for the controller-owned destination.

        Args:
            date: Optional grounded date in YYYY-MM-DD format.

        Returns:
            The snapshot observation and next bounded policy state.
        """
        return await self._act("get_weather", {"date": date} if date else {})

    async def search_pois(
        self,
        keywords: list[str] | None = None,
        category: Literal["attraction", "restaurant", "hotel", "shopping"] | None = None,
    ) -> str:
        """Search POIs in the controller-owned destination.

        Args:
            keywords: Grounded preference keywords.
            category: Optional POI category filter.

        Returns:
            The snapshot observation and next bounded policy state.
        """
        arguments: dict[str, Any] = {"keywords": keywords or []}
        if category is not None:
            arguments["category"] = category
        return await self._act("search_pois", arguments)

    async def get_poi_detail(self) -> str:
        """Collect details for the controller-selected POI candidates.

        Returns:
            The snapshot observations and next bounded policy state.
        """
        return await self._act("get_poi_detail", {})

    async def get_route_matrix(self) -> str:
        """Build a route matrix from trusted candidate artifacts.

        Returns:
            The snapshot observation and next bounded policy state.
        """
        return await self._act("get_route_matrix", {})

    async def solve_itinerary(self, strategy: Literal["auto", "cpsat", "greedy"] = "auto") -> str:
        """Run deterministic solving over controller-owned artifacts.

        Args:
            strategy: Solver strategy; use auto unless evidence justifies an override.

        Returns:
            The solver observation and next bounded policy state.
        """
        return await self._act("solve_itinerary", {"strategy": strategy})

    async def validate_itinerary(self) -> str:
        """Run programmatic hard-constraint validation.

        Returns:
            The validation observation and next bounded policy state.
        """
        return await self._act("validate_itinerary", {})

    async def compose_draft(self) -> str:
        """Project a draft only from verified solver artifacts.

        Returns:
            The draft transition and next bounded policy state.
        """
        return await self._act("compose_draft", {})

    async def finish(self) -> str:
        """Present the verified draft and wait for user confirmation.

        Returns:
            The terminal transition.
        """
        return await self._act("finish", {})

    async def get_reward(self) -> float:
        """Return the gated six-component trajectory reward."""
        session = self._require_session()
        episode = session.recorder.episode
        if episode.status == "running":
            await session.aclose()
            session.recorder.finalize(
                AgentLoopResult(
                    ledger=session.ledger,
                    status="failed",
                    termination_reason="rollout_truncated",
                    events=[],
                )
            )
        self._reward = self._reward_engine.score(session.recorder.episode)
        return self._reward.episode_reward

    @property
    def reward_record(self) -> EpisodeReward | None:
        """Expose the auditable breakdown to logging callbacks, never to the model."""
        return self._reward

    async def _act(self, action: str, arguments: dict[str, Any]) -> str:
        session = self._require_session()
        if not self._started:
            started = await session.start()
            self._started = True
            if started.done:
                self._transition = started
                return self._render_transition(started)
        validated = validate_policy_arguments(action, arguments)
        transition = await session.submit(PolicyAction(action=action, arguments=validated))
        self._transition = transition
        return self._render_transition(transition)

    def _require_session(self) -> InteractiveAgentSession:
        if self._session is None:
            raise RuntimeError("environment must be reset before use")
        return self._session

    @staticmethod
    def _render_context(context: Any) -> str:
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
            payload["policy_state"] = policy_prompt_payload(transition.next_context)
        if transition.done:
            payload["status"] = transition.status
            payload["termination_reason"] = transition.termination_reason
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = ["TRLTravelEnvironment"]
