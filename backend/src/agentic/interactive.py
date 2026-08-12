"""Interactive driver for the production BoundedAgentLoop.

Training and evaluation code can submit one policy action at a time while the
same online scheduler, executor, verifier, budget and recorder remain in
control.  No second RL-only state machine is introduced.
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from agentic.loop import (
    ActionExecutor,
    AgentLoopResult,
    BoundedAgentLoop,
    PolicyAction,
    PolicyContext,
)
from agentic.scheduler import TaskScheduler
from agentic.state import AgentLedgerState
from agentic.trajectory import AgentEpisode, EpisodeRecorder, TrajectoryStep


class InteractiveTransition(BaseModel):
    done: bool
    next_context: PolicyContext | None = None
    committed_step: TrajectoryStep | None = None
    status: str | None = None
    termination_reason: str | None = None
    episode: AgentEpisode | None = None


class _InteractivePolicy:
    def __init__(self) -> None:
        self.requests: asyncio.Queue[tuple[PolicyContext, asyncio.Future[PolicyAction]]] = (
            asyncio.Queue()
        )

    async def propose(self, context: PolicyContext) -> PolicyAction:
        future: asyncio.Future[PolicyAction] = asyncio.get_running_loop().create_future()
        await self.requests.put((context, future))
        return await future


class InteractiveAgentSession:
    """Drive exactly one isolated episode with externally generated actions."""

    def __init__(
        self,
        ledger: AgentLedgerState,
        *,
        executor: ActionExecutor,
        environment_version: str,
        validator_version: str,
        policy_name: str,
        policy_version: str,
    ) -> None:
        self.ledger = ledger.model_copy(deep=True)
        self.executor = executor
        self.policy = _InteractivePolicy()
        self.recorder = EpisodeRecorder(
            self.ledger,
            environment_version=environment_version,
            validator_version=validator_version,
            policy_name=policy_name,
            policy_version=policy_version,
        )
        self.loop = BoundedAgentLoop(scheduler=TaskScheduler(max_parallel_tasks=1))
        self._run_task: asyncio.Task[AgentLoopResult] | None = None
        self._pending_action: asyncio.Future[PolicyAction] | None = None
        self._closed = False
        self._committed_steps = 0

    async def start(self) -> InteractiveTransition:
        if self._run_task is not None:
            raise RuntimeError("interactive session already started")
        self._run_task = asyncio.create_task(
            self.loop.run(
                self.ledger,
                policy=self.policy,
                executor=self.executor,
                recorder=self.recorder,
            )
        )
        return await self._next_transition()

    async def submit(self, action: PolicyAction | dict[str, Any]) -> InteractiveTransition:
        if self._run_task is None:
            raise RuntimeError("interactive session has not started")
        if self._pending_action is None or self._pending_action.done():
            raise RuntimeError("no policy action is currently requested")
        parsed = action if isinstance(action, PolicyAction) else PolicyAction(**action)
        self._pending_action.set_result(parsed)
        self._pending_action = None
        return await self._next_transition()

    async def aclose(self) -> None:
        self._closed = True
        if self._pending_action is not None and not self._pending_action.done():
            self._pending_action.cancel()
        if self._run_task is not None and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except asyncio.CancelledError:
                pass

    async def _next_transition(self) -> InteractiveTransition:
        if self._closed:
            raise RuntimeError("interactive session is closed")
        assert self._run_task is not None

        if not self.policy.requests.empty():
            return self._prompt_transition(await self.policy.requests.get())
        if self._run_task.done():
            return self._terminal_transition(await self._run_task)

        request_task = asyncio.create_task(self.policy.requests.get())
        done, _ = await asyncio.wait(
            {request_task, self._run_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if request_task in done:
            return self._prompt_transition(request_task.result())
        request_task.cancel()
        try:
            await request_task
        except asyncio.CancelledError:
            pass
        return self._terminal_transition(await self._run_task)

    def _prompt_transition(
        self,
        request: tuple[PolicyContext, asyncio.Future[PolicyAction]],
    ) -> InteractiveTransition:
        context, future = request
        self._pending_action = future
        return InteractiveTransition(
            done=False,
            next_context=context,
            committed_step=self._latest_new_step(),
        )

    def _terminal_transition(self, result: AgentLoopResult) -> InteractiveTransition:
        return InteractiveTransition(
            done=True,
            committed_step=self._latest_new_step(),
            status=result.status,
            termination_reason=result.termination_reason,
            episode=self.recorder.episode,
        )

    def _latest_new_step(self) -> TrajectoryStep | None:
        if len(self.recorder.episode.steps) <= self._committed_steps:
            return None
        step = self.recorder.episode.steps[-1]
        self._committed_steps = len(self.recorder.episode.steps)
        return step
