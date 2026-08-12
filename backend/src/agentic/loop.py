"""Bounded, evidence-gated runtime shared by API and future local policies."""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic.observations import ObservationEnvelope
from agentic.scheduler import ScheduledBatch, TaskScheduler
from agentic.state import (
    AgentLedgerState,
    ArtifactRecord,
    BudgetExceeded,
    FactRecord,
    FailureRecord,
    StateTransitionError,
    TaskGraphController,
    TaskNode,
)
from agentic.termination import CompletionGuard
from agentic.verifier import SubtaskVerifier


NO_TOOL_ACTIONS = frozenset(
    {"abort", "ask_user", "capability_check", "compose_draft", "finish", "propose_tradeoff"}
)


class PolicyContext(BaseModel):
    trajectory_id: str
    goal_version: int
    plan_version: int
    original_request: str
    current_subtask: dict[str, Any]
    hard_constraints: dict[str, Any]
    soft_preferences: dict[str, Any]
    relevant_fact_refs: list[str]
    relevant_artifact_refs: list[str]
    relevant_facts: list[dict[str, Any]] = Field(default_factory=list)
    relevant_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    failure_summary: list[dict[str, Any]]
    remaining_tasks: int
    remaining_steps: int
    allowed_actions: list[str]


class PolicyAction(BaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    token_usage: int = Field(default=0, ge=0)


class ActionOutcome(BaseModel):
    status: Literal["completed", "failed", "awaiting_user"] = "completed"
    observations: list[ObservationEnvelope] = Field(default_factory=list)
    facts: list[FactRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False


class AgentPolicy(Protocol):
    async def propose(self, context: PolicyContext) -> PolicyAction: ...


class ActionExecutor(Protocol):
    async def execute(
        self,
        *,
        task: TaskNode,
        action: PolicyAction,
        ledger: AgentLedgerState,
    ) -> ActionOutcome: ...


class AgentLoopEvent(BaseModel):
    sequence: int
    event_type: str
    task_id: str | None = None
    action_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentLoopResult(BaseModel):
    ledger: AgentLedgerState
    status: Literal["finished", "interrupted", "failed"]
    termination_reason: str
    events: list[AgentLoopEvent]


class EpisodeRecorderProtocol(Protocol):
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
    ) -> None: ...

    def finalize(self, result: AgentLoopResult) -> Any: ...


class _TaskExecution(BaseModel):
    task_id: str
    action: PolicyAction
    outcome: ActionOutcome


class BoundedAgentLoop:
    """Run tasks until verified completion, interruption or a durable limit."""

    def __init__(
        self,
        *,
        scheduler: TaskScheduler | None = None,
        verifier: SubtaskVerifier | None = None,
        completion_guard: CompletionGuard | None = None,
    ) -> None:
        self.scheduler = scheduler or TaskScheduler()
        self.verifier = verifier or SubtaskVerifier()
        self.completion_guard = completion_guard or CompletionGuard(mode="enforce")
        self.controller = TaskGraphController()

    async def run(
        self,
        ledger: AgentLedgerState,
        *,
        policy: AgentPolicy,
        executor: ActionExecutor,
        recorder: EpisodeRecorderProtocol | None = None,
    ) -> AgentLoopResult:
        events: list[AgentLoopEvent] = []
        while True:
            ledger.task_graph, batch = self.scheduler.select(ledger.task_graph)
            if batch is None:
                return self._record_final(self._terminal_result(ledger, events), recorder)

            try:
                ledger.task_graph = self.scheduler.start(ledger.task_graph, batch)
                contexts = [
                    self._policy_context(ledger, ledger.task_graph.get(task_id))
                    for task_id in batch.task_ids
                ]
                proposals = await asyncio.gather(*(policy.propose(context) for context in contexts))
                ledger.budget = ledger.budget.consume(
                    episode_steps=len(proposals),
                    tool_calls=sum(
                        proposal.action not in NO_TOOL_ACTIONS for proposal in proposals
                    ),
                    solver_calls=sum(
                        proposal.action == "solve_itinerary" for proposal in proposals
                    ),
                    tokens=sum(proposal.token_usage for proposal in proposals),
                )
            except BudgetExceeded as exc:
                return self._record_final(
                    self._finish(
                        ledger,
                        events,
                        "failed",
                        "budget_exhausted_fallback",
                        error=str(exc),
                    ),
                    recorder,
                )

            state_before = ledger.model_copy(deep=True)
            executions = await asyncio.gather(
                *(
                    self._execute(
                        ledger=ledger,
                        task=ledger.task_graph.get(task_id),
                        action=action,
                        executor=executor,
                    )
                    for task_id, action in zip(batch.task_ids, proposals, strict=True)
                )
            )
            try:
                interrupt = self._commit_batch(ledger, batch, executions, events)
            except StateTransitionError as exc:
                return self._record_final(
                    self._finish(
                        ledger,
                        events,
                        "failed",
                        "stale_or_invalid_state",
                        error=str(exc),
                    ),
                    recorder,
                )
            if recorder is not None:
                state_after = ledger.model_copy(deep=True)
                for context, execution in zip(contexts, executions, strict=True):
                    recorder.record_step(
                        task_id=execution.task_id,
                        context=context,
                        action=execution.action,
                        observations=execution.outcome.observations,
                        verification={
                            "task_status": ledger.task_graph.get(execution.task_id).status,
                            "error_code": execution.outcome.error_code,
                        },
                        state_before=state_before,
                        state_after=state_after,
                    )
            if interrupt:
                return self._record_final(
                    self._finish(ledger, events, "interrupted", interrupt), recorder
                )

    async def _execute(
        self,
        *,
        ledger: AgentLedgerState,
        task: TaskNode,
        action: PolicyAction,
        executor: ActionExecutor,
    ) -> _TaskExecution:
        if action.action not in task.allowed_actions:
            return _TaskExecution(
                task_id=task.task_id,
                action=action,
                outcome=ActionOutcome(
                    status="failed",
                    error_code="ACTION_NOT_ALLOWED",
                    error_message=f"{action.action} is not allowed for {task.task_id}",
                ),
            )
        if action.action == "abort":
            return _TaskExecution(
                task_id=task.task_id,
                action=action,
                outcome=ActionOutcome(
                    status="failed",
                    error_code="POLICY_ABORT",
                    error_message=str(action.arguments.get("reason") or "policy aborted"),
                ),
            )
        try:
            outcome = await executor.execute(task=task, action=action, ledger=ledger)
        except Exception as exc:  # executor boundary must isolate provider failures
            outcome = ActionOutcome(
                status="failed",
                observations=[
                    ObservationEnvelope.failure(
                        tool=action.action,
                        code="EXECUTOR_ERROR",
                        message=str(exc),
                        retryable=True,
                        tool_call_id=action.action_id,
                    )
                ],
                error_code="EXECUTOR_ERROR",
                error_message=str(exc),
                retryable=True,
            )
        return _TaskExecution(task_id=task.task_id, action=action, outcome=outcome)

    def _commit_batch(
        self,
        ledger: AgentLedgerState,
        batch: ScheduledBatch,
        executions: list[_TaskExecution],
        events: list[AgentLoopEvent],
    ) -> str | None:
        """Join barrier: validate all outputs, then commit state serially."""
        for execution in executions:
            self._assert_current_versions(ledger, execution.outcome)

        for execution in executions:
            task = ledger.task_graph.get(execution.task_id)
            outcome = execution.outcome
            self._event(
                events,
                "action_completed",
                task_id=task.task_id,
                action_id=execution.action.action_id,
                payload={"action": execution.action.action, "status": outcome.status},
            )

            if outcome.status == "awaiting_user":
                ledger.task_graph = self.controller.transition(
                    ledger.task_graph, task.task_id, "blocked"
                )
                return "awaiting_user"

            if outcome.status == "failed":
                self._record_failure(ledger, execution)
                ledger.task_graph = self.controller.retry_or_fail(
                    ledger.task_graph,
                    task.task_id,
                    {
                        "code": outcome.error_code or "ACTION_FAILED",
                        "message": outcome.error_message or "action failed",
                    },
                )
                continue

            for fact in outcome.facts:
                ledger.facts[fact.fact_id] = fact
            for artifact in outcome.artifacts:
                ledger.artifacts[artifact.artifact_id] = artifact

            verification = self.verifier.verify(
                task,
                facts=ledger.facts,
                artifacts=ledger.artifacts,
                observations=outcome.observations,
            )
            if verification.passed:
                ledger.task_graph = self.controller.transition(
                    ledger.task_graph,
                    task.task_id,
                    "succeeded",
                    evidence_refs=verification.evidence_refs,
                )
                self._event(
                    events,
                    "task_verified",
                    task_id=task.task_id,
                    payload={"evidence_refs": verification.evidence_refs},
                )
            else:
                failed = _TaskExecution(
                    task_id=task.task_id,
                    action=execution.action,
                    outcome=ActionOutcome(
                        status="failed",
                        error_code="SUBTASK_VERIFICATION_FAILED",
                        error_message=", ".join(verification.failure_codes),
                        retryable=True,
                    ),
                )
                self._record_failure(ledger, failed)
                ledger.task_graph = self.controller.retry_or_fail(
                    ledger.task_graph,
                    task.task_id,
                    {
                        "code": "SUBTASK_VERIFICATION_FAILED",
                        "failure_codes": verification.failure_codes,
                    },
                )
        return None

    @staticmethod
    def _assert_current_versions(ledger: AgentLedgerState, outcome: ActionOutcome) -> None:
        for item in [*outcome.facts, *outcome.artifacts]:
            if item.goal_version != ledger.goal.goal_version:
                raise StateTransitionError("outcome targets a stale goal version")
            if item.plan_version != ledger.task_graph.plan_version:
                raise StateTransitionError("outcome targets a stale plan version")

    @staticmethod
    def _record_failure(ledger: AgentLedgerState, execution: _TaskExecution) -> None:
        outcome = execution.outcome
        ledger.failures.append(
            FailureRecord(
                task_id=execution.task_id,
                action_id=execution.action.action_id,
                code=outcome.error_code or "ACTION_FAILED",
                message=outcome.error_message or "action failed",
                retryable=outcome.retryable,
                evidence_refs=[
                    item.tool_call_id
                    for item in outcome.observations
                    if item.tool_call_id is not None
                ],
                attempted_strategy=execution.action.action,
            )
        )

    def _terminal_result(
        self, ledger: AgentLedgerState, events: list[AgentLoopEvent]
    ) -> AgentLoopResult:
        report_artifacts = [
            artifact
            for artifact in ledger.artifacts.values()
            if artifact.artifact_type == "validation_report"
            and artifact.goal_version == ledger.goal.goal_version
            and artifact.plan_version == ledger.task_graph.plan_version
        ]
        report = report_artifacts[-1].payload if report_artifacts else None
        decision = self.completion_guard.evaluate(report, ledger=ledger)
        if decision.allowed:
            return self._finish(ledger, events, "finished", "validated_finish")
        statuses = {task.status for task in ledger.task_graph.tasks}
        if "blocked" in statuses:
            return self._finish(ledger, events, "interrupted", "awaiting_user")
        if "failed" in statuses:
            return self._finish(ledger, events, "failed", "unsolvable_constraints")
        return self._finish(
            ledger,
            events,
            "failed",
            "partial_finish",
            error=", ".join(block.code for block in decision.blocks),
        )

    @staticmethod
    def _policy_context(ledger: AgentLedgerState, task: TaskNode) -> PolicyContext:
        current_facts = [
            fact
            for fact in ledger.facts.values()
            if fact.goal_version == ledger.goal.goal_version
            and fact.plan_version == ledger.task_graph.plan_version
        ]
        current_artifacts = [
            artifact
            for artifact in ledger.artifacts.values()
            if artifact.goal_version == ledger.goal.goal_version
            and artifact.plan_version == ledger.task_graph.plan_version
        ]
        relevant_facts = [fact.fact_id for fact in current_facts if fact.key in task.required_facts]
        failures = [
            failure.model_dump(mode="json")
            for failure in ledger.failures[-3:]
            if failure.task_id == task.task_id
        ]
        remaining = sum(
            item.required and item.status not in {"succeeded", "skipped"}
            for item in ledger.task_graph.tasks
        )
        return PolicyContext(
            trajectory_id=ledger.trajectory_id,
            goal_version=ledger.goal.goal_version,
            plan_version=ledger.task_graph.plan_version,
            original_request=ledger.goal.original_request,
            current_subtask=task.model_dump(mode="json"),
            hard_constraints=ledger.goal.hard_constraints,
            soft_preferences=ledger.goal.soft_preferences,
            relevant_fact_refs=relevant_facts,
            relevant_artifact_refs=[artifact.artifact_id for artifact in current_artifacts],
            relevant_facts=[
                {
                    "fact_id": fact.fact_id,
                    "key": fact.key,
                    "value": _compact_value(fact.value),
                    "source": fact.source,
                    "confidence": fact.confidence,
                }
                for fact in current_facts[-8:]
            ],
            relevant_artifacts=[_artifact_summary(artifact) for artifact in current_artifacts[-8:]],
            failure_summary=failures,
            remaining_tasks=remaining,
            remaining_steps=ledger.budget.remaining_episode_steps,
            allowed_actions=list(task.allowed_actions),
        )

    @staticmethod
    def _event(
        events: list[AgentLoopEvent],
        event_type: str,
        *,
        task_id: str | None = None,
        action_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        events.append(
            AgentLoopEvent(
                sequence=len(events) + 1,
                event_type=event_type,
                task_id=task_id,
                action_id=action_id,
                payload=payload or {},
            )
        )

    def _finish(
        self,
        ledger: AgentLedgerState,
        events: list[AgentLoopEvent],
        status: Literal["finished", "interrupted", "failed"],
        reason: str,
        *,
        error: str | None = None,
    ) -> AgentLoopResult:
        ledger.termination_reason = reason
        self._event(
            events,
            "episode_terminated",
            payload={"status": status, "reason": reason, "error": error},
        )
        return AgentLoopResult(
            ledger=ledger,
            status=status,
            termination_reason=reason,
            events=events,
        )

    @staticmethod
    def _record_final(
        result: AgentLoopResult, recorder: EpisodeRecorderProtocol | None
    ) -> AgentLoopResult:
        if recorder is not None:
            recorder.finalize(result)
        return result


def _artifact_summary(artifact: ArtifactRecord) -> dict[str, Any]:
    """Expose useful evidence to the policy without replaying large payloads."""
    payload = artifact.payload
    summary: dict[str, Any] = {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type,
    }
    if artifact.artifact_type == "poi_candidate_set":
        pois = payload.get("pois") or []
        summary["poi_count"] = len(pois)
        summary["poi_names"] = [
            str(item.get("name"))
            for item in pois[:10]
            if isinstance(item, dict) and item.get("name")
        ]
    elif artifact.artifact_type == "poi_detail_set":
        details = payload.get("details") or []
        summary["detail_count"] = len(details)
        summary["expected_count"] = payload.get("expected_count")
    elif artifact.artifact_type == "route_matrix":
        matrix = payload.get("time_minutes") or []
        summary["matrix_rows"] = len(matrix)
        summary["matrix_columns"] = len(matrix[0]) if matrix else 0
        summary["poi_ids"] = _compact_value(payload.get("poi_ids") or [])
    elif artifact.artifact_type == "solver_result":
        summary.update(
            {
                "status": payload.get("status"),
                "day_count": len(payload.get("days") or []),
                "solve_time_ms": payload.get("solve_time_ms"),
                "message": _compact_value(payload.get("message")),
            }
        )
    elif artifact.artifact_type == "validation_report":
        summary.update(
            {
                "hard_pass": payload.get("hard_pass"),
                "violation_codes": [
                    item.get("code")
                    for item in (payload.get("hard_violations") or [])[:10]
                    if isinstance(item, dict)
                ],
                "soft_scores": _compact_value(payload.get("soft_scores") or {}),
            }
        )
    elif artifact.artifact_type == "weather_snapshot":
        days = payload.get("days") or payload.get("data") or []
        summary["day_count"] = len(days) if isinstance(days, list) else 0
        summary["conditions"] = [
            item.get("condition") for item in days[:7] if isinstance(item, dict)
        ]
    else:
        summary["payload"] = _compact_value(payload)
    return summary


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    """Deterministically bound policy projections by depth, width and text size."""
    if depth >= 3:
        return "[TRUNCATED]"
    if isinstance(value, str):
        return value if len(value) <= 240 else value[:237] + "..."
    if isinstance(value, dict):
        items = list(value.items())[:12]
        compact = {str(key): _compact_value(item, depth=depth + 1) for key, item in items}
        if len(value) > len(items):
            compact["_truncated_fields"] = len(value) - len(items)
        return compact
    if isinstance(value, (list, tuple)):
        items = list(value)[:12]
        compact_items = [_compact_value(item, depth=depth + 1) for item in items]
        if len(value) > len(items):
            compact_items.append({"_truncated_items": len(value) - len(items)})
        return compact_items
    return value
