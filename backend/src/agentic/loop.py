"""Bounded, evidence-gated runtime shared by API and future local policies."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic.observations import ObservationEnvelope
from agentic.scheduler import ScheduledBatch, TaskScheduler
from agentic.state import (
    AgentLedgerState,
    ArtifactRecord,
    BudgetExceeded,
    DecisionRecord,
    FactRecord,
    FailureRecord,
    PlanVersion,
    StateTransitionError,
    TaskGraphController,
    TaskNode,
)
from agentic.termination import CompletionGuard
from agentic.verifier import SubtaskVerifier
from core.inference_metrics import InferenceMetrics


NO_TOOL_ACTIONS = frozenset(
    {
        "abort",
        "accept_candidates",
        "accept_itinerary",
        "ask_user",
        "capability_check",
        "compose_draft",
        "finish",
        "finalize_research",
        "propose_tradeoff",
        "retry_solve",
    }
)


class PolicyContext(BaseModel):
    trajectory_id: str
    goal_version: int
    plan_version: int
    original_request: str
    current_subtask: dict[str, Any]
    hard_constraints: dict[str, Any]
    soft_preferences: dict[str, Any]
    capability: dict[str, Any] = Field(default_factory=dict)
    missing_information: list[str] = Field(default_factory=list)
    relevant_fact_refs: list[str]
    relevant_artifact_refs: list[str]
    relevant_facts: list[dict[str, Any]] = Field(default_factory=list)
    relevant_artifacts: list[dict[str, Any]] = Field(default_factory=list)
    failure_summary: list[dict[str, Any]]
    decision_history: list[dict[str, Any]] = Field(default_factory=list)
    remaining_tasks: int
    remaining_steps: int
    allowed_actions: list[str]
    policy_feedback: list[dict[str, Any]] = Field(default_factory=list)


class PolicyRouteTrace(BaseModel):
    """Per-decision routing evidence persisted with the agent trajectory."""

    requested_target: Literal["student", "teacher"]
    executed_target: Literal["student", "teacher"]
    family: Literal["clarification", "search", "recovery", "tradeoff", "complex"]
    reason: str
    fallback_used: bool = False
    fallback_error_code: str | None = None


class PolicyShadowTrace(BaseModel):
    """Non-authoritative challenger decision persisted for Shadow analysis."""

    candidate_model: str
    status: Literal["completed", "failed"]
    action: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    token_usage: int = Field(default=0, ge=0)
    inference_metrics: InferenceMetrics | None = None
    route_trace: PolicyRouteTrace | None = None
    error_code: str | None = None


class PolicyAction(BaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    token_usage: int = Field(default=0, ge=0)
    decision_source: Literal["policy", "controller"] = "policy"
    inference_metrics: InferenceMetrics | None = None
    route_trace: PolicyRouteTrace | None = None
    shadow_trace: PolicyShadowTrace | None = None
    repair_attempts: int = Field(default=0, ge=0)
    repair_error_codes: list[str] = Field(default_factory=list)


class ActionOutcome(BaseModel):
    status: Literal["completed", "failed", "awaiting_user"] = "completed"
    observations: list[ObservationEnvelope] = Field(default_factory=list)
    facts: list[FactRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    tool_calls_used: int = Field(default=0, ge=0)
    loop_control: Literal["continue", "replan_local", "replan_global"] | None = None


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
    status: Literal["running", "finished", "interrupted", "failed"]
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
        policy_latency_ms: int = 0,
        action_latency_ms: int = 0,
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
        max_batches: int | None = None,
    ) -> AgentLoopResult:
        if max_batches is not None and max_batches < 1:
            raise ValueError("max_batches must be positive when supplied")
        events: list[AgentLoopEvent] = []
        completed_batches = 0
        while True:
            batch_started = time.monotonic()
            ledger.task_graph, batch = self.scheduler.select(ledger.task_graph)
            if batch is None:
                return self._record_final(self._terminal_result(ledger, events), recorder)

            try:
                ledger.task_graph = self.scheduler.start(ledger.task_graph, batch)
                contexts = [
                    self._policy_context(ledger, ledger.task_graph.get(task_id))
                    for task_id in batch.task_ids
                ]
                policy_started = time.monotonic()
                remaining_seconds = max(
                    0.001,
                    (ledger.budget.timeout_ms - ledger.budget.used_latency_ms) / 1000,
                )
                async with asyncio.timeout(remaining_seconds):
                    proposals = await asyncio.gather(
                        *(policy.propose(context) for context in contexts)
                    )
                policy_latency_ms = int((time.monotonic() - policy_started) * 1000)
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
            except TimeoutError:
                return self._record_final(
                    self._finish(
                        ledger,
                        events,
                        "failed",
                        "agent_deadline_exceeded",
                        error="policy inference exceeded the remaining episode deadline",
                    ),
                    recorder,
                )
            except Exception as exc:  # policy boundary isolates model/provider failures
                return self._record_final(
                    self._finish(
                        ledger,
                        events,
                        "failed",
                        "policy_error_fallback",
                        error=f"{type(exc).__name__}: {exc}",
                    ),
                    recorder,
                )

            state_before = ledger.model_copy(deep=True)
            action_started = time.monotonic()
            try:
                remaining_seconds = max(
                    0.001,
                    (ledger.budget.timeout_ms - ledger.budget.used_latency_ms) / 1000,
                )
                async with asyncio.timeout(remaining_seconds):
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
            except TimeoutError:
                return self._record_final(
                    self._finish(
                        ledger,
                        events,
                        "failed",
                        "agent_deadline_exceeded",
                        error="tool execution exceeded the remaining episode deadline",
                    ),
                    recorder,
                )
            action_latency_ms = int((time.monotonic() - action_started) * 1000)
            try:
                extra_tool_calls = sum(
                    max(
                        0,
                        execution.outcome.tool_calls_used
                        - (execution.action.action not in NO_TOOL_ACTIONS),
                    )
                    for execution in executions
                )
                ledger.budget = ledger.budget.consume(
                    tool_calls=extra_tool_calls,
                    latency_ms=int((time.monotonic() - batch_started) * 1000),
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
                        policy_latency_ms=policy_latency_ms,
                        action_latency_ms=action_latency_ms,
                    )
            if interrupt:
                return self._record_final(
                    self._finish(ledger, events, "interrupted", interrupt), recorder
                )
            completed_batches += 1
            if max_batches is not None and completed_batches >= max_batches:
                ledger.termination_reason = None
                self._event(
                    events,
                    "agent_checkpoint",
                    payload={"completed_batches": completed_batches},
                )
                return self._record_final(
                    AgentLoopResult(
                        ledger=ledger,
                        status="running",
                        termination_reason="continue",
                        events=events,
                    ),
                    recorder,
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
            progress_made = self._record_decision(ledger, execution)
            self._event(
                events,
                "action_completed",
                task_id=task.task_id,
                action_id=execution.action.action_id,
                payload={"action": execution.action.action, "status": outcome.status},
            )

            if outcome.status == "awaiting_user":
                for fact in outcome.facts:
                    ledger.facts[fact.fact_id] = fact
                for artifact in outcome.artifacts:
                    ledger.artifacts[artifact.artifact_id] = artifact
                ledger.task_graph = self.controller.transition(
                    ledger.task_graph, task.task_id, "blocked"
                )
                return "awaiting_user"

            if outcome.status == "failed":
                self._record_failure(ledger, execution)
                failure = {
                    "code": outcome.error_code or "ACTION_FAILED",
                    "message": outcome.error_message or "action failed",
                }
                if execution.action.action == "abort":
                    # A policy abort is an intentional terminal decision, not
                    # a transient executor failure. Retrying it manufactures a
                    # false recovery turn and corrupts SFT credit assignment.
                    ledger.task_graph = self.controller.transition(
                        ledger.task_graph,
                        task.task_id,
                        "failed",
                        failure=failure,
                    )
                else:
                    ledger.task_graph = self.controller.retry_or_fail(
                        ledger.task_graph,
                        task.task_id,
                        failure,
                    )
                continue

            for fact in outcome.facts:
                ledger.facts[fact.fact_id] = fact
            for artifact in outcome.artifacts:
                ledger.artifacts[artifact.artifact_id] = artifact

            if outcome.loop_control is not None:
                if not progress_made:
                    repeated = _TaskExecution(
                        task_id=task.task_id,
                        action=execution.action,
                        outcome=ActionOutcome(
                            status="failed",
                            error_code="REPEATED_NO_PROGRESS_ACTION",
                            error_message=(
                                "action returned the same evidence as a previous loop turn"
                            ),
                            retryable=True,
                        ),
                    )
                    self._record_failure(ledger, repeated)
                    ledger.task_graph = self.controller.retry_or_fail(
                        ledger.task_graph,
                        task.task_id,
                        {"code": "REPEATED_NO_PROGRESS_ACTION"},
                    )
                    continue
                if outcome.loop_control == "continue":
                    ledger.task_graph = self.controller.transition(
                        ledger.task_graph,
                        task.task_id,
                        "ready",
                    )
                    self._event(
                        events,
                        "observation_received",
                        task_id=task.task_id,
                        action_id=execution.action.action_id,
                        payload={"next": "policy_decision"},
                    )
                    continue
                self._apply_replan(ledger, outcome.loop_control, events, execution)
                continue

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
                attempted_arguments=execution.action.arguments,
            )
        )

    @staticmethod
    def _record_decision(ledger: AgentLedgerState, execution: _TaskExecution) -> bool:
        """Append one bounded decision record and detect repeated observations."""
        signature_payload = {
            "facts": [{"key": item.key, "value": item.value} for item in execution.outcome.facts],
            "artifacts": [
                {"type": item.artifact_type, "payload": item.payload}
                for item in execution.outcome.artifacts
            ],
            "observations": [
                {
                    "ok": item.ok,
                    "tool": item.tool,
                    "data": item.data,
                    "error": item.error.model_dump(mode="json") if item.error else None,
                }
                for item in execution.outcome.observations
            ],
        }
        has_evidence = any(signature_payload.values())
        signature = (
            hashlib.sha256(
                json.dumps(
                    signature_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if has_evidence
            else None
        )
        canonical_arguments = json.dumps(
            execution.action.arguments,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        repeated = any(
            item.task_id == execution.task_id
            and item.action == execution.action.action
            and json.dumps(
                item.arguments,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            )
            == canonical_arguments
            and signature is not None
            and item.observation_signature == signature
            for item in reversed(ledger.decision_history[-8:])
        )
        ledger.decision_history.append(
            DecisionRecord(
                task_id=execution.task_id,
                action_id=execution.action.action_id,
                action=execution.action.action,
                arguments=execution.action.arguments,
                outcome_status=execution.outcome.status,
                observation_signature=signature,
                progress_made=not repeated,
            )
        )
        if len(ledger.decision_history) > 32:
            ledger.decision_history = ledger.decision_history[-32:]
        return not repeated

    def _apply_replan(
        self,
        ledger: AgentLedgerState,
        control: Literal["replan_local", "replan_global"],
        events: list[AgentLoopEvent],
        execution: _TaskExecution,
    ) -> None:
        """Reopen the smallest safe subgraph while preserving grounded evidence."""
        if control == "replan_local":
            roots = ["solve_itinerary"]
        else:
            task_ids = {task.task_id for task in ledger.task_graph.tasks}
            roots = [
                "research_evidence" if "research_evidence" in task_ids else "search_candidates"
            ]
        old_version = ledger.task_graph.plan_version
        new_version = old_version + 1
        invalidated = self.controller.invalidate(ledger.task_graph, roots, cascade=True)
        task_ids = [task.task_id for task in invalidated.tasks if task.status == "invalidated"]
        invalidated_artifact_types = {
            "solver_result",
            "validation_report",
            "verified_itinerary_acceptance",
            "itinerary_draft",
        }
        if control == "replan_global":
            invalidated_artifact_types.update(
                {
                    "candidate_selection",
                    "poi_detail_set",
                    "research_bundle",
                    "route_matrix",
                    "solver_strategy_override",
                }
            )

        promoted_facts: list[FactRecord] = []
        for fact in list(ledger.facts.values()):
            if fact.goal_version != ledger.goal.goal_version or fact.plan_version != old_version:
                continue
            promoted_facts.append(
                fact.model_copy(
                    update={
                        "fact_id": f"{fact.fact_id}:plan-{new_version}",
                        "plan_version": new_version,
                    }
                )
            )
        promoted_artifacts: list[ArtifactRecord] = []
        for artifact in list(ledger.artifacts.values()):
            if (
                artifact.goal_version != ledger.goal.goal_version
                or artifact.plan_version != old_version
                or artifact.artifact_type in invalidated_artifact_types
            ):
                continue
            promoted_artifacts.append(
                artifact.model_copy(
                    update={
                        "artifact_id": f"{artifact.artifact_id}:plan-{new_version}",
                        "plan_version": new_version,
                    }
                )
            )
        for fact in promoted_facts:
            ledger.facts[fact.fact_id] = fact
        for artifact in promoted_artifacts:
            ledger.artifacts[artifact.artifact_id] = artifact

        invalidated = invalidated.model_copy(update={"plan_version": new_version})
        ledger.plan_versions.append(
            PlanVersion(
                plan_version=new_version,
                goal_version=ledger.goal.goal_version,
                trigger=control,
                evidence_refs=[execution.action.action_id],
                invalidated_task_ids=task_ids,
                preserved_task_ids=[
                    task.task_id for task in invalidated.tasks if task.task_id not in task_ids
                ],
            )
        )
        ledger.task_graph = self.controller.reopen_invalidated(invalidated, task_ids)
        self._event(
            events,
            "plan_reopened",
            task_id=execution.task_id,
            action_id=execution.action.action_id,
            payload={"scope": control, "reopened_task_ids": task_ids},
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
        now = datetime.now(UTC)
        current_facts = [
            fact
            for fact in ledger.facts.values()
            if fact.goal_version == ledger.goal.goal_version
            and fact.plan_version == ledger.task_graph.plan_version
            and (fact.expires_at is None or fact.expires_at > now)
        ]
        current_artifacts = [
            artifact
            for artifact in ledger.artifacts.values()
            if artifact.goal_version == ledger.goal.goal_version
            and artifact.plan_version == ledger.task_graph.plan_version
            and (artifact.expires_at is None or artifact.expires_at > now)
        ]
        required_fact_keys = set(task.required_facts)
        current_facts.sort(
            key=lambda fact: (
                fact.key in required_fact_keys,
                fact.key in {"fixed_events", "transport_time_windows"}
                or fact.key.startswith("user_input."),
                fact.created_at,
            )
        )
        required_artifact_types = set(
            task.success_criteria.get("required_artifact_types") or []
        ) | set(task.success_criteria.get("research_required_artifact_types") or [])
        current_artifacts.sort(
            key=lambda artifact: (
                artifact.artifact_type in required_artifact_types,
                artifact.artifact_type
                in {
                    "event_search_result",
                    "transport_search_result",
                    "validation_report",
                },
                artifact.created_at,
            )
        )
        relevant_facts = [fact.fact_id for fact in current_facts if fact.key in task.required_facts]
        retry_budget_remaining = max(0, task.max_attempts - task.attempts)
        failures = []
        for failure in ledger.failures[-3:]:
            if failure.task_id != task.task_id:
                continue
            visible_failure = failure.model_dump(mode="json")
            visible_failure["retry_budget_remaining"] = (
                retry_budget_remaining if failure.retryable else 0
            )
            failures.append(visible_failure)
        decision_history = [
            {
                "task_id": item.task_id,
                "action": item.action,
                "arguments": item.arguments,
                "outcome_status": item.outcome_status,
                "progress_made": item.progress_made,
            }
            for item in ledger.decision_history[-6:]
        ]
        remaining = sum(
            item.required and item.status not in {"succeeded", "skipped"}
            for item in ledger.task_graph.tasks
        )
        allowed_actions = BoundedAgentLoop._runtime_allowed_actions(
            ledger,
            task,
            current_artifacts,
        )
        current_subtask = task.model_dump(mode="json")
        current_subtask["allowed_actions"] = allowed_actions
        action_attempt_counts: dict[str, int] = {}
        for item in ledger.decision_history:
            if item.task_id != task.task_id:
                continue
            action_attempt_counts[item.action] = action_attempt_counts.get(item.action, 0) + 1
        current_subtask["action_attempt_counts"] = action_attempt_counts
        return PolicyContext(
            trajectory_id=ledger.trajectory_id,
            goal_version=ledger.goal.goal_version,
            plan_version=ledger.task_graph.plan_version,
            original_request=ledger.goal.original_request,
            current_subtask=current_subtask,
            hard_constraints=ledger.goal.hard_constraints,
            soft_preferences=ledger.goal.soft_preferences,
            capability=ledger.goal.capability.model_dump(mode="json"),
            missing_information=ledger.goal.missing_information,
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
            decision_history=decision_history,
            remaining_tasks=remaining,
            remaining_steps=ledger.budget.remaining_episode_steps,
            allowed_actions=allowed_actions,
        )

    @staticmethod
    def _runtime_allowed_actions(
        ledger: AgentLedgerState,
        task: TaskNode,
        current_artifacts: list[ArtifactRecord],
    ) -> list[str]:
        """Derive the state-scoped allowlist before any policy implementation runs."""
        allowed = list(task.allowed_actions)
        if task.task_id == "search_candidates":
            candidates = next(
                (
                    artifact
                    for artifact in reversed(current_artifacts)
                    if artifact.artifact_type == "poi_candidate_set"
                ),
                None,
            )
            if candidates is None or not (candidates.payload.get("pois") or []):
                return [action for action in allowed if action in {"search_pois", "ask_user"}]
        if task.task_id == "review_itinerary":
            report = next(
                (
                    artifact
                    for artifact in reversed(current_artifacts)
                    if artifact.artifact_type == "validation_report"
                ),
                None,
            )
            if report is not None and report.payload.get("hard_pass") is True:
                return [action for action in allowed if action == "accept_itinerary"]
            return [action for action in allowed if action != "accept_itinerary"]
        return allowed

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
    elif artifact.artifact_type == "city_knowledge":
        # The policy only needs coverage signals to choose its next action.  Replaying
        # every POI record (even recursively truncated) makes later ReAct turns grow
        # linearly and teaches the student to attend to placeholder noise.
        pois = payload.get("pois") or []
        summary.update(
            {
                "city": payload.get("city"),
                "topic": payload.get("topic"),
                "record_count": payload.get("record_count", len(pois)),
                "poi_names": [
                    str(item.get("name"))
                    for item in pois[:8]
                    if isinstance(item, dict) and item.get("name")
                ],
                "evidence_source": payload.get("_evidence_source"),
                "evidence_confidence": payload.get("_evidence_confidence"),
                "is_fallback": bool(payload.get("_is_fallback", False)),
            }
        )
    elif artifact.artifact_type in {
        "current_info_search",
        "event_search_result",
        "transport_search_result",
    }:
        results = payload.get("results") or []
        summary.update(
            {
                "trust_tier": "untrusted_external",
                "info_type": payload.get("info_type"),
                "date": payload.get("date"),
                "queried_at": payload.get("queried_at"),
                "source_count": len(results) if isinstance(results, list) else 0,
                # Full redirect URLs can be thousands of characters and are neither
                # actionable nor safe policy input.  Domains preserve provenance while
                # the verifier retains the complete source URLs in the artifact ledger.
                "source_domains": list(
                    dict.fromkeys(
                        domain
                        for item in results[:8]
                        if isinstance(item, dict)
                        for domain in [_source_domain(item.get("url"))]
                        if domain
                    )
                ),
                "security_flags": list(
                    dict.fromkeys(
                        flag
                        for item in results[:8]
                        if isinstance(item, dict)
                        for flag in (item.get("security_flags") or [])
                    )
                ),
            }
        )
        if artifact.artifact_type == "event_search_result":
            summary["event"] = _compact_value(payload.get("event") or {})
        if artifact.artifact_type == "transport_search_result":
            summary["legs"] = _compact_value(payload.get("legs") or [])
    elif artifact.artifact_type == "research_bundle":
        summary["artifact_types"] = sorted(
            str(item) for item in (payload.get("artifact_types") or [])
        )
    else:
        summary["payload"] = _compact_value(payload)
    return summary


def _source_domain(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return urlsplit(value).hostname
    except ValueError:
        return None


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
