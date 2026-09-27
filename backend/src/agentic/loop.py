"""Bounded, evidence-gated runtime shared by API and future local policies."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Literal, Protocol
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel, Field

from agentic.observations import ObservationEnvelope
from agentic.clock import reference_now
from agentic.policy_actions import authorize_policy_action
from agentic.scheduler import TaskScheduler
from agentic.state import (
    AgentLedgerState,
    ArtifactRecord,
    BudgetExceeded,
    DecisionRecord,
    FactRecord,
    FailureRecord,
    GoalCapability,
    StateTransitionError,
    TaskGraphController,
    TaskNode,
)
from agentic.termination import CompletionGuard
from agentic.verifier_repair import (
    constraint_handling_for_violation,
    parse_constraint_flexibility,
    relaxation_options_for_violation,
)
from agentic.verifier import SubtaskVerifier
from core.inference_metrics import InferenceMetrics


NO_TOOL_ACTIONS = frozenset(
    {
        "abort",
        "ask_user",
        "finish",
        "propose_tradeoff",
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
    # Recomputed from the full ledger for runtime guards. Never a model hint,
    # serialized training input, or replacement for the visible failure history.
    resolved_precondition_failures: list[str] = Field(default_factory=list, exclude=True)
    decision_history: list[dict[str, Any]] = Field(default_factory=list)
    remaining_tasks: int
    remaining_steps: int
    remaining_budget: dict[str, int] = Field(default_factory=dict)
    observation_time: str | None = None
    allowed_actions: list[str]
    policy_feedback: list[dict[str, Any]] = Field(default_factory=list)


class PolicyAction(BaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid4()))
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    token_usage: int = Field(default=0, ge=0)
    decision_source: Literal["policy", "controller"] = "policy"
    inference_metrics: InferenceMetrics | None = None
    repair_attempts: int = Field(default=0, ge=0)
    repair_error_codes: list[str] = Field(default_factory=list)
    # ``arguments`` is the authorized/executed payload.  These fields preserve
    # the policy-authored payload so safety and model quality can be audited
    # independently when the controller hydrates trusted arguments.
    model_arguments: dict[str, Any] | None = None
    executed_arguments: dict[str, Any] | None = None
    argument_sources: dict[str, str] = Field(default_factory=dict)
    controller_override_attempt: bool = False
    model_contract_compliant: bool = True
    controller_hydration_exact: bool | None = None
    controller_hydrated_fields: list[str] = Field(default_factory=list)


class ActionOutcome(BaseModel):
    status: Literal["completed", "failed", "awaiting_user"] = "completed"
    observations: list[ObservationEnvelope] = Field(default_factory=list)
    facts: list[FactRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    tool_calls_used: int = Field(default=0, ge=0)


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


async def gather_or_cancel(*coroutines):
    """Retain completed results in callers and drain siblings before finalizing."""
    tasks = [asyncio.create_task(coroutine) for coroutine in coroutines]
    try:
        return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


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
        from agentic.harness import assert_session_contract

        assert_session_contract(ledger)
        events: list[AgentLoopEvent] = []
        pending_batch = None

        def record_final(result, recorder):
            from agentic.episode_accounting import CONTRACT
            from agentic.trajectory import redact_pii
            if pending_batch is not None:
                for attempt in pending_batch['policy_attempts'] + pending_batch['execution_attempts']:
                    if attempt['status'] == 'pending': attempt['status'] = 'not_started'
                pending_batch['budget_after'] = ledger.budget.model_dump(mode='json')
                pending_batch['state_after'] = ledger.model_dump(mode='json')
                self._event(result.events, 'agent_uncommitted_batch', payload=redact_pii(pending_batch))
            self._event(result.events, 'agent_accounting_contract', payload={'schema_version': CONTRACT})
            return self._record_final(result, recorder)

        async def propose_one(index, context):
            attempt = pending_batch['policy_attempts'][index]
            attempt['status'] = 'running'
            try:
                proposal = await policy.propose(context.model_copy(deep=True))
                attempt.update(status='returned', proposal=proposal.model_dump(mode='json'))
                return proposal
            except BaseException as exc:
                attempt.update(status='cancelled' if isinstance(exc, asyncio.CancelledError) else 'failed', error_type=type(exc).__name__)
                raise

        async def execute_one(index, task_id, action):
            attempt = pending_batch['execution_attempts'][index]
            attempt['status'] = 'running'
            try:
                execution = await self._execute(ledger=ledger, task=ledger.task_graph.get(task_id),
                    action=action, executor=executor, execution_audit=attempt)
                attempt.update(status='returned', action=execution.action.model_dump(mode='json'),
                    outcome=execution.outcome.model_dump(mode='json'))
                return execution
            except BaseException as exc:
                attempt.update(status='cancelled' if isinstance(exc, asyncio.CancelledError) else 'failed',
                    error_type=type(exc).__name__, action=action.model_dump(mode='json'))
                raise

        completed_batches = 0
        while True:
            if ledger.budget.remaining_episode_steps <= 0:
                return record_final(
                    self._finish(ledger, events, "failed", "budget_exhausted_fallback"), recorder
                )
            batch_started = time.monotonic()
            ledger.task_graph, batch = self.scheduler.select(ledger.task_graph)
            if batch is None:
                return record_final(self._terminal_result(ledger, events), recorder)

            pending_batch = dict(schema_version='uncommitted-batch.v1', phase='policy',
                training_labels_authorized=False, budget_before=ledger.budget.model_dump(mode='json'),
                policy_attempts=[], execution_attempts=[])
            try:
                ledger.task_graph = self.scheduler.start(ledger.task_graph, batch)
                contexts = [
                    self._policy_context(ledger, ledger.task_graph.get(task_id))
                    for task_id in batch.task_ids
                ]
                pending_batch['contexts'] = [c.model_dump(mode='json') for c in contexts]
                pending_batch['policy_attempts'] = [dict(task_id=t, status='pending') for t in batch.task_ids]
                policy_started = time.monotonic()
                remaining_seconds = max(
                    0.001,
                    (ledger.budget.timeout_ms - ledger.budget.used_latency_ms) / 1000,
                )
                async with asyncio.timeout(remaining_seconds):
                    proposals = await gather_or_cancel(
                        *(propose_one(i, context) for i, context in enumerate(contexts))
                    )
                proposals = [
                    authorize_policy_action(context, proposal)
                    for context, proposal in zip(contexts, proposals, strict=True)
                ]
                policy_latency_ms = int((time.monotonic() - policy_started) * 1000)
                pending_batch['phase'] = 'proposal_budget'
                pending_batch['requested_charge'] = dict(episode_steps=len(proposals), tokens=sum(p.token_usage for p in proposals))
                ledger.budget = ledger.budget.consume(
                    episode_steps=len(proposals),
                    tokens=sum(proposal.token_usage for proposal in proposals),
                )
            except BudgetExceeded as exc:
                return record_final(
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
                return record_final(
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
                failure_class = (
                    "model_policy_failure"
                    if type(exc).__name__ == "PolicyOutputError"
                    else "runtime_error"
                )
                return record_final(
                    self._finish(
                        ledger,
                        events,
                        "failed",
                        "policy_error_fallback",
                        error=f"{type(exc).__name__}: {exc}",
                        failure_class=failure_class,
                        error_type=type(exc).__name__,
                        error_code=getattr(exc, "code", None),
                        error_detail_code=getattr(exc, "detail_code", None),
                        policy_output_summary=getattr(exc, "output_summary", None),
                    ),
                    recorder,
                )
            except asyncio.CancelledError:
                record_final(self._finish(ledger, events, 'failed', 'agent_cancelled'), recorder)
                raise

            pending_batch['phase'] = 'execution'
            pending_batch['execution_attempts'] = [dict(task_id=t, action_id=a.action_id, status='pending', executor_started=False)
                for t,a in zip(batch.task_ids,proposals,strict=True)]
            state_before = ledger.model_copy(deep=True)
            action_started = time.monotonic()
            try:
                remaining_seconds = max(
                    0.001,
                    (ledger.budget.timeout_ms - ledger.budget.used_latency_ms) / 1000,
                )
                async with asyncio.timeout(remaining_seconds):
                    executions = await gather_or_cancel(
                        *(execute_one(i, task_id, action)
                          for i,(task_id,action) in enumerate(zip(batch.task_ids,proposals,strict=True)))
                    )
            except BudgetExceeded as exc:
                return record_final(
                    self._finish(
                        ledger, events, "failed", "budget_exhausted_fallback", error=str(exc)
                    ),
                    recorder,
                )
            except TimeoutError:
                return record_final(
                    self._finish(
                        ledger,
                        events,
                        "failed",
                        "agent_deadline_exceeded",
                        error="tool execution exceeded the remaining episode deadline",
                    ),
                    recorder,
                )
            except asyncio.CancelledError:
                record_final(self._finish(ledger, events, 'failed', 'agent_cancelled'), recorder)
                raise
            except Exception as exc:
                return record_final(self._finish(ledger, events, 'failed', 'execution_error_fallback',
                    error=f'{type(exc).__name__}: {exc}', failure_class='runtime_error'), recorder)
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
                pending_batch['phase'] = 'post_execution_budget'
                pending_batch['requested_charge'] = dict(tool_calls=extra_tool_calls,
                    latency_ms=int((time.monotonic()-batch_started)*1000))
                ledger.budget = ledger.budget.consume(
                    tool_calls=extra_tool_calls,
                    latency_ms=int((time.monotonic() - batch_started) * 1000),
                )
            except BudgetExceeded as exc:
                return record_final(
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
                pending_batch['phase'] = 'commit'
                interrupt = self._commit_batch(ledger, batch, executions, events)
            except StateTransitionError as exc:
                return record_final(
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
            pending_batch = None
            if interrupt:
                return record_final(
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
                return record_final(
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
        execution_audit: dict | None = None,
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
        now = reference_now()
        current_artifacts = [
            artifact
            for artifact in ledger.artifacts.values()
            if artifact.goal_version == ledger.goal.goal_version
            and artifact.plan_version == ledger.task_graph.plan_version
            and (artifact.expires_at is None or artifact.expires_at > now)
        ]
        runtime_allowed = self._runtime_allowed_actions(
            ledger,
            task,
            current_artifacts,
        )
        if action.action not in runtime_allowed:
            return _TaskExecution(
                task_id=task.task_id,
                action=action,
                outcome=ActionOutcome(
                    status="failed",
                    error_code="ACTION_NOT_AUTHORIZED",
                    error_message=(
                        f"{action.action} is not authorized by the current runtime state"
                    ),
                ),
            )
        from agentic.harness import preflight_action

        rejection = preflight_action(ledger, action)
        if rejection is not None:
            return _TaskExecution(task_id=task.task_id, action=action, outcome=rejection)
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
        # Rejected proposals already consumed a model turn and tokens, but did
        # not execute a tool. Reserve resources only after the harness accepts
        # execution, and before crossing the executor boundary (including failures).
        if execution_audit is not None: execution_audit['phase'] = 'resource_admission'
        if execution_audit is not None:
            execution_audit['requested_charge'] = dict(tool_calls=int(action.action not in NO_TOOL_ACTIONS),
                solver_calls=int(action.action == 'solve_itinerary'))
        ledger.budget = ledger.budget.consume(
            tool_calls=int(action.action not in NO_TOOL_ACTIONS),
            solver_calls=int(action.action == "solve_itinerary"),
        )
        if execution_audit is not None:
            execution_audit.update(phase='executor', executor_started=True)
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

    def _commit_batch(self, ledger, batch, executions, events):
        from agentic.harness import invalidate_derived_evidence

        for execution in executions:
            self._assert_current_versions(ledger, execution.outcome)
            task = ledger.task_graph.get(execution.task_id)
            outcome = execution.outcome
            progress = self._record_decision(ledger, execution)
            self._event(
                events,
                "action_completed",
                task_id=task.task_id,
                action_id=execution.action.action_id,
                payload={"action": execution.action.action, "status": outcome.status},
            )
            if outcome.status != "failed":
                invalidate_derived_evidence(
                    ledger,
                    execution.action.action,
                    artifacts=outcome.artifacts,
                    facts=outcome.facts,
                )
                for fact in outcome.facts:
                    ledger.facts[fact.fact_id] = fact
                for artifact in outcome.artifacts:
                    ledger.artifacts[artifact.artifact_id] = artifact
                from agentic.harness import refresh_constraint_evidence
                refresh_constraint_evidence(ledger)
                self._refresh_post_validation_capability(ledger, outcome.artifacts)
            if outcome.status == "awaiting_user":
                ledger.task_graph = self.controller.transition(
                    ledger.task_graph, task.task_id, "blocked"
                )
                return "awaiting_user"
            if outcome.status == "failed":
                self._record_failure(ledger, execution)
                # Only an accepted abort ends the task. A rejected stop proposal
                # returns its evidence error to the model like any other rejection.
                target = "failed" if execution.action.action == "abort" and outcome.error_code == "POLICY_ABORT" else "ready"
                ledger.task_graph = self.controller.transition(
                    ledger.task_graph,
                    task.task_id,
                    target,
                    failure={"code": outcome.error_code, "message": outcome.error_message},
                )
                continue
            if not progress:
                outcome.error_code = "REPEATED_NO_PROGRESS_ACTION"
                outcome.error_message = "No new evidence; choose a different query or next action."
                self._record_failure(ledger, execution)
            # Every result, including solver and validation, returns to the model.
            ledger.task_graph = self.controller.transition(ledger.task_graph, task.task_id, "ready")
            self._event(
                events,
                "observation_received",
                task_id=task.task_id,
                payload={"next": "policy_decision"},
            )
        return None

    @staticmethod
    def _refresh_post_validation_capability(
        ledger: AgentLedgerState,
        artifacts: list[ArtifactRecord],
    ) -> None:
        """Derive capability from neutral user constraints and verifier evidence."""
        report = next(
            (
                artifact
                for artifact in reversed(artifacts)
                if artifact.artifact_type == "validation_report"
            ),
            None,
        )
        if report is None:
            return
        if report.payload.get("hard_pass") is True:
            ledger.goal = ledger.goal.model_copy(
                update={"capability": GoalCapability(status="solvable")}
            )
            return

        feasibility = report.payload.get("feasibility") or {}
        if feasibility.get("schema_version") == "required-facts-feasibility.v1":
            witnesses = feasibility.get("witnesses") or []
            if feasibility.get("status") == "infeasible" and witnesses:
                from agentic.harness import capability_from_feasibility
                capability = capability_from_feasibility(ledger, feasibility)
            else:
                # No certificate says nothing about global feasibility. A
                # missing POI or one failed draft must not become a hard stop.
                capability = GoalCapability(status="undetermined", evidence=[
                    str(v.get("message") or "") for v in report.payload.get("hard_violations") or []
                    if isinstance(v, dict) and v.get("message")])
            ledger.goal = ledger.goal.model_copy(update={"capability": capability})
            return

        violations = [
            item for item in report.payload.get("hard_violations") or [] if isinstance(item, dict)
        ]
        visible_messages = [
            str(item.get("message") or "").strip()
            for item in violations
            if str(item.get("message") or "").strip()
        ]
        if (
            violations
            and visible_messages
            and len(visible_messages) == len(violations)
            and {str(item.get("code") or "").strip() for item in violations}
            == {"MUST_VISIT_MISSING"}
            and ledger.goal.hard_constraints.get("must_visit")
        ):
            # The user has already supplied the missing POI names.  This is a
            # candidate-retrieval/solver repair state, never a clarification
            # state: asking whether the explicitly required POIs are required
            # would discard a hard constraint.
            ledger.goal = ledger.goal.model_copy(
                update={
                    "capability": GoalCapability(
                        status="solvable",
                        evidence=visible_messages,
                    )
                }
            )
            return
        # Clear capability derived from any older validation round before
        # assessing the new report. Unknown/malformed/multi-violation states
        # therefore fail closed to the generalist ask-user path instead of
        # inheriting a stale terminal decision.
        ledger.goal = ledger.goal.model_copy(
            update={
                "capability": GoalCapability(
                    status="needs_user",
                    evidence=visible_messages,
                )
            }
        )
        if len(violations) != 1:
            return
        code = str(violations[0].get("code") or "").strip()
        message = str(violations[0].get("message") or "").strip()
        if not code or not message:
            return

        contract = parse_constraint_flexibility(
            ledger.goal.hard_constraints.get("constraint_flexibility")
        )
        if contract is None:
            return
        handling = constraint_handling_for_violation(code, contract)
        if handling is None:
            return

        if handling == "solver_adjustable":
            capability = GoalCapability(status="solvable", evidence=[message])
        elif handling == "relaxable":
            allowed_relaxations = relaxation_options_for_violation(code, contract)
            if not allowed_relaxations:
                return
            capability = GoalCapability(
                status="infeasible",
                evidence=[message],
                actionable_alternatives=True,
                alternatives=allowed_relaxations,
            )
        elif handling == "locked":
            capability = GoalCapability(
                status="infeasible",
                evidence=[message],
                actionable_alternatives=False,
            )
        else:  # defensive fail-closed branch for future schema extensions
            return
        ledger.goal = ledger.goal.model_copy(update={"capability": capability})

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
        def evidence_payload(payload: Any, *, solver_result: bool) -> Any:
            # Runtime telemetry is retained in the ledger and observations, but
            # must not turn an identical solver result into new task evidence.
            if solver_result and isinstance(payload, dict):
                return {key: value for key, value in payload.items() if key != "solve_time_ms"}
            return payload

        signature_payload = {
            "facts": [{"key": item.key, "value": item.value} for item in execution.outcome.facts],
            "artifacts": [
                {
                    "type": item.artifact_type,
                    "payload": evidence_payload(
                        item.payload, solver_result=item.artifact_type == "solver_result"
                    ),
                }
                for item in execution.outcome.artifacts
            ],
            "observations": [
                {
                    "ok": item.ok,
                    "tool": item.tool,
                    "data": evidence_payload(item.data, solver_result=item.tool == "solve_itinerary"),
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
        now = reference_now()
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
        # Repeated snapshots must not crowd a still-current solver/validation
        # result out of the bounded context. Keep latest state per type; retain
        # distinct search queries so the model can compare conflicting evidence.
        latest_visible = {}
        for artifact in current_artifacts:
            query = (
                str(artifact.payload.get("query") or "")
                if artifact.artifact_type == "current_info_search"
                else ""
            )
            latest_visible[(artifact.artifact_type, query)] = artifact
        current_artifacts = list(latest_visible.values())
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
                    "solver_result",
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
        from agentic.harness import resolved_precondition_failure_ids

        resolved_failures = resolved_precondition_failure_ids(ledger, failures)
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
            capability=ledger.goal.capability.model_dump(mode="json",
                exclude={"evidence_ids"} if not ledger.goal.capability.evidence_ids else None),
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
            resolved_precondition_failures=resolved_failures,
            decision_history=decision_history,
            remaining_tasks=remaining,
            remaining_steps=ledger.budget.remaining_episode_steps,
            remaining_budget={
                "tool_calls": ledger.budget.remaining_tool_calls,
                "solver_calls": ledger.budget.max_solver_calls - ledger.budget.used_solver_calls,
                "completion_tokens": ledger.budget.max_tokens - ledger.budget.used_tokens,
                "latency_ms": ledger.budget.timeout_ms - ledger.budget.used_latency_ms,
            },
            observation_time=now.isoformat(),
            allowed_actions=allowed_actions,
        )

    @staticmethod
    def _runtime_allowed_actions(ledger, task, current_artifacts):
        from agentic.policy_actions import model_visible_policy_actions

        return model_visible_policy_actions(
            task.allowed_actions, capability=ledger.goal.capability.model_dump(mode="json")
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
        failure_class: str | None = None,
        error_type: str | None = None,
        error_code: str | None = None,
        error_detail_code: str | None = None,
        policy_output_summary: dict[str, Any] | None = None,
    ) -> AgentLoopResult:
        ledger.termination_reason = reason
        payload = {"status": status, "reason": reason, "error": error}
        payload.update(
            {
                key: value
                for key, value in {
                    "failure_class": failure_class,
                    "error_type": error_type,
                    "error_code": error_code,
                    "error_detail_code": error_detail_code,
                    "policy_output_summary": policy_output_summary,
                }.items()
                if value is not None
            }
        )
        self._event(
            events,
            "episode_terminated",
            payload=payload,
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
        summary["trust_tier"] = "untrusted_external"
        summary["details"] = [_poi_evidence(item) for item in details[:8] if isinstance(item, dict)]
        summary["omitted_details"] = max(0, len(details) - 8)
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
                "candidate_available": any(
                    day.get("activities")
                    for day in payload.get("days") or []
                    if isinstance(day, dict)
                ),
                "status_scope": "solver_algorithm_only",
                "validation_is_separate": True,
                "days": [
                    {
                        "day_number": day.get("day_number"),
                        "total_cost": day.get("total_cost"),
                        "activities": [
                            _evidence_fields(
                                item, ("poi_name", "start_time", "end_time", "ticket_price")
                            )
                            for item in (day.get("activities") or [])[:8]
                            if isinstance(item, dict)
                        ],
                        "omitted_activities": max(0, len(day.get("activities") or []) - 8),
                    }
                    for day in (payload.get("days") or [])[:3]
                    if isinstance(day, dict)
                ],
                "omitted_days": max(0, len(payload.get("days") or []) - 3),
            }
        )
    elif artifact.artifact_type == "validation_report":
        violations = [
            {
                "code": item.get("code"),
                "message": _compact_value(item.get("message")),
            }
            for item in (payload.get("hard_violations") or [])[:10]
            if isinstance(item, dict)
        ]
        summary.update(
            {
                "hard_pass": payload.get("hard_pass"),
                "violation_codes": [item["code"] for item in violations],
                # Review decisions need the verifier's bounded, user-facing
                # evidence.  Codes alone cannot ground a retry, trade-off, or
                # safe abort reason and made the previous reward contract
                # impossible to satisfy from model-visible state.
                "violations": violations,
                "soft_scores": _compact_value(payload.get("soft_scores") or {}),
            }
        )
    elif artifact.artifact_type == "weather_snapshot":
        days = payload.get("days") or payload.get("data") or []
        summary["day_count"] = len(days) if isinstance(days, list) else 0
        summary["conditions"] = [
            item.get("condition") for item in days[:7] if isinstance(item, dict)
        ]
        summary["days"] = (
            [
                _evidence_fields(item, ("date", "condition", "temperature", "warning"))
                for item in days[:7]
                if isinstance(item, dict)
            ]
            if isinstance(days, list)
            else []
        )
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
                "query": _compact_value(payload.get("query")),
                "results": [
                    {
                        **_evidence_fields(
                            item, ("title", "snippet", "published_at", "date", "security_flags")
                        ),
                        "source_domain": _source_domain(item.get("url")),
                    }
                    for item in results[:5]
                    if isinstance(item, dict)
                ],
                "omitted_results": max(0, len(results) - 5),
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


def _evidence_fields(item: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Bound each selected fact without truncating scalar values by nesting depth."""
    return {key: _compact_value(item[key]) for key in keys if key in item}


def _poi_evidence(item: dict[str, Any]) -> dict[str, Any]:
    return _evidence_fields(
        item,
        (
            "name",
            "category",
            "open_time",
            "close_time",
            "opening_hours",
            "closed_dates",
            "ticket_price",
            "duration_minutes",
            "reservation_required",
            "reservation_note",
            "date",
            "source",
            "queried_at",
        ),
    )


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
