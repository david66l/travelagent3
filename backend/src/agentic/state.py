"""Authoritative state models for the long-horizon agent loop.

These models are deliberately independent from model messages.  The policy may
propose actions, but only :class:`TaskGraphController` can change task status.
That keeps online execution, replay and future RL training on the same state
transition semantics.
"""

from __future__ import annotations

from collections import deque
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


AGENTIC_STATE_SCHEMA_VERSION = "agentic-state.v1"

CapabilityStatus = Literal["solvable", "needs_user", "missing_tool", "infeasible", "unsafe"]
TaskStatus = Literal[
    "pending",
    "ready",
    "running",
    "blocked",
    "succeeded",
    "failed",
    "invalidated",
    "skipped",
]


def _now() -> datetime:
    return datetime.now(UTC)


class StateTransitionError(ValueError):
    """Raised when a controller operation would violate the task state machine."""


class BudgetExceeded(StateTransitionError):
    """Raised before an action would exceed a durable episode budget."""


class GoalCapability(BaseModel):
    status: CapabilityStatus = "solvable"
    evidence: list[str] = Field(default_factory=list)


class GoalLedger(BaseModel):
    goal_version: int = Field(default=1, ge=1)
    original_request: str = Field(min_length=1)
    success_definition: list[str] = Field(default_factory=list)
    hard_constraints: dict[str, Any] = Field(default_factory=dict)
    soft_preferences: dict[str, Any] = Field(default_factory=dict)
    locked_items: list[dict[str, Any]] = Field(default_factory=list)
    user_authorizations: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    capability: GoalCapability = Field(default_factory=GoalCapability)


class TaskNode(BaseModel):
    model_config = {"frozen": True}

    task_id: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    status: TaskStatus = "pending"
    required: bool = True
    depends_on: tuple[str, ...] = ()
    required_facts: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = ()
    success_criteria: dict[str, Any] = Field(default_factory=dict)
    artifact_refs: tuple[str, ...] = ()
    verifier_evidence_refs: tuple[str, ...] = ()
    attempts: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=2, ge=1)
    failure: dict[str, Any] | None = None
    invalidates_on: tuple[str, ...] = ()
    updated_at: datetime = Field(default_factory=_now)


class TaskGraph(BaseModel):
    schema_version: str = AGENTIC_STATE_SCHEMA_VERSION
    goal_version: int = Field(ge=1)
    plan_version: int = Field(default=1, ge=1)
    tasks: tuple[TaskNode, ...]

    @model_validator(mode="after")
    def validate_dag(self) -> TaskGraph:
        task_ids = [task.task_id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id values must be unique")

        known = set(task_ids)
        for task in self.tasks:
            missing = set(task.depends_on) - known
            if missing:
                raise ValueError(f"task {task.task_id} has missing dependencies: {sorted(missing)}")
            if task.task_id in task.depends_on:
                raise ValueError(f"task {task.task_id} cannot depend on itself")

        dependencies = {task.task_id: task.depends_on for task in self.tasks}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("task graph must be acyclic")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in dependencies[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in task_ids:
            visit(task_id)
        return self

    def get(self, task_id: str) -> TaskNode:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)


class FactRecord(BaseModel):
    fact_id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    value: Any
    observation_ref: str = Field(min_length=1)
    goal_version: int = Field(ge=1)
    plan_version: int = Field(ge=1)
    source: str
    confidence: float = Field(ge=0, le=1)
    expires_at: datetime | None = None
    created_at: datetime = Field(default_factory=_now)


class ArtifactRecord(BaseModel):
    artifact_id: str = Field(min_length=1)
    artifact_type: str = Field(min_length=1)
    payload: dict[str, Any]
    evidence_refs: list[str] = Field(default_factory=list)
    goal_version: int = Field(ge=1)
    plan_version: int = Field(ge=1)
    created_at: datetime = Field(default_factory=_now)


class FailureRecord(BaseModel):
    failure_id: str = Field(default_factory=lambda: str(uuid4()))
    task_id: str
    action_id: str | None = None
    code: str
    message: str
    retryable: bool = False
    evidence_refs: list[str] = Field(default_factory=list)
    attempted_strategy: str | None = None
    created_at: datetime = Field(default_factory=_now)


class PlanVersion(BaseModel):
    plan_version: int = Field(ge=1)
    goal_version: int = Field(ge=1)
    trigger: str
    evidence_refs: list[str] = Field(default_factory=list)
    invalidated_task_ids: list[str] = Field(default_factory=list)
    preserved_task_ids: list[str] = Field(default_factory=list)
    changed_constraints: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class BudgetLedger(BaseModel):
    model_config = {"frozen": True}

    max_episode_steps: int = Field(default=16, ge=1)
    max_tool_calls: int = Field(default=16, ge=0)
    max_solver_calls: int = Field(default=3, ge=0)
    max_tokens: int = Field(default=32_000, ge=0)
    timeout_ms: int = Field(default=120_000, ge=1)
    used_episode_steps: int = Field(default=0, ge=0)
    used_tool_calls: int = Field(default=0, ge=0)
    used_solver_calls: int = Field(default=0, ge=0)
    used_tokens: int = Field(default=0, ge=0)
    used_latency_ms: int = Field(default=0, ge=0)

    def consume(
        self,
        *,
        episode_steps: int = 0,
        tool_calls: int = 0,
        solver_calls: int = 0,
        tokens: int = 0,
        latency_ms: int = 0,
    ) -> BudgetLedger:
        updates = {
            "used_episode_steps": self.used_episode_steps + episode_steps,
            "used_tool_calls": self.used_tool_calls + tool_calls,
            "used_solver_calls": self.used_solver_calls + solver_calls,
            "used_tokens": self.used_tokens + tokens,
            "used_latency_ms": self.used_latency_ms + latency_ms,
        }
        limits = {
            "used_episode_steps": self.max_episode_steps,
            "used_tool_calls": self.max_tool_calls,
            "used_solver_calls": self.max_solver_calls,
            "used_tokens": self.max_tokens,
            "used_latency_ms": self.timeout_ms,
        }
        exceeded = [name for name, value in updates.items() if value > limits[name]]
        if exceeded:
            raise BudgetExceeded(f"budget exceeded: {', '.join(exceeded)}")
        return self.model_copy(update=updates)

    @property
    def remaining_episode_steps(self) -> int:
        return self.max_episode_steps - self.used_episode_steps

    @property
    def remaining_tool_calls(self) -> int:
        return self.max_tool_calls - self.used_tool_calls


class AgentLedgerState(BaseModel):
    schema_version: str = AGENTIC_STATE_SCHEMA_VERSION
    trajectory_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: GoalLedger
    task_graph: TaskGraph
    facts: dict[str, FactRecord] = Field(default_factory=dict)
    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)
    failures: list[FailureRecord] = Field(default_factory=list)
    plan_versions: list[PlanVersion] = Field(default_factory=list)
    budget: BudgetLedger = Field(default_factory=BudgetLedger)
    current_task_id: str | None = None
    termination_reason: str | None = None

    @model_validator(mode="after")
    def versions_are_consistent(self) -> AgentLedgerState:
        if self.task_graph.goal_version != self.goal.goal_version:
            raise ValueError("task graph must target the current goal version")
        return self


_ALLOWED_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    "pending": {"ready", "invalidated", "skipped"},
    "ready": {"running", "invalidated", "skipped"},
    "running": {"ready", "blocked", "succeeded", "failed", "invalidated"},
    "blocked": {"ready", "failed", "invalidated"},
    "succeeded": {"invalidated"},
    "failed": {"ready", "invalidated"},
    "invalidated": {"pending", "skipped"},
    "skipped": {"invalidated"},
}


class TaskGraphController:
    """The only supported writer for task status transitions."""

    @staticmethod
    def _replace(graph: TaskGraph, replacement: TaskNode) -> TaskGraph:
        tasks = tuple(
            replacement if task.task_id == replacement.task_id else task for task in graph.tasks
        )
        return TaskGraph(
            goal_version=graph.goal_version,
            plan_version=graph.plan_version,
            tasks=tasks,
        )

    def transition(
        self,
        graph: TaskGraph,
        task_id: str,
        target: TaskStatus,
        *,
        evidence_refs: list[str] | None = None,
        failure: dict[str, Any] | None = None,
    ) -> TaskGraph:
        task = graph.get(task_id)
        if target not in _ALLOWED_TRANSITIONS[task.status]:
            raise StateTransitionError(f"invalid transition: {task.status} -> {target}")
        evidence = tuple(evidence_refs or ())
        if target == "succeeded" and not evidence:
            raise StateTransitionError("verifier evidence is required for succeeded")
        attempts = task.attempts + 1 if target == "running" else task.attempts
        if target == "running" and attempts > task.max_attempts:
            raise StateTransitionError(f"task {task_id} exhausted its attempt budget")
        replacement = task.model_copy(
            update={
                "status": target,
                "attempts": attempts,
                "verifier_evidence_refs": evidence or task.verifier_evidence_refs,
                "failure": failure,
                "updated_at": _now(),
            }
        )
        return self._replace(graph, replacement)

    def refresh_ready(self, graph: TaskGraph) -> TaskGraph:
        refreshed = graph
        terminal_dependencies = {"succeeded", "skipped"}
        for task in graph.tasks:
            if task.status != "pending":
                continue
            dependencies_ready = all(
                refreshed.get(dependency).status in terminal_dependencies
                for dependency in task.depends_on
            )
            if dependencies_ready:
                refreshed = self.transition(refreshed, task.task_id, "ready")
        return refreshed

    @staticmethod
    def ready_tasks(graph: TaskGraph) -> list[TaskNode]:
        return [task for task in graph.tasks if task.status == "ready"]

    def retry_or_fail(self, graph: TaskGraph, task_id: str, failure: dict[str, Any]) -> TaskGraph:
        task = graph.get(task_id)
        if task.status != "running":
            raise StateTransitionError("only a running task can fail or retry")
        target: TaskStatus = "ready" if task.attempts < task.max_attempts else "failed"
        return self.transition(graph, task_id, target, failure=failure)

    def invalidate(
        self, graph: TaskGraph, task_ids: list[str], *, cascade: bool = True
    ) -> TaskGraph:
        descendants: dict[str, list[str]] = {task.task_id: [] for task in graph.tasks}
        for task in graph.tasks:
            for dependency in task.depends_on:
                descendants[dependency].append(task.task_id)

        queue = deque(task_ids)
        affected: list[str] = []
        while queue:
            task_id = queue.popleft()
            if task_id in affected:
                continue
            graph.get(task_id)
            affected.append(task_id)
            if cascade:
                queue.extend(descendants[task_id])

        updated = graph
        for task_id in affected:
            task = updated.get(task_id)
            if task.status == "invalidated":
                continue
            updated = self.transition(updated, task_id, "invalidated")
        return updated

    def reopen_invalidated(self, graph: TaskGraph, task_ids: list[str]) -> TaskGraph:
        updated = graph
        for task_id in task_ids:
            updated = self.transition(updated, task_id, "pending")
        return self.refresh_ready(updated)
