"""Trigger-based minimal invalidation for long-horizon plans."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic.state import AgentLedgerState, PlanVersion, TaskGraphController


class ReplanDecision(BaseModel):
    triggered: bool
    trigger: str
    evidence_refs: list[str] = Field(default_factory=list)
    invalidated_task_ids: list[str] = Field(default_factory=list)
    preserved_task_ids: list[str] = Field(default_factory=list)
    changed_constraints: dict = Field(default_factory=dict)
    requires_user_confirmation: bool = False
    new_plan_version: int | None = None


class ReplanDecider:
    """Invalidate the smallest dependency-closed subgraph matching an event."""

    def __init__(self) -> None:
        self.controller = TaskGraphController()

    def decide(
        self,
        ledger: AgentLedgerState,
        *,
        trigger: str,
        evidence_refs: list[str] | None = None,
        changed_constraints: dict | None = None,
        requires_user_confirmation: bool = False,
    ) -> ReplanDecision:
        roots = [task.task_id for task in ledger.task_graph.tasks if trigger in task.invalidates_on]
        if not roots:
            return ReplanDecision(triggered=False, trigger=trigger)

        invalidated_graph = self.controller.invalidate(ledger.task_graph, roots, cascade=True)
        invalidated = [
            task.task_id
            for task in invalidated_graph.tasks
            if task.status == "invalidated"
            and ledger.task_graph.get(task.task_id).status != "invalidated"
        ]
        preserved = [
            task.task_id for task in ledger.task_graph.tasks if task.task_id not in invalidated
        ]
        new_version = ledger.task_graph.plan_version + 1
        ledger.task_graph = invalidated_graph.model_copy(update={"plan_version": new_version})
        ledger.plan_versions.append(
            PlanVersion(
                plan_version=new_version,
                goal_version=ledger.goal.goal_version,
                trigger=trigger,
                evidence_refs=evidence_refs or [],
                invalidated_task_ids=invalidated,
                preserved_task_ids=preserved,
                changed_constraints=changed_constraints or {},
            )
        )
        return ReplanDecision(
            triggered=True,
            trigger=trigger,
            evidence_refs=evidence_refs or [],
            invalidated_task_ids=invalidated,
            preserved_task_ids=preserved,
            changed_constraints=changed_constraints or {},
            requires_user_confirmation=requires_user_confirmation,
            new_plan_version=new_version,
        )

    def apply(self, ledger: AgentLedgerState, decision: ReplanDecision) -> None:
        if not decision.triggered:
            return
        ledger.task_graph = self.controller.reopen_invalidated(
            ledger.task_graph, decision.invalidated_task_ids
        )
