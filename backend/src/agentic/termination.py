"""Global completion guard for online execution and training episodes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from agentic.guard import GuardMode
from agentic.state import AgentLedgerState
from evaluation.validator import ValidationReport


class CompletionBlock(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class CompletionDecision(BaseModel):
    mode: GuardMode
    allowed: bool
    would_block: bool = False
    blocks: list[CompletionBlock] = Field(default_factory=list)
    validator_version: str | None = None


class CompletionGuard:
    """Require a programmatic hard-pass before accepting ``finish``."""

    def __init__(self, mode: GuardMode = "shadow") -> None:
        self.mode = mode

    def evaluate(
        self,
        report: ValidationReport | dict[str, Any] | None,
        *,
        ledger: AgentLedgerState | dict[str, Any] | None = None,
        now: datetime | None = None,
    ) -> CompletionDecision:
        if self.mode == "off":
            return CompletionDecision(mode="off", allowed=True)

        blocks: list[CompletionBlock] = []
        parsed: ValidationReport | None = None
        if report is None:
            blocks.append(
                CompletionBlock(
                    code="VALIDATOR_NOT_RUN",
                    message="validate_itinerary must run before finish",
                )
            )
        else:
            parsed = report if isinstance(report, ValidationReport) else ValidationReport(**report)
            if not parsed.hard_pass:
                blocks.append(
                    CompletionBlock(
                        code="HARD_CONSTRAINT_FAILED",
                        message="itinerary has unresolved hard-constraint violations",
                        details={
                            "violation_codes": [
                                violation.code for violation in parsed.hard_violations
                            ]
                        },
                    )
                )

        if ledger is not None:
            state = ledger if isinstance(ledger, AgentLedgerState) else AgentLedgerState(**ledger)
            blocks.extend(self._ledger_blocks(state, now=now or datetime.now(UTC)))

        would_block = bool(blocks)
        return CompletionDecision(
            mode=self.mode,
            allowed=not would_block or self.mode == "shadow",
            would_block=would_block,
            blocks=blocks,
            validator_version=parsed.validator_version if parsed else None,
        )

    @staticmethod
    def _ledger_blocks(state: AgentLedgerState, *, now: datetime) -> list[CompletionBlock]:
        blocks: list[CompletionBlock] = []
        incomplete = [
            task.task_id
            for task in state.task_graph.tasks
            if task.required and task.status not in {"succeeded", "skipped"}
        ]
        if incomplete:
            blocks.append(
                CompletionBlock(
                    code="REQUIRED_TASKS_INCOMPLETE",
                    message="required task graph is not closed",
                    details={"task_ids": incomplete},
                )
            )

        unsafe_states = [
            task.task_id
            for task in state.task_graph.tasks
            if task.status in {"running", "blocked", "invalidated"}
        ]
        if unsafe_states:
            blocks.append(
                CompletionBlock(
                    code="TASK_GRAPH_UNSTABLE",
                    message="task graph contains unfinished or invalidated work",
                    details={"task_ids": unsafe_states},
                )
            )

        expired = [
            fact.fact_id
            for fact in state.facts.values()
            if fact.expires_at is not None and fact.expires_at <= now
        ]
        if expired:
            blocks.append(
                CompletionBlock(
                    code="FACTS_EXPIRED",
                    message="one or more facts expired before completion",
                    details={"fact_ids": expired},
                )
            )

        validation_artifacts = [
            artifact
            for artifact in state.artifacts.values()
            if artifact.artifact_type == "validation_report"
        ]
        current_validation = [
            artifact
            for artifact in validation_artifacts
            if artifact.goal_version == state.goal.goal_version
            and artifact.plan_version == state.task_graph.plan_version
        ]
        if validation_artifacts and not current_validation:
            blocks.append(
                CompletionBlock(
                    code="STALE_VALIDATION_ARTIFACT",
                    message="validation report does not belong to the current goal and plan",
                )
            )
        return blocks
