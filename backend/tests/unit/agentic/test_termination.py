"""Tests for the global completion guard."""

from datetime import UTC, datetime, timedelta

from agentic.termination import CompletionGuard
from agentic.state import (
    AgentLedgerState,
    ArtifactRecord,
    FactRecord,
    GoalLedger,
    TaskGraph,
    TaskNode,
)


def test_enforce_requires_validator_report():
    decision = CompletionGuard(mode="enforce").evaluate(None)

    assert decision.allowed is False
    assert decision.would_block is True
    assert decision.blocks[0].code == "VALIDATOR_NOT_RUN"


def test_shadow_reports_failed_constraints_without_blocking():
    decision = CompletionGuard(mode="shadow").evaluate(
        {
            "hard_pass": False,
            "hard_violations": [{"code": "OVER_BUDGET", "message": "budget exceeded"}],
        }
    )

    assert decision.allowed is True
    assert decision.would_block is True
    assert decision.blocks[0].details["violation_codes"] == ["OVER_BUDGET"]


def test_enforce_accepts_programmatic_hard_pass():
    decision = CompletionGuard(mode="enforce").evaluate({"hard_pass": True, "hard_violations": []})

    assert decision.allowed is True
    assert decision.would_block is False
    assert decision.validator_version == "travel-validator.v1"


def _ledger(*, status: str = "succeeded", plan_version: int = 1) -> AgentLedgerState:
    return AgentLedgerState(
        goal=GoalLedger(original_request="Plan a trip"),
        task_graph=TaskGraph(
            goal_version=1,
            plan_version=plan_version,
            tasks=(
                TaskNode(
                    task_id="validate_itinerary",
                    goal="validate",
                    status=status,
                    verifier_evidence_refs=("report-1",) if status == "succeeded" else (),
                ),
            ),
        ),
    )


def test_enforce_rejects_false_finish_with_incomplete_task():
    ledger = _ledger(status="running")
    decision = CompletionGuard(mode="enforce").evaluate(
        {"hard_pass": True, "hard_violations": []}, ledger=ledger
    )

    assert decision.allowed is False
    assert {block.code for block in decision.blocks} == {
        "REQUIRED_TASKS_INCOMPLETE",
        "TASK_GRAPH_UNSTABLE",
    }


def test_enforce_rejects_expired_facts_and_stale_validation():
    ledger = _ledger(plan_version=2)
    ledger.facts["weather"] = FactRecord(
        fact_id="weather",
        key="weather",
        value="sunny",
        observation_ref="obs-1",
        goal_version=1,
        plan_version=1,
        source="api",
        confidence=1,
        expires_at=datetime.now(UTC) - timedelta(seconds=1),
    )
    ledger.artifacts["report"] = ArtifactRecord(
        artifact_id="report",
        artifact_type="validation_report",
        payload={"hard_pass": True},
        goal_version=1,
        plan_version=1,
    )

    decision = CompletionGuard(mode="enforce").evaluate(
        {"hard_pass": True, "hard_violations": []}, ledger=ledger
    )
    assert {block.code for block in decision.blocks} == {
        "FACTS_EXPIRED",
        "STALE_VALIDATION_ARTIFACT",
    }


def test_enforce_accepts_closed_current_ledger():
    ledger = _ledger()
    ledger.artifacts["report"] = ArtifactRecord(
        artifact_id="report",
        artifact_type="validation_report",
        payload={"hard_pass": True},
        goal_version=1,
        plan_version=1,
    )
    decision = CompletionGuard(mode="enforce").evaluate(
        {"hard_pass": True, "hard_violations": []}, ledger=ledger
    )

    assert decision.allowed is True
    assert decision.blocks == []
