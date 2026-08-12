"""Tests for legacy-to-Agent-Loop state projection."""

import pytest

from agentic.runtime import initialize_agent_ledger, resume_agent_ledger
from agentic.state import AgentLedgerState, StateTransitionError, TaskGraphController


def test_deterministic_mode_does_not_create_agent_state():
    result = initialize_agent_ledger({"user_input": "Plan Shanghai"}, mode="deterministic")

    assert result == {"policy_mode": "deterministic", "agent_status": "disabled"}


def test_shadow_mode_creates_versioned_goal_and_default_dag():
    result = initialize_agent_ledger(
        {
            "user_input": "Shanghai for three days under 5000",
            "slots": {
                "destination": "Shanghai",
                "travel_days": 3,
                "budget_range": 5000,
                "interests": ["museum"],
            },
            "missing_slots": [],
        },
        mode="shadow",
    )
    ledger = AgentLedgerState(**result["agent_ledger"])

    assert result["agent_status"] == "initialized"
    assert result["current_task_id"] == "capability_check"
    assert ledger.goal.hard_constraints["destination"] == "Shanghai"
    assert ledger.goal.soft_preferences["interests"] == ["museum"]
    assert ledger.task_graph.get("capability_check").status == "ready"


def test_existing_ledger_is_resumed_instead_of_reset():
    initial = initialize_agent_ledger(
        {"user_input": "Plan Shanghai", "slots": {"destination": "Shanghai"}},
        mode="shadow",
    )
    ledger = AgentLedgerState(**initial["agent_ledger"])
    ledger.budget = ledger.budget.consume(episode_steps=2)

    resumed = initialize_agent_ledger(
        {"user_input": "ignored", "agent_ledger": ledger.model_dump(mode="json")},
        mode="shadow",
    )

    assert AgentLedgerState(**resumed["agent_ledger"]).budget.used_episode_steps == 2


def test_user_response_resumes_blocked_task_and_preserves_budget():
    initial = initialize_agent_ledger(
        {
            "user_input": "Plan my trip",
            "missing_slots": ["destination", "travel_dates"],
        },
        mode="shadow",
    )
    ledger = AgentLedgerState(**initial["agent_ledger"])
    controller = TaskGraphController()
    ledger.task_graph = controller.transition(ledger.task_graph, "capability_check", "running")
    ledger.task_graph = controller.transition(
        ledger.task_graph,
        "capability_check",
        "succeeded",
        evidence_refs=["capability-1"],
    )
    ledger.task_graph = controller.refresh_ready(ledger.task_graph)
    ledger.task_graph = controller.transition(
        ledger.task_graph, "resolve_missing_information", "running"
    )
    ledger.task_graph = controller.transition(
        ledger.task_graph, "resolve_missing_information", "blocked"
    )
    ledger.budget = ledger.budget.consume(episode_steps=2)

    resumed = resume_agent_ledger(
        ledger,
        task_id="resolve_missing_information",
        user_value="Shanghai",
        fact_key="user_input.destination",
    )

    assert resumed.task_graph.get("resolve_missing_information").status == "blocked"
    assert resumed.budget.used_episode_steps == 2
    assert next(iter(resumed.facts.values())).value == "Shanghai"


def test_complete_user_response_is_verified_and_unlocks_dependents():
    initial = initialize_agent_ledger(
        {"user_input": "Plan my trip", "missing_slots": ["destination"]},
        mode="shadow",
    )
    ledger = AgentLedgerState(**initial["agent_ledger"])
    controller = TaskGraphController()
    ledger.task_graph = controller.transition(ledger.task_graph, "capability_check", "running")
    ledger.task_graph = controller.transition(
        ledger.task_graph,
        "capability_check",
        "succeeded",
        evidence_refs=["capability-1"],
    )
    ledger.task_graph = controller.refresh_ready(ledger.task_graph)
    ledger.task_graph = controller.transition(
        ledger.task_graph, "resolve_missing_information", "running"
    )
    ledger.task_graph = controller.transition(
        ledger.task_graph, "resolve_missing_information", "blocked"
    )

    resumed = resume_agent_ledger(
        ledger,
        task_id="resolve_missing_information",
        user_value="Shanghai",
        fact_key="user_input.destination",
    )

    assert resumed.task_graph.get("resolve_missing_information").status == "succeeded"
    assert resumed.task_graph.get("collect_weather").status == "ready"


def test_user_response_cannot_resume_non_blocked_task():
    initial = initialize_agent_ledger({"user_input": "Plan Shanghai"}, mode="shadow")
    with pytest.raises(StateTransitionError, match="blocked"):
        resume_agent_ledger(
            initial["agent_ledger"],
            task_id="capability_check",
            user_value="x",
            fact_key="x",
        )
