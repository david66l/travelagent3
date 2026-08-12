"""Tests for legacy-to-Agent-Loop state projection."""

from agentic.runtime import initialize_agent_ledger
from agentic.state import AgentLedgerState


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
