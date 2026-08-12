"""Adapters that project the existing TravelAgent state into Agent Loop state."""

from __future__ import annotations

from typing import Any, Literal

from agentic.planner import DefaultTaskGraphPlanner
from agentic.state import (
    AgentLedgerState,
    FactRecord,
    GoalCapability,
    GoalLedger,
    StateTransitionError,
    TaskGraphController,
)
from agentic.verifier import SubtaskVerifier
from agentic.termination import CompletionDecision, CompletionGuard
from core.conversation_state import flatten_profile


PolicyMode = Literal["deterministic", "shadow", "agent"]


def initialize_agent_ledger(state: dict[str, Any], *, mode: PolicyMode) -> dict[str, Any]:
    """Initialize authoritative state without changing the legacy output path."""
    if mode == "deterministic":
        return {"policy_mode": mode, "agent_status": "disabled"}
    if state.get("agent_ledger"):
        ledger = AgentLedgerState(**state["agent_ledger"])
        ready = TaskGraphController.ready_tasks(ledger.task_graph)
        return {
            "policy_mode": mode,
            "agent_ledger": ledger.model_dump(mode="json"),
            "current_task_id": ready[0].task_id if ready else ledger.current_task_id,
            "agent_status": "initialized",
        }

    flat = flatten_profile(state.get("profile") or {})
    slots = state.get("slots") or {}

    def value(key: str) -> Any:
        slot_value = slots.get(key)
        return slot_value if slot_value not in (None, "", []) else flat.get(key)

    hard_constraints = {
        key: item
        for key in (
            "destination",
            "travel_days",
            "start_date",
            "end_date",
            "budget_range",
            "must_visit",
            "mobility_constraints",
        )
        if (item := value(key)) not in (None, "", [])
    }
    soft_preferences = {
        key: item
        for key in (
            "interests",
            "food_preferences",
            "transport_preference",
            "hotel_preference",
            "pace_preference",
        )
        if (item := value(key)) not in (None, "", [])
    }
    feasibility = state.get("feasibility_report") or {}
    feasible = feasibility.get("feasible", True)
    missing = list(state.get("missing_slots") or [])
    capability = GoalCapability(
        status="needs_user" if missing else ("solvable" if feasible else "infeasible"),
        evidence=[str(item) for item in feasibility.get("reasons", [])],
    )
    goal = GoalLedger(
        original_request=str(state.get("user_input") or "Travel planning request"),
        success_definition=[
            "generate an executable itinerary",
            "pass deterministic hard-constraint validation",
            "wait for user confirmation",
        ],
        hard_constraints=hard_constraints,
        soft_preferences=soft_preferences,
        missing_information=missing,
        capability=capability,
    )
    graph = DefaultTaskGraphPlanner().plan(goal)
    graph = TaskGraphController().refresh_ready(graph)
    ready = TaskGraphController.ready_tasks(graph)
    ledger = AgentLedgerState(goal=goal, task_graph=graph)
    return {
        "policy_mode": mode,
        "agent_ledger": ledger.model_dump(mode="json"),
        "current_task_id": ready[0].task_id if ready else None,
        "agent_step": 0,
        "subtask_step": 0,
        "agent_status": "initialized",
    }


def resume_agent_ledger(
    ledger: AgentLedgerState | dict[str, Any],
    *,
    task_id: str,
    user_value: Any,
    fact_key: str,
) -> AgentLedgerState:
    """Resume a blocked task without resetting graph versions or budgets."""
    state = ledger if isinstance(ledger, AgentLedgerState) else AgentLedgerState(**ledger)
    task = state.task_graph.get(task_id)
    if task.status != "blocked":
        raise StateTransitionError("only a blocked task can be resumed by user input")
    observation_ref = f"user:{state.trajectory_id}:{task_id}:{task.attempts}"
    fact = FactRecord(
        fact_id=observation_ref,
        key=fact_key,
        value=user_value,
        observation_ref=observation_ref,
        goal_version=state.goal.goal_version,
        plan_version=state.task_graph.plan_version,
        source="user",
        confidence=1.0,
    )
    state.facts[fact.fact_id] = fact
    controller = TaskGraphController()
    state.task_graph = controller.transition(
        state.task_graph, task_id, "ready", evidence_refs=[observation_ref]
    )
    state.task_graph = controller.transition(state.task_graph, task_id, "running")
    verification = SubtaskVerifier().verify(
        state.task_graph.get(task_id), facts=state.facts, artifacts=state.artifacts
    )
    if verification.passed:
        state.task_graph = controller.transition(
            state.task_graph,
            task_id,
            "succeeded",
            evidence_refs=verification.evidence_refs,
        )
        state.task_graph = controller.refresh_ready(state.task_graph)
        ready = controller.ready_tasks(state.task_graph)
        state.current_task_id = ready[0].task_id if ready else None
    else:
        state.task_graph = controller.transition(state.task_graph, task_id, "blocked")
        state.current_task_id = task_id
    state.termination_reason = None
    return state


def confirm_agent_ledger(
    ledger: AgentLedgerState | dict[str, Any],
) -> tuple[AgentLedgerState, CompletionDecision]:
    """Close the confirmation task and enforce the global completion gate."""
    state = ledger if isinstance(ledger, AgentLedgerState) else AgentLedgerState(**ledger)
    state = resume_agent_ledger(
        state,
        task_id="await_confirmation",
        user_value=True,
        fact_key="user_confirmation",
    )
    reports = [
        artifact
        for artifact in state.artifacts.values()
        if artifact.artifact_type == "validation_report"
        and artifact.goal_version == state.goal.goal_version
        and artifact.plan_version == state.task_graph.plan_version
    ]
    report = reports[-1].payload if reports else None
    decision = CompletionGuard(mode="enforce").evaluate(report, ledger=state)
    if not decision.allowed:
        codes = ", ".join(block.code for block in decision.blocks)
        raise StateTransitionError(f"global completion guard rejected confirmation: {codes}")
    state.termination_reason = "validated_finish"
    return state, decision
