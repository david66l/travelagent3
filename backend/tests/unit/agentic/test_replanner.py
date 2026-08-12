"""Tests for trigger-based local replanning."""

from agentic.planner import DefaultTaskGraphPlanner
from agentic.replanner import ReplanDecider
from agentic.state import AgentLedgerState, GoalLedger, TaskGraphController


def _ledger() -> AgentLedgerState:
    goal = GoalLedger(original_request="Plan Shanghai")
    graph = DefaultTaskGraphPlanner().plan(goal)
    controller = TaskGraphController()
    graph = controller.refresh_ready(graph)
    graph = controller.transition(graph, "capability_check", "running")
    graph = controller.transition(
        graph, "capability_check", "succeeded", evidence_refs=["capability-1"]
    )
    graph = controller.refresh_ready(graph)
    graph = controller.transition(graph, "search_candidates", "running")
    graph = controller.transition(
        graph, "search_candidates", "succeeded", evidence_refs=["candidates-1"]
    )
    graph = controller.refresh_ready(graph)
    return AgentLedgerState(goal=goal, task_graph=graph)


def test_irrelevant_event_preserves_entire_graph():
    ledger = _ledger()
    before = ledger.task_graph.model_dump()

    decision = ReplanDecider().decide(ledger, trigger="unrelated_event")

    assert decision.triggered is False
    assert ledger.task_graph.model_dump() == before


def test_transport_change_invalidates_route_and_downstream_only():
    ledger = _ledger()
    decider = ReplanDecider()

    decision = decider.decide(
        ledger,
        trigger="transport_mode_changed",
        evidence_refs=["user-event-1"],
        changed_constraints={"transport_mode": "public_transit"},
    )

    assert decision.triggered is True
    assert decision.new_plan_version == 2
    assert set(decision.invalidated_task_ids) == {
        "collect_route_matrix",
        "solve_itinerary",
        "validate_itinerary",
        "compose_draft",
        "await_confirmation",
    }
    assert "collect_weather" in decision.preserved_task_ids
    assert "collect_poi_details" in decision.preserved_task_ids
    assert ledger.plan_versions[-1].evidence_refs == ["user-event-1"]


def test_apply_reopens_invalidated_roots_but_respects_dependencies():
    ledger = _ledger()
    decider = ReplanDecider()
    decision = decider.decide(ledger, trigger="transport_mode_changed")

    decider.apply(ledger, decision)

    assert ledger.task_graph.get("collect_route_matrix").status == "ready"
    assert ledger.task_graph.get("solve_itinerary").status == "pending"
    assert ledger.task_graph.get("validate_itinerary").status == "pending"
