"""Tests for authoritative long-horizon agent state."""

import pytest
from pydantic import ValidationError

from agentic.state import (
    AgentLedgerState,
    BudgetExceeded,
    BudgetLedger,
    GoalLedger,
    StateTransitionError,
    TaskGraph,
    TaskGraphController,
    TaskNode,
)


def _graph() -> TaskGraph:
    return TaskGraph(
        goal_version=1,
        tasks=(
            TaskNode(task_id="search", goal="find candidates", allowed_actions=("search_pois",)),
            TaskNode(
                task_id="solve",
                goal="solve itinerary",
                depends_on=("search",),
                allowed_actions=("solve_itinerary",),
            ),
            TaskNode(
                task_id="validate",
                goal="validate itinerary",
                depends_on=("solve",),
                allowed_actions=("validate_itinerary",),
            ),
        ),
    )


def test_task_graph_rejects_missing_dependencies_and_cycles():
    with pytest.raises(ValidationError, match="missing dependencies"):
        TaskGraph(
            goal_version=1,
            tasks=(TaskNode(task_id="a", goal="a", depends_on=("missing",)),),
        )

    with pytest.raises(ValidationError, match="acyclic"):
        TaskGraph(
            goal_version=1,
            tasks=(
                TaskNode(task_id="a", goal="a", depends_on=("b",)),
                TaskNode(task_id="b", goal="b", depends_on=("a",)),
            ),
        )


def test_controller_only_unlocks_tasks_after_verified_dependency():
    controller = TaskGraphController()
    graph = controller.refresh_ready(_graph())
    assert [task.task_id for task in controller.ready_tasks(graph)] == ["search"]

    graph = controller.transition(graph, "search", "running")
    with pytest.raises(StateTransitionError, match="verifier evidence"):
        controller.transition(graph, "search", "succeeded")

    graph = controller.transition(graph, "search", "succeeded", evidence_refs=["obs-1"])
    graph = controller.refresh_ready(graph)
    assert graph.get("solve").status == "ready"
    assert graph.get("validate").status == "pending"


def test_retry_budget_turns_repeated_failure_terminal():
    controller = TaskGraphController()
    graph = controller.refresh_ready(_graph())
    graph = controller.transition(graph, "search", "running")
    graph = controller.retry_or_fail(graph, "search", {"code": "TIMEOUT"})
    assert graph.get("search").status == "ready"

    graph = controller.transition(graph, "search", "running")
    graph = controller.retry_or_fail(graph, "search", {"code": "TIMEOUT"})
    assert graph.get("search").status == "failed"


def test_invalidation_propagates_only_to_descendants():
    graph = TaskGraph(
        goal_version=1,
        tasks=_graph().tasks
        + (TaskNode(task_id="hotel", goal="find hotel", allowed_actions=("find_hotels",)),),
    )
    graph = TaskGraphController().invalidate(graph, ["search"])

    assert graph.get("search").status == "invalidated"
    assert graph.get("solve").status == "invalidated"
    assert graph.get("validate").status == "invalidated"
    assert graph.get("hotel").status == "pending"


def test_budget_is_durable_and_rejects_overspend():
    budget = BudgetLedger(max_episode_steps=2, max_tool_calls=1)
    budget = budget.consume(episode_steps=1, tool_calls=1, tokens=100)

    assert budget.remaining_episode_steps == 1
    assert budget.remaining_tool_calls == 0
    with pytest.raises(BudgetExceeded, match="used_tool_calls"):
        budget.consume(tool_calls=1)


def test_agent_ledger_round_trip_preserves_versions():
    state = AgentLedgerState(
        goal=GoalLedger(original_request="Plan a two-day Shanghai trip"),
        task_graph=_graph(),
    )

    restored = AgentLedgerState.model_validate_json(state.model_dump_json())
    assert restored.schema_version == "agentic-state.v1"
    assert restored.task_graph.goal_version == restored.goal.goal_version
