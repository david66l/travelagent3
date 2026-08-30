"""Tests for deterministic graph planning and scheduling."""

import pytest

from agentic.planner import DefaultTaskGraphPlanner
from agentic.scheduler import TaskScheduler
from agentic.state import GoalLedger, TaskGraph, TaskGraphController, TaskNode


def test_default_graph_keeps_solver_and_validator_gates():
    goal = GoalLedger(original_request="Three days in Shanghai")
    graph = DefaultTaskGraphPlanner().plan(goal)

    DefaultTaskGraphPlanner.ensure_mandatory_gates(graph)
    assert graph.get("validate_itinerary").depends_on == ("solve_itinerary",)
    assert graph.get("review_itinerary").depends_on == ("validate_itinerary",)
    assert graph.get("compose_draft").depends_on == ("review_itinerary",)


def test_missing_user_information_becomes_explicit_task():
    goal = GoalLedger(
        original_request="Plan my trip",
        missing_information=["destination", "travel_dates"],
    )
    graph = DefaultTaskGraphPlanner().plan(goal)

    task = graph.get("resolve_missing_information")
    assert task.allowed_actions == ("ask_user",)
    assert graph.get("search_candidates").depends_on == (task.task_id,)


def test_mandatory_gate_removal_is_rejected():
    graph = TaskGraph(
        goal_version=1,
        tasks=(TaskNode(task_id="capability_check", goal="check"),),
    )
    with pytest.raises(ValueError, match="mandatory gates"):
        DefaultTaskGraphPlanner.ensure_mandatory_gates(graph)


def test_scheduler_parallelizes_only_independent_read_tasks():
    graph = TaskGraph(
        goal_version=1,
        tasks=(
            TaskNode(task_id="weather", goal="weather", allowed_actions=("get_weather",)),
            TaskNode(task_id="hotel", goal="hotel", allowed_actions=("find_hotels",)),
            TaskNode(task_id="solve", goal="solve", allowed_actions=("solve_itinerary",)),
        ),
    )
    graph, batch = TaskScheduler().select(graph)

    assert batch is not None
    assert batch.mode == "parallel"
    assert batch.task_ids == ["weather", "hotel"]


def test_scheduler_waits_while_a_task_is_running():
    controller = TaskGraphController()
    graph = TaskGraph(
        goal_version=1,
        tasks=(TaskNode(task_id="weather", goal="weather", allowed_actions=("get_weather",)),),
    )
    graph = controller.refresh_ready(graph)
    graph = controller.transition(graph, "weather", "running")

    unchanged, batch = TaskScheduler().select(graph)
    assert unchanged == graph
    assert batch is None
