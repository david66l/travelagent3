"""Tests for deterministic graph planning and scheduling."""

from agentic.scheduler import TaskScheduler
from agentic.state import TaskGraph, TaskGraphController, TaskNode


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
