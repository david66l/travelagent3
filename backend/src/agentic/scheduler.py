"""Dependency-aware bounded scheduler for long-horizon tasks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agentic.state import TaskGraph, TaskGraphController, TaskNode


PARALLEL_READ_ACTIONS = frozenset(
    {
        "search_pois",
        "get_poi_detail",
        "check_reservation",
        "get_weather",
        "get_route",
        "get_route_matrix",
        "find_hotels",
        "search_transport",
    }
)


class ScheduledBatch(BaseModel):
    mode: Literal["serial", "parallel"]
    task_ids: list[str] = Field(min_length=1)


class TaskScheduler:
    def __init__(self, *, max_parallel_tasks: int = 4) -> None:
        if max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be positive")
        self.max_parallel_tasks = max_parallel_tasks
        self.controller = TaskGraphController()

    @staticmethod
    def _parallel_safe(task: TaskNode) -> bool:
        return bool(task.allowed_actions) and set(task.allowed_actions) <= PARALLEL_READ_ACTIONS

    def select(self, graph: TaskGraph) -> tuple[TaskGraph, ScheduledBatch | None]:
        if any(task.status == "running" for task in graph.tasks):
            return graph, None

        graph = self.controller.refresh_ready(graph)
        ready = self.controller.ready_tasks(graph)
        if not ready:
            return graph, None

        parallel = [task for task in ready if self._parallel_safe(task)]
        if len(parallel) >= 2:
            selected = parallel[: self.max_parallel_tasks]
            return graph, ScheduledBatch(
                mode="parallel", task_ids=[task.task_id for task in selected]
            )
        return graph, ScheduledBatch(mode="serial", task_ids=[ready[0].task_id])

    def start(self, graph: TaskGraph, batch: ScheduledBatch) -> TaskGraph:
        updated = graph
        for task_id in batch.task_ids:
            updated = self.controller.transition(updated, task_id, "running")
        return updated
