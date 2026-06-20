"""OR-Tools CP-SAT callbacks for the VRP solver service."""

from __future__ import annotations

import time

from ortools.sat.python import cp_model


class TimeoutCallback(cp_model.CpSolverSolutionCallback):
    """Stop search after a wall-clock time limit and track the best objective seen."""

    def __init__(self, time_limit_seconds: float):
        super().__init__()
        self._time_limit = time_limit_seconds
        self._start = time.time()
        self._solution_count = 0
        self._best_objective: int | None = None

    def on_solution_callback(self) -> None:
        self._solution_count += 1
        self._best_objective = int(self.ObjectiveValue())
        if time.time() - self._start >= self._time_limit:
            self.StopSearch()

    @property
    def solution_count(self) -> int:
        return self._solution_count

    @property
    def best_objective(self) -> int | None:
        return self._best_objective
