"""OR-Tools CP-SAT callbacks for the VRP solver service."""

from __future__ import annotations

import time

from ortools.sat.python import cp_model


class TimeoutCallback(cp_model.CpSolverSolutionCallback):
    """Stop search on a wall-clock cap OR once the incumbent has converged.

    Convergence early-stop: large 5-day instances kept improving the objective by
    tiny amounts for many seconds and then spent the rest of the budget *proving*
    optimality (which the user never sees), running the full ~18s cap. Once a fresh
    solution improves the objective by less than ``rel_improve_stop`` (relative) and
    a minimum-time floor has passed, the plan has effectively settled, so we stop —
    trimming the long tail without dropping solution quality. The floor guarantees
    we never bail before giving the solver a real chance, so small instances that
    already finish in <1s and quality on hard instances are unaffected.
    """

    def __init__(
        self,
        time_limit_seconds: float,
        min_seconds: float = 0.0,
        rel_improve_stop: float = 0.0,
    ):
        super().__init__()
        self._time_limit = time_limit_seconds
        self._min_seconds = min_seconds
        self._rel_improve_stop = rel_improve_stop
        self._start = time.time()
        self._solution_count = 0
        self._best_objective: int | None = None

    def on_solution_callback(self) -> None:
        self._solution_count += 1
        obj = int(self.ObjectiveValue())
        prev = self._best_objective
        self._best_objective = obj

        elapsed = time.time() - self._start
        if elapsed >= self._time_limit:
            self.StopSearch()
            return

        # Converged: the latest improvement is negligible after the min-time floor.
        if (
            self._rel_improve_stop > 0
            and prev is not None
            and elapsed >= self._min_seconds
        ):
            denom = abs(prev) or 1
            if abs(obj - prev) / denom < self._rel_improve_stop:
                self.StopSearch()

    @property
    def solution_count(self) -> int:
        return self._solution_count

    @property
    def best_objective(self) -> int | None:
        return self._best_objective
