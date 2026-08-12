"""CP-SAT parameter tuning based on problem scale."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CPSATTuningGuide:
    """Recommend CP-SAT search parameters for a given instance."""

    def recommend(self, constraints, poi_count: int) -> dict:
        """Return solver parameter dict.

        v4.0 spec: use 4 search workers and a time limit scaled by instance size.
        """
        nodes = poi_count - 1  # excluding hotel
        days = constraints.travel_days
        arcs = nodes * nodes * days

        # Small instances find a proven-optimal solution in a few seconds. Large
        # ones (5-day, ~25+ POI) were stopping at the 12s cap with status=FEASIBLE
        # — i.e. the incumbent was still improving when time ran out, so the day
        # grouping was visibly sub-optimal. Single-worker CP-SAT keeps improving
        # the incumbent monotonically, so giving big instances more wall time
        # directly buys a better route (the extra seconds are not just optimality
        # proof here). Tiers raised; small trips stay snappy.
        if arcs <= 900:  # <= ~15 POIs, 3 days
            time_limit = 4.0
        elif arcs <= 3600:
            time_limit = 10.0
        elif arcs <= 8000:
            time_limit = 18.0
        else:
            time_limit = 25.0

        return {
            "max_time_in_seconds": time_limit,
            # Single worker on purpose: multi-worker CP-SAT (>=2) dead-hangs on
            # this platform (macOS + ortools 9.15), and crucially the hang occurs
            # before search starts, so `max_time_in_seconds` never fires and the
            # request blocks indefinitely. A single deterministic worker solves
            # these small instances (<=30 POIs) in well under the time limit.
            "num_search_workers": 1,
            "log_search_progress": False,
            "maximize": True,
        }
