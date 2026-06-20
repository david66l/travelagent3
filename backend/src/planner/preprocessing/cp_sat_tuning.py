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

        if arcs <= 900:  # <= ~15 POIs, 3 days
            time_limit = 5.0
        elif arcs <= 3600:
            time_limit = 15.0
        else:
            time_limit = 30.0

        return {
            "max_time_in_seconds": time_limit,
            "num_search_workers": 4,  # v4.0 fixed 4 workers
            "log_search_progress": False,
            "maximize": True,
        }
