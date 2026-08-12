"""Inject lunch / dinner dummy nodes when explicitly opted in."""

from __future__ import annotations

import logging
import uuid

from vrp_solver_service.models import POIInput

logger = logging.getLogger(__name__)


class RestaurantHandler:
    """Inject dining dummy POIs per travel day (opt-in)."""

    # A meal is a fixed-length block that can slot anywhere inside its window;
    # it must NOT fill the whole window, or the time-window constraint
    # (arrival >= open AND arrival + duration <= close) pins it to a single
    # instant and the solver drops it as infeasible.
    _MEAL_MINUTES = (60, 90)  # lunch, dinner

    def inject(self, pois: list[POIInput], constraints) -> list[POIInput]:
        """Add meal dummy nodes only when include_restaurant=True and meals_per_day>0."""
        if not constraints.include_restaurant or constraints.meals_per_day <= 0:
            return pois

        meals_per_day = min(constraints.meals_per_day, 2)
        meal_names = ("午餐", "晚餐")
        new_pois = list(pois)
        windows = [constraints.lunch_window, constraints.dinner_window]
        for d in range(constraints.travel_days):
            for meal in range(1, meals_per_day + 1):
                start_min, end_min = windows[meal - 1]
                # Cap the meal length so it fits with slack inside the window.
                duration = min(self._MEAL_MINUTES[meal - 1], max(30, end_min - start_min))
                start = f"{start_min // 60:02d}:{start_min % 60:02d}"
                end = f"{end_min // 60:02d}:{end_min % 60:02d}"
                node = POIInput(
                    id=f"__meal_d{d + 1}_m{meal}_{uuid.uuid4().hex[:6]}",
                    name=meal_names[meal - 1],
                    category="restaurant",
                    duration_minutes=duration,
                    min_play_time=duration,
                    max_play_time=duration,
                    open_time=start,
                    close_time=end,
                    # A meal node is a time block only; the food spend is already
                    # represented by the per-day ``food_day`` term in the cost
                    # model. Giving it a ticket price too would double-count food
                    # and distort the daily-cost balancing objective.
                    ticket_price=0.0,
                    walk_intensity=0,
                )
                new_pois.append(node)
        return new_pois
