"""Inject lunch / dinner dummy nodes when explicitly opted in."""

from __future__ import annotations

import logging
import uuid

from vrp_solver_service.models import POIInput

logger = logging.getLogger(__name__)


class RestaurantHandler:
    """Inject dining dummy POIs per travel day (opt-in)."""

    def inject(self, pois: list[POIInput], constraints) -> list[POIInput]:
        """Add meal dummy nodes only when include_restaurant=True and meals_per_day>0."""
        if not constraints.include_restaurant or constraints.meals_per_day <= 0:
            return pois

        meals_per_day = min(constraints.meals_per_day, 2)
        new_pois = list(pois)
        windows = [constraints.lunch_window, constraints.dinner_window]
        for d in range(constraints.travel_days):
            for meal in range(1, meals_per_day + 1):
                start_min, end_min = windows[meal - 1]
                start = f"{start_min // 60:02d}:{start_min % 60:02d}"
                end = f"{end_min // 60:02d}:{end_min % 60:02d}"
                node = POIInput(
                    id=f"__meal_d{d + 1}_m{meal}_{uuid.uuid4().hex[:6]}",
                    name="用餐",
                    category="restaurant",
                    duration_minutes=end_min - start_min,
                    open_time=start,
                    close_time=end,
                    ticket_price=constraints.food_day / (constraints.travel_days * meals_per_day),
                    walk_intensity=0,
                )
                new_pois.append(node)
        return new_pois
