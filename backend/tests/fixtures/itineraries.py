"""Test data factories for itineraries and day plans."""

from schemas import DayPlan, Activity


def make_activity(
    poi_name: str = "景点A",
    category: str = "attraction",
    start_time: str = "09:00",
    end_time: str = "11:00",
    duration_min: int = 120,
    ticket_price: float | None = None,
    meal_cost: float | None = None,
    **overrides,
) -> Activity:
    """Factory for constructing Activity test data."""
    return Activity(
        poi_name=poi_name,
        category=category,
        start_time=start_time,
        end_time=end_time,
        duration_min=duration_min,
        ticket_price=ticket_price,
        meal_cost=meal_cost,
        **overrides,
    )


def make_day_plan(
    day_number: int = 1,
    activities: list[Activity] | None = None,
    **overrides,
) -> DayPlan:
    """Factory for constructing DayPlan test data."""
    return DayPlan(
        day_number=day_number,
        activities=activities or [],
        **overrides,
    )
