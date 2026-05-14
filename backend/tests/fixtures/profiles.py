"""Test data factories for UserProfile."""

from schemas import UserProfile


def make_profile(
    destination: str = "北京",
    travel_days: int = 3,
    travel_dates: str = "2026-05-01",
    travelers_count: int = 2,
    travelers_type: str = "情侣",
    budget_range: float | None = 5000,
    food_preferences: list[str] | None = None,
    interests: list[str] | None = None,
    pace: str = "moderate",
    accommodation_preference: str | None = None,
    special_requests: list[str] | None = None,
    **overrides,
) -> UserProfile:
    """Factory for constructing UserProfile test data."""
    return UserProfile(
        destination=destination,
        travel_days=travel_days,
        travel_dates=travel_dates,
        travelers_count=travelers_count,
        travelers_type=travelers_type,
        budget_range=budget_range,
        food_preferences=food_preferences or ["辣", "海鲜"],
        interests=interests or ["历史", "美食"],
        pace=pace,
        accommodation_preference=accommodation_preference,
        special_requests=special_requests or [],
        **overrides,
    )
