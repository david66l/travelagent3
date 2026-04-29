"""Preference & Budget Agent - manage preferences and calculate budget."""

from schemas import UserProfile, BudgetPanel, DayPlan


CITY_FACTORS = {
    "北京": 1.3, "上海": 1.3, "广州": 1.2, "深圳": 1.2,
    "杭州": 1.2, "南京": 1.1, "苏州": 1.1, "厦门": 1.1,
    "成都": 1.0, "西安": 1.0, "重庆": 1.0, "武汉": 1.0,
    "长沙": 1.0, "青岛": 1.0, "昆明": 0.9, "桂林": 0.9,
    "三亚": 1.2, "大理": 0.9, "丽江": 0.9, "拉萨": 0.9,
}

BUFFER_RATES = {
    "一线": 0.15, "二线": 0.12, "三线": 0.10,
}

TIER_MAP = {
    "北京": "一线", "上海": "一线",
    "广州": "二线", "深圳": "二线", "杭州": "二线",
}


class PreferenceBudgetAgent:
    """Manage user preferences and calculate budgets."""

    def update_preferences(self, profile: UserProfile, changes: list[dict]) -> UserProfile:
        """Apply preference changes to profile."""
        for change in changes:
            field = change.get("field")
            new_value = change.get("new_value")
            if field and hasattr(profile, field):
                # Handle list fields (append mode)
                if field in ("food_preferences", "interests", "special_requests"):
                    current = getattr(profile, field, [])
                    if isinstance(new_value, str):
                        if new_value not in current:
                            current.append(new_value)
                    elif isinstance(new_value, list):
                        for v in new_value:
                            if v not in current:
                                current.append(v)
                    setattr(profile, field, current)
                else:
                    # Convert type if needed
                    if field in ("travel_days", "travelers_count") and new_value:
                        new_value = int(new_value)
                    elif field == "budget_range" and new_value:
                        new_value = float(new_value)
                    setattr(profile, field, new_value)

        return profile

    def build_preference_panel(self, profile: UserProfile) -> dict:
        """Build preference panel data."""
        return {
            "destination": profile.destination,
            "travel_days": profile.travel_days,
            "travel_dates": profile.travel_dates,
            "travelers_count": profile.travelers_count,
            "travelers_type": profile.travelers_type,
            "budget_range": profile.budget_range,
            "food_preferences": profile.food_preferences,
            "interests": profile.interests,
            "pace": profile.pace,
            "special_requests": profile.special_requests,
        }

    def calculate_budget(self, itinerary: list[DayPlan], profile: UserProfile) -> BudgetPanel:
        """Calculate detailed budget breakdown."""
        city = profile.destination or ""
        days = len(itinerary) if itinerary else (profile.travel_days or 1)
        factor = CITY_FACTORS.get(city, 1.0)
        tier = TIER_MAP.get(city, "二线")
        buffer_rate = BUFFER_RATES.get(tier, 0.12)

        # Accommodation
        accommodation_cost = 300 * factor * days

        # Meals
        meal_per_day = 150 * factor
        meals_cost = meal_per_day * days

        # Tickets
        tickets_cost = sum(
            (a.ticket_price or 50) for day in itinerary for a in day.activities
        )

        # Transport (between cities/activities)
        transport_cost = 50 * factor * days

        # Shopping buffer
        shopping_cost = 200 * factor * days

        # Subtotal
        subtotal = accommodation_cost + meals_cost + tickets_cost + transport_cost + shopping_cost

        # Buffer
        buffer = subtotal * buffer_rate

        total = subtotal + buffer

        return BudgetPanel(
            total_budget=profile.budget_range,
            spent=total,
            remaining=(profile.budget_range - total) if profile.budget_range else None,
            breakdown={
                "accommodation": round(accommodation_cost, 2),
                "meals": round(meals_cost, 2),
                "transport": round(transport_cost, 2),
                "tickets": round(tickets_cost, 2),
                "shopping": round(shopping_cost, 2),
                "buffer": round(buffer, 2),
            },
            status="within_budget" if (not profile.budget_range or total <= profile.budget_range) else "over_budget",
        )

    def init_panel(self, profile: UserProfile) -> BudgetPanel:
        """Initialize empty budget panel."""
        return BudgetPanel(
            total_budget=profile.budget_range,
            breakdown={
                "accommodation": 0,
                "meals": 0,
                "transport": 0,
                "tickets": 0,
                "shopping": 0,
                "buffer": 0,
            },
        )