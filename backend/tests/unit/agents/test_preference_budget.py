"""Tests for PreferenceBudgetAgent."""

import pytest
from schemas import UserProfile, DayPlan, Activity, BudgetPanel
from agents.preference_budget import PreferenceBudgetAgent, CITY_FACTORS, BUFFER_RATES


class TestPreferenceBudgetAgent:
    """Test preference updates and budget calculations."""

    def setup_method(self):
        self.agent = PreferenceBudgetAgent()

    # === Preference Update Tests ===

    def test_update_single_field(self):
        profile = UserProfile(destination="北京", travel_days=3)
        changes = [{"field": "travel_days", "new_value": "5"}]
        result = self.agent.update_preferences(profile, changes)
        assert result.travel_days == 5

    def test_update_list_field_append(self):
        profile = UserProfile(food_preferences=["辣"])
        changes = [{"field": "food_preferences", "new_value": "海鲜"}]
        result = self.agent.update_preferences(profile, changes)
        assert "辣" in result.food_preferences
        assert "海鲜" in result.food_preferences

    def test_update_list_field_duplicate_not_added(self):
        profile = UserProfile(food_preferences=["辣", "海鲜"])
        changes = [{"field": "food_preferences", "new_value": "辣"}]
        result = self.agent.update_preferences(profile, changes)
        assert result.food_preferences.count("辣") == 1

    def test_update_list_field_multiple_values(self):
        profile = UserProfile(interests=["历史"])
        changes = [{"field": "interests", "new_value": ["美食", "拍照"]}]
        result = self.agent.update_preferences(profile, changes)
        assert "历史" in result.interests
        assert "美食" in result.interests
        assert "拍照" in result.interests

    def test_update_type_conversion_int(self):
        profile = UserProfile()
        changes = [{"field": "travel_days", "new_value": "7"}]
        result = self.agent.update_preferences(profile, changes)
        assert result.travel_days == 7
        assert isinstance(result.travel_days, int)

    def test_update_type_conversion_float(self):
        profile = UserProfile()
        changes = [{"field": "budget_range", "new_value": "10000"}]
        result = self.agent.update_preferences(profile, changes)
        assert result.budget_range == 10000.0
        assert isinstance(result.budget_range, float)

    def test_update_pace(self):
        profile = UserProfile(pace="moderate")
        changes = [{"field": "pace", "new_value": "relaxed"}]
        result = self.agent.update_preferences(profile, changes)
        assert result.pace == "relaxed"

    def test_update_empty_changes(self):
        profile = UserProfile(destination="北京")
        result = self.agent.update_preferences(profile, [])
        assert result.destination == "北京"

    def test_update_nonexistent_field_ignored(self):
        profile = UserProfile(destination="北京")
        changes = [{"field": "nonexistent_field", "new_value": "test"}]
        result = self.agent.update_preferences(profile, changes)
        assert result.destination == "北京"

    # === Budget Calculation Tests ===

    def test_calculate_budget_beijing_tier1(self):
        """Beijing uses tier-1 factor 1.3 and buffer rate 0.15."""
        profile = UserProfile(destination="北京", travel_days=2, budget_range=10000)
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="故宫", category="attraction", ticket_price=60),
                    Activity(poi_name="午餐", category="restaurant", meal_cost=80),
                ],
            ),
        ]
        result = self.agent.calculate_budget(itinerary, profile)
        assert result.total_budget == 10000
        assert result.status == "within_budget"
        assert "accommodation" in result.breakdown
        assert "meals" in result.breakdown
        assert "tickets" in result.breakdown
        assert "transport" in result.breakdown
        assert "shopping" in result.breakdown
        assert "buffer" in result.breakdown

    def test_calculate_budget_over_limit(self):
        profile = UserProfile(destination="北京", travel_days=1, budget_range=100)
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", category="attraction", ticket_price=200),
                ],
            ),
        ]
        result = self.agent.calculate_budget(itinerary, profile)
        assert result.status == "over_budget"

    def test_calculate_budget_no_limit(self):
        profile = UserProfile(destination="成都", travel_days=3)
        itinerary = [DayPlan(day_number=1, activities=[])]
        result = self.agent.calculate_budget(itinerary, profile)
        assert result.total_budget is None
        assert result.remaining is None
        assert result.status == "within_budget"

    def test_calculate_budget_empty_itinerary(self):
        profile = UserProfile(destination="杭州", travel_days=2, budget_range=5000)
        result = self.agent.calculate_budget([], profile)
        assert result.breakdown["tickets"] == 0
        assert result.spent > 0  # accommodation + meals + transport + shopping still calculated

    def test_calculate_budget_ticket_sum(self):
        profile = UserProfile(destination="北京", travel_days=1)
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", category="attraction", ticket_price=100),
                    Activity(poi_name="B", category="attraction", ticket_price=50),
                ],
            ),
        ]
        result = self.agent.calculate_budget(itinerary, profile)
        assert result.breakdown["tickets"] == 150

    # === Budget Panel Tests ===

    def test_build_preference_panel(self):
        profile = UserProfile(
            destination="西安",
            travel_days=4,
            food_preferences=["面食", "羊肉"],
            interests=["历史", "博物馆"],
        )
        panel = self.agent.build_preference_panel(profile)
        assert panel["destination"] == "西安"
        assert panel["travel_days"] == 4
        assert "面食" in panel["food_preferences"]

    def test_init_panel_empty(self):
        profile = UserProfile(budget_range=3000)
        panel = self.agent.init_panel(profile)
        assert panel.total_budget == 3000
        assert panel.breakdown["accommodation"] == 0
        assert panel.breakdown["meals"] == 0

    # === City Factors Sanity Check ===

    def test_city_factors_beijing(self):
        assert CITY_FACTORS["北京"] == 1.3

    def test_city_factors_chengdu(self):
        assert CITY_FACTORS["成都"] == 1.0

    def test_buffer_rates_tier1(self):
        assert BUFFER_RATES["一线"] == 0.15
