"""Tests for itinerary validation logic."""

import pytest
from agents.validation import ValidationAgent
from schemas import DayPlan, Activity, UserProfile


class TestBudgetCheck:
    """Test budget validation."""

    def setup_method(self):
        self.agent = ValidationAgent()

    @pytest.mark.asyncio
    async def test_within_budget(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="景点A", category="attraction", ticket_price=100),
                    Activity(poi_name="餐厅B", category="restaurant", meal_cost=80),
                ],
                total_cost=180,
            )
        ]
        profile = UserProfile(budget_range=500)
        result = await self.agent._check_budget(itinerary, profile)
        assert result["scores"]["budget_compliance"] == 1.0
        assert len(result["critical_failures"]) == 0

    @pytest.mark.asyncio
    async def test_over_budget(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="景点A", category="attraction", ticket_price=300),
                ],
                total_cost=300,
            )
        ]
        profile = UserProfile(budget_range=200)
        result = await self.agent._check_budget(itinerary, profile)
        assert result["scores"]["budget_compliance"] < 1.0
        assert "budget_compliance" in result["critical_failures"]


class TestTimeFeasibility:
    """Test time conflict detection."""

    def setup_method(self):
        self.agent = ValidationAgent()

    @pytest.mark.asyncio
    async def test_valid_schedule(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", category="attraction", start_time="09:00", end_time="11:00"),
                    Activity(poi_name="B", category="attraction", start_time="11:30", end_time="13:00"),
                ],
            )
        ]
        result = await self.agent._check_time_feasibility(itinerary)
        assert result["scores"]["time_feasibility"] == 1.0

    @pytest.mark.asyncio
    async def test_overlapping_activities(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", category="attraction", start_time="09:00", end_time="11:00"),
                    Activity(poi_name="B", category="attraction", start_time="10:30", end_time="12:00"),
                ],
            )
        ]
        result = await self.agent._check_time_feasibility(itinerary)
        assert result["scores"]["time_feasibility"] < 1.0

    @pytest.mark.asyncio
    async def test_out_of_bounds(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", category="attraction", start_time="08:00", end_time="10:00"),
                ],
            )
        ]
        result = await self.agent._check_time_feasibility(itinerary)
        assert result["scores"]["time_feasibility"] < 1.0
