"""Tests for itinerary validation logic."""

import pytest
from unittest.mock import AsyncMock, patch
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

    @pytest.mark.asyncio
    async def test_budget_exactly_at_limit(self):
        itinerary = [
            DayPlan(day_number=1, activities=[], total_cost=500),
        ]
        profile = UserProfile(budget_range=500)
        result = await self.agent._check_budget(itinerary, profile)
        assert result["scores"]["budget_compliance"] == 1.0

    @pytest.mark.asyncio
    async def test_budget_no_limit(self):
        itinerary = [DayPlan(day_number=1, activities=[], total_cost=1000)]
        profile = UserProfile()
        result = await self.agent._check_budget(itinerary, profile)
        assert result["scores"]["budget_compliance"] == 1.0


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

    @pytest.mark.asyncio
    async def test_missing_time_fields(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", category="attraction"),
                ],
            )
        ]
        result = await self.agent._check_time_feasibility(itinerary)
        assert result["scores"]["time_feasibility"] == 1.0

    @pytest.mark.asyncio
    async def test_end_after_day_end(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", category="attraction", start_time="20:00", end_time="22:00"),
                ],
            )
        ]
        result = await self.agent._check_time_feasibility(itinerary)
        assert result["scores"]["time_feasibility"] < 1.0


class TestPOIExistence:
    """Test POI verification via search."""

    def setup_method(self):
        self.agent = ValidationAgent()

    @pytest.mark.asyncio
    async def test_poi_verified(self):
        from skills.web_search import SearchResult
        mock_search = AsyncMock(return_value=[
            SearchResult(title="故宫博物院", url="http://example.com", snippet="北京著名景点")
        ])
        with patch.object(self.agent.search_skill, "search", mock_search):
            itinerary = [
                DayPlan(
                    day_number=1,
                    activities=[
                        Activity(poi_name="故宫博物院", category="attraction"),
                    ],
                )
            ]
            profile = UserProfile(destination="北京")
            result = await self.agent._check_poi_existence(itinerary, profile)
            assert result["scores"]["factuality"] == 1.0

    @pytest.mark.asyncio
    async def test_poi_not_found(self):
        mock_search = AsyncMock(return_value=[])
        with patch.object(self.agent.search_skill, "search", mock_search):
            itinerary = [
                DayPlan(
                    day_number=1,
                    activities=[
                        Activity(poi_name="不存在的景点XYZ123", category="attraction"),
                    ],
                )
            ]
            profile = UserProfile(destination="北京")
            result = await self.agent._check_poi_existence(itinerary, profile)
            assert result["scores"]["factuality"] < 1.0

    @pytest.mark.asyncio
    async def test_poi_empty_itinerary(self):
        result = await self.agent._check_poi_existence([], UserProfile())
        assert result["scores"]["factuality"] == 1.0


class TestOpeningHours:
    """Test opening hours validation."""

    def setup_method(self):
        self.agent = ValidationAgent()

    @pytest.mark.asyncio
    async def test_within_opening_hours(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", start_time="09:00", end_time="11:00", open_time="08:00", close_time="17:00"),
                ],
            )
        ]
        result = await self.agent._check_opening_hours(itinerary)
        assert result["scores"]["opening_hours"] == 1.0

    @pytest.mark.asyncio
    async def test_outside_opening_hours(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", start_time="18:00", end_time="20:00", open_time="08:00", close_time="17:00"),
                ],
            )
        ]
        result = await self.agent._check_opening_hours(itinerary)
        assert result["scores"]["opening_hours"] < 1.0

    @pytest.mark.asyncio
    async def test_missing_open_time(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", start_time="09:00", end_time="11:00"),
                ],
            )
        ]
        result = await self.agent._check_opening_hours(itinerary)
        assert result["scores"]["opening_hours"] == 1.0


class TestPreferenceCoverage:
    """Test preference coverage validation."""

    def setup_method(self):
        self.agent = ValidationAgent()

    @pytest.mark.asyncio
    async def test_full_preference_coverage(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", category="attraction", tags=["历史", "文化"]),
                    Activity(poi_name="B", category="restaurant", tags=["美食"]),
                ],
            )
        ]
        profile = UserProfile(interests=["历史"], food_preferences=["美食"])
        result = await self.agent._check_preference_coverage(itinerary, profile)
        assert result["scores"]["preference_coverage"] == 1.0

    @pytest.mark.asyncio
    async def test_partial_preference_coverage(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", category="attraction", tags=["历史"]),
                ],
            )
        ]
        profile = UserProfile(interests=["历史", "自然", "拍照"])
        result = await self.agent._check_preference_coverage(itinerary, profile)
        assert 0 < result["scores"]["preference_coverage"] < 1.0

    @pytest.mark.asyncio
    async def test_no_user_preferences(self):
        itinerary = [DayPlan(day_number=1, activities=[Activity(poi_name="A", tags=["历史"])])]
        profile = UserProfile()
        result = await self.agent._check_preference_coverage(itinerary, profile)
        assert result["scores"]["preference_coverage"] == 1.0


class TestFullValidation:
    """Test the full validate() orchestration."""

    def setup_method(self):
        self.agent = ValidationAgent()

    @pytest.mark.asyncio
    async def test_validate_passes(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", start_time="09:00", end_time="11:00"),
                ],
                total_cost=100,
            )
        ]
        profile = UserProfile(budget_range=500)
        with patch.object(self.agent.search_skill, "search", AsyncMock(return_value=[])):
            result = await self.agent.validate(itinerary, profile)
            assert isinstance(result.passed, bool)
            assert "budget_compliance" in result.scores
            assert "time_feasibility" in result.scores

    @pytest.mark.asyncio
    async def test_validate_with_critical_failure(self):
        itinerary = [
            DayPlan(
                day_number=1,
                activities=[
                    Activity(poi_name="A", start_time="09:00", end_time="11:00"),
                ],
                total_cost=1000,
            )
        ]
        profile = UserProfile(budget_range=100)
        with patch.object(self.agent.search_skill, "search", AsyncMock(return_value=[])):
            result = await self.agent.validate(itinerary, profile)
            assert result.passed is False
            assert len(result.critical_failures) > 0
