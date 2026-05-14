"""Extended tests for all LangGraph node functions."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core.state import ItineraryState
from graph.nodes import (
    collect_info_node,
    prepare_context_node,
    poi_search_node,
    weather_node,
    budget_init_node,
    context_enrichment_node,
    planner_node,
    validation_node,
    route_node,
    apply_routes_node,
    budget_calc_node,
    proposal_node,
    update_prefs_node,
    save_memory_node,
    _split_dates,
)
from schemas import UserProfile, DayPlan, Activity, ScoredPOI, Location, WeatherDay


class TestCollectInfoNode:
    """Test collect info node."""

    @pytest.mark.asyncio
    async def test_generate_response(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="我想去旅游",
            missing_required=["destination"],
        )
        result = await collect_info_node(state)
        assert "assistant_response" in result
        assert result["needs_clarification"] is True

    @pytest.mark.asyncio
    async def test_empty_missing(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            missing_required=[],
        )
        result = await collect_info_node(state)
        assert "assistant_response" in result


class TestPrepareContextNode:
    """Test prepare context node."""

    @pytest.mark.asyncio
    async def test_with_user_entities(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_entities={"destination": "北京", "travel_days": 3},
        )
        result = await prepare_context_node(state)
        assert result["user_profile"] is not None
        assert result["preference_panel"] is not None

    @pytest.mark.asyncio
    async def test_with_existing_profile(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_profile={"destination": "成都", "travel_days": 4},
        )
        result = await prepare_context_node(state)
        assert result["user_profile"]["destination"] == "成都"

    @pytest.mark.asyncio
    async def test_empty_profile(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
        )
        result = await prepare_context_node(state)
        assert result["user_profile"] is None or result["preference_panel"] is not None


class TestPOISearchNode:
    """Test POI search node."""

    @pytest.mark.asyncio
    async def test_search_pois(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="我想去北京",
            user_profile={"destination": "北京", "interests": ["历史"], "food_preferences": ["烤鸭"]},
        )
        result = await poi_search_node(state)
        assert "candidate_pois" in result
        assert isinstance(result["candidate_pois"], list)

    @pytest.mark.asyncio
    async def test_empty_profile(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_profile={},
        )
        result = await poi_search_node(state)
        assert "candidate_pois" in result


class TestWeatherNode:
    """Test weather node."""

    @pytest.mark.asyncio
    async def test_with_dates(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_profile={"destination": "北京", "travel_dates": "2026-05-01 to 2026-05-03"},
        )
        result = await weather_node(state)
        assert "weather_data" in result
        assert isinstance(result["weather_data"], list)

    @pytest.mark.asyncio
    async def test_no_dates(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_profile={"destination": "北京"},
        )
        result = await weather_node(state)
        assert "weather_data" in result

    @pytest.mark.asyncio
    async def test_single_date(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_profile={"destination": "上海", "travel_dates": "2026-06-01"},
        )
        result = await weather_node(state)
        assert "weather_data" in result


class TestBudgetInitNode:
    """Test budget init node."""

    @pytest.mark.asyncio
    async def test_init_panel(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_profile={"destination": "北京", "travel_days": 3, "budget_range": 5000},
        )
        result = await budget_init_node(state)
        assert "budget_panel" in result
        assert result["budget_panel"]["total_budget"] == 5000


class TestContextEnrichmentNode:
    """Test context enrichment node."""

    @pytest.mark.asyncio
    async def test_enrich(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_profile={"destination": "北京", "travel_days": 3, "travel_dates": "2026-05-01"},
        )
        result = await context_enrichment_node(state)
        assert "travel_context" in result


class TestPlannerNode:
    """Test planner node."""

    @pytest.mark.asyncio
    async def test_plan(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="我想去北京3天",
            user_profile={"destination": "北京", "travel_days": 3, "budget_range": 5000, "interests": ["历史"]},
            candidate_pois=[
                ScoredPOI(name="故宫", category="attraction", score=0.9, location=Location(lat=39.9, lng=116.4), ticket_price=60).model_dump(),
            ],
            weather_data=[WeatherDay(date="2026-05-01", condition="晴", temp_high=25, temp_low=15, precipitation_chance=0).model_dump()],
        )
        result = await planner_node(state)
        assert "current_itinerary" in result
        assert "planning_json" in result
        assert result["itinerary_status"] == "draft"

    @pytest.mark.asyncio
    async def test_empty_pois(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_profile={"destination": "北京", "travel_days": 1},
            candidate_pois=[],
            weather_data=[],
        )
        with patch("graph.nodes.ItineraryPlannerAgent.plan", AsyncMock(return_value=[DayPlan(day_number=1, activities=[])])):
            result = await planner_node(state)
            assert "current_itinerary" in result


class TestValidationNode:
    """Test validation node."""

    @pytest.mark.asyncio
    async def test_validate(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_profile={"destination": "北京", "budget_range": 5000},
            current_itinerary=[
                DayPlan(
                    day_number=1,
                    activities=[Activity(poi_name="故宫", category="attraction", ticket_price=60)],
                    total_cost=60,
                ).model_dump(),
            ],
        )
        result = await validation_node(state)
        assert "validation_result" in result


class TestRouteNode:
    """Test route optimization node."""

    @pytest.mark.asyncio
    async def test_optimize(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            current_itinerary=[
                DayPlan(
                    day_number=1,
                    activities=[
                        Activity(poi_name="故宫", category="attraction", location=Location(lat=39.9, lng=116.4)),
                        Activity(poi_name="天坛", category="attraction", location=Location(lat=39.88, lng=116.41)),
                    ],
                ).model_dump(),
            ],
        )
        result = await route_node(state)
        assert "optimized_routes" in result


class TestApplyRoutesNode:
    """Test apply routes node."""

    @pytest.mark.asyncio
    async def test_apply_routes(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            current_itinerary=[
                DayPlan(
                    day_number=1,
                    activities=[Activity(poi_name="故宫", category="attraction")],
                ).model_dump(),
            ],
            optimized_routes={
                1: [Activity(poi_name="天坛", category="attraction").model_dump()],
            },
        )
        result = await apply_routes_node(state)
        assert "current_itinerary" in result
        activities = result["current_itinerary"][0]["activities"]
        assert activities[0]["poi_name"] == "天坛"

    @pytest.mark.asyncio
    async def test_apply_routes_string_key(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            current_itinerary=[
                DayPlan(
                    day_number=1,
                    activities=[Activity(poi_name="故宫", category="attraction")],
                ).model_dump(),
            ],
            optimized_routes={
                "1": [Activity(poi_name="天坛", category="attraction").model_dump()],
            },
        )
        result = await apply_routes_node(state)
        activities = result["current_itinerary"][0]["activities"]
        assert activities[0]["poi_name"] == "天坛"

    @pytest.mark.asyncio
    async def test_no_optimized_routes(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            current_itinerary=[
                DayPlan(day_number=1, activities=[Activity(poi_name="故宫", category="attraction")]).model_dump(),
            ],
        )
        result = await apply_routes_node(state)
        assert result["current_itinerary"][0]["activities"][0]["poi_name"] == "故宫"


class TestBudgetCalcNode:
    """Test budget calculation node."""

    @pytest.mark.asyncio
    async def test_calculate(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_profile={"destination": "北京"},
            current_itinerary=[
                DayPlan(
                    day_number=1,
                    activities=[Activity(poi_name="故宫", category="attraction", ticket_price=60)],
                    total_cost=60,
                ).model_dump(),
            ],
        )
        result = await budget_calc_node(state)
        assert "budget_panel" in result


class TestProposalNode:
    """Test proposal generation node."""

    @pytest.mark.asyncio
    async def test_generate_proposal(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="确认行程",
            user_profile={"destination": "北京", "travel_days": 2},
            current_itinerary=[DayPlan(day_number=1, activities=[]).model_dump()],
            planning_json={"trip_profile": {"destination": "北京"}},
        )
        result = await proposal_node(state)
        assert "proposal_text" in result
        assert result["waiting_for_confirmation"] is True

    @pytest.mark.asyncio
    async def test_generate_without_planning_json(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="test",
            user_profile={"destination": "北京", "travel_days": 1},
            current_itinerary=[],
        )
        result = await proposal_node(state)
        assert "proposal_text" in result


class TestUpdatePrefsNode:
    """Test preference update node."""

    @pytest.mark.asyncio
    async def test_update_with_replan(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="我喜欢吃辣",
            user_profile={"destination": "成都", "food_preferences": ["辣"]},
            preference_changes=[{"field": "food_preferences", "new_value": ["辣", "甜品"]}],
            current_itinerary=[DayPlan(day_number=1, activities=[]).model_dump()],
        )
        result = await update_prefs_node(state)
        assert result["needs_replan"] is True
        assert "重新规划" in result["assistant_response"]

    @pytest.mark.asyncio
    async def test_update_without_replan(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="我喜欢吃辣",
            user_profile={"destination": "成都"},
            preference_changes=[{"field": "food_preferences", "new_value": ["辣"]}],
        )
        result = await update_prefs_node(state)
        assert result["needs_replan"] is False


class TestSaveMemoryNode:
    """Test save memory node."""

    @pytest.mark.asyncio
    async def test_save(self):
        state = ItineraryState(
            session_id="s1",
            user_id="u1",
            user_input="确认",
            user_profile={"destination": "北京", "travel_days": 2},
            current_itinerary=[DayPlan(day_number=1, activities=[]).model_dump()],
            budget_panel={"total": 1000},
        )
        mock_db = MagicMock()
        mock_db.__aenter__ = AsyncMock(return_value=mock_db)
        mock_db.__aexit__ = AsyncMock(return_value=False)

        with patch("graph.nodes.async_session_maker", return_value=mock_db):
            with patch("skills.memory_store.MemoryStoreSkill.save_itinerary", AsyncMock()):
                result = await save_memory_node(state)
                assert result["itinerary_status"] == "confirmed"


class TestSplitDates:
    """Test date splitting helper."""

    def test_range_with_to(self):
        assert _split_dates("2026-05-01 to 2026-05-05") == ("2026-05-01", "2026-05-05")

    def test_range_with_tilde(self):
        assert _split_dates("2026-05-01 ~ 2026-05-05") == ("2026-05-01", "2026-05-05")

    def test_range_with_dash(self):
        assert _split_dates("2026-05-01 - 2026-05-05") == ("2026-05-01", "2026-05-05")

    def test_single_date(self):
        assert _split_dates("2026-05-01") == ("2026-05-01", "2026-05-01")

    def test_empty(self):
        assert _split_dates("") == ("", "")

    def test_chinese_separator(self):
        # Note: separator is " 到 " with spaces, not "到"
        assert _split_dates("5月1日 到 5月5日") == ("5月1日", "5月5日")
