"""Tests for ItineraryPlannerAgent algorithm methods."""

import pytest
from agents.itinerary_planner import ItineraryPlannerAgent
from schemas import UserProfile, ScoredPOI, Location, WeatherDay


class TestFindOriginalPOI:
    """Test fuzzy POI matching."""

    def setup_method(self):
        self.agent = ItineraryPlannerAgent()
        self.pois = [
            ScoredPOI(name="故宫博物院", category="attraction", score=0.9, location=Location(lat=39.9, lng=116.4)),
            ScoredPOI(name="长城", category="attraction", score=0.9, location=Location(lat=40.4, lng=116.0)),
            ScoredPOI(name="全聚德", category="restaurant", score=0.8, location=Location(lat=39.9, lng=116.4)),
        ]

    def test_exact_match(self):
        result = self.agent._find_original_poi("长城", self.pois)
        assert result is not None
        assert result.name == "长城"

    def test_substring_match_long(self):
        result = self.agent._find_original_poi("故宫博物院", self.pois)
        assert result is not None
        assert "故宫" in result.name

    def test_substring_match_query_longer(self):
        result = self.agent._find_original_poi("北京故宫博物院", self.pois)
        assert result is not None

    def test_no_match(self):
        result = self.agent._find_original_poi("不存在的景点XYZ", self.pois)
        assert result is None

    def test_empty_name(self):
        result = self.agent._find_original_poi("", self.pois)
        assert result is None

    def test_fuzzy_match_typo(self):
        result = self.agent._find_original_poi("故官博物院", self.pois)
        assert result is not None
        assert "故宫" in result.name


class TestScorePOIs:
    """Test POI scoring by preferences."""

    def setup_method(self):
        self.agent = ItineraryPlannerAgent()

    def test_interest_match(self):
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.5, tags=["历史"]),
            ScoredPOI(name="科技馆", category="attraction", score=0.5, tags=["科技"]),
        ]
        profile = UserProfile(interests=["历史"])
        result = self.agent._score_pois(pois, profile)
        assert result[0].name == "故宫"
        assert result[0].score > 0.5

    def test_food_match(self):
        pois = [
            ScoredPOI(name="火锅店", category="restaurant", score=0.5, tags=["辣"]),
            ScoredPOI(name="甜品店", category="restaurant", score=0.5, tags=["甜"]),
        ]
        profile = UserProfile(food_preferences=["辣"])
        result = self.agent._score_pois(pois, profile)
        assert result[0].name == "火锅店"

    def test_pace_bonus(self):
        pois = [ScoredPOI(name="A", category="attraction", score=0.5, tags=[])]
        profile_relaxed = UserProfile(pace="relaxed")
        profile_intensive = UserProfile(pace="intensive")
        r1 = self.agent._score_pois([pois[0].model_copy()], profile_relaxed)[0]
        r2 = self.agent._score_pois([pois[0].model_copy()], profile_intensive)[0]
        assert r1.score > r2.score

    def test_score_capped_at_1(self):
        pois = [ScoredPOI(name="A", category="attraction", score=0.9, tags=["历史", "文化", "美食", "自然"])]
        profile = UserProfile(interests=["历史", "文化", "美食", "自然", "拍照"])
        result = self.agent._score_pois(pois, profile)
        assert result[0].score == 1.0

    def test_sort_order(self):
        pois = [
            ScoredPOI(name="低分", category="attraction", score=0.3, tags=[]),
            ScoredPOI(name="高分", category="attraction", score=0.9, tags=[]),
        ]
        profile = UserProfile()
        result = self.agent._score_pois(pois, profile)
        assert result[0].name == "高分"


class TestGroupByArea:
    """Test area grouping."""

    def setup_method(self):
        self.agent = ItineraryPlannerAgent()

    def test_group_by_area(self):
        pois = [
            ScoredPOI(name="A", category="attraction", score=0.5, area="东城区"),
            ScoredPOI(name="B", category="attraction", score=0.5, area="东城区"),
            ScoredPOI(name="C", category="attraction", score=0.5, area="朝阳区"),
        ]
        result = self.agent._group_pois_by_area(pois)
        assert len(result) == 2
        assert len(result["东城区"]) == 2
        assert len(result["朝阳区"]) == 1

    def test_missing_area(self):
        pois = [ScoredPOI(name="A", category="attraction", score=0.5)]
        result = self.agent._group_pois_by_area(pois)
        assert "其他" in result

    def test_sort_by_size(self):
        pois = [
            ScoredPOI(name="A", category="attraction", score=0.5, area="小"),
            ScoredPOI(name="B", category="attraction", score=0.5, area="大"),
            ScoredPOI(name="C", category="attraction", score=0.5, area="大"),
            ScoredPOI(name="D", category="attraction", score=0.5, area="大"),
        ]
        result = self.agent._group_pois_by_area(pois)
        first_group = list(result.values())[0]
        assert len(first_group) == 3


class TestMarkTimeConstraints:
    """Test time constraint marking."""

    def setup_method(self):
        self.agent = ItineraryPlannerAgent()

    def test_evening_only(self):
        pois = [ScoredPOI(name="夜景", category="attraction", score=0.5, tags=["夜景"])]
        result = self.agent._mark_time_constraints(pois)
        assert result[0].time_constraint == "evening_only"

    def test_morning_only(self):
        pois = [ScoredPOI(name="早茶", category="restaurant", score=0.5, tags=["早茶"])]
        result = self.agent._mark_time_constraints(pois)
        assert result[0].time_constraint == "morning_only"

    def test_flexible_default(self):
        pois = [ScoredPOI(name="A", category="attraction", score=0.5, tags=["历史"])]
        result = self.agent._mark_time_constraints(pois)
        assert result[0].time_constraint == "flexible"


class TestAssignDays:
    """Test day assignment logic."""

    def setup_method(self):
        self.agent = ItineraryPlannerAgent()

    def test_assign_to_multiple_days(self):
        pois = [
            ScoredPOI(name="A", category="attraction", score=0.5, area="东城"),
            ScoredPOI(name="B", category="attraction", score=0.5, area="东城"),
            ScoredPOI(name="C", category="attraction", score=0.5, area="朝阳"),
            ScoredPOI(name="D", category="attraction", score=0.5, area="朝阳"),
        ]
        groups = self.agent._group_pois_by_area(pois)
        result = self.agent._assign_days(pois, groups, 2)
        assert len(result) == 2
        assert len(result[0]) > 0
        assert len(result[1]) > 0

    def test_no_duplicate_assignment(self):
        pois = [
            ScoredPOI(name="A", category="attraction", score=0.5, area="东城"),
            ScoredPOI(name="B", category="attraction", score=0.5, area="东城"),
        ]
        groups = self.agent._group_pois_by_area(pois)
        result = self.agent._assign_days(pois, groups, 2)
        all_names = [p.name for day in result for p in day]
        assert len(all_names) == len(set(all_names))

    def test_all_pois_assigned(self):
        pois = [
            ScoredPOI(name="A", category="attraction", score=0.5, area="东城"),
            ScoredPOI(name="B", category="attraction", score=0.5, area="东城"),
            ScoredPOI(name="C", category="attraction", score=0.5, area="朝阳"),
        ]
        groups = self.agent._group_pois_by_area(pois)
        result = self.agent._assign_days(pois, groups, 2)
        all_names = [p.name for day in result for p in day]
        assert set(all_names) == {"A", "B", "C"}


class TestOptimizeRoutes:
    """Test route optimization."""

    def setup_method(self):
        self.agent = ItineraryPlannerAgent()

    def test_short_day_no_change(self):
        day = [ScoredPOI(name="A", category="attraction", score=0.5)]
        result = self.agent._optimize_daily_routes([day])
        assert result[0][0].name == "A"

    def test_two_pois_no_change(self):
        day = [
            ScoredPOI(name="A", category="attraction", score=0.5, location=Location(lat=0, lng=0)),
            ScoredPOI(name="B", category="attraction", score=0.5, location=Location(lat=1, lng=1)),
        ]
        result = self.agent._optimize_daily_routes([day])
        assert len(result[0]) == 2

    def test_three_pois_optimized(self):
        day = [
            ScoredPOI(name="A", category="attraction", score=0.5, location=Location(lat=0, lng=0)),
            ScoredPOI(name="B", category="attraction", score=0.5, location=Location(lat=0.01, lng=0)),
            ScoredPOI(name="C", category="attraction", score=0.5, location=Location(lat=10, lng=10)),
        ]
        result = self.agent._optimize_daily_routes([day])
        assert len(result[0]) == 3


class TestNearestNeighbor:
    """Test greedy nearest neighbor."""

    def setup_method(self):
        self.agent = ItineraryPlannerAgent()

    def test_empty(self):
        assert self.agent._nearest_neighbor([]) == []

    def test_single(self):
        pois = [ScoredPOI(name="A", category="attraction", score=0.5, location=Location(lat=0, lng=0))]
        result = self.agent._nearest_neighbor(pois)
        assert len(result) == 1

    def test_orders_by_distance(self):
        pois = [
            ScoredPOI(name="A", category="attraction", score=0.5, location=Location(lat=0, lng=0)),
            ScoredPOI(name="B", category="attraction", score=0.5, location=Location(lat=0.001, lng=0)),
            ScoredPOI(name="C", category="attraction", score=0.5, location=Location(lat=10, lng=10)),
        ]
        result = self.agent._nearest_neighbor(pois)
        assert result[0].name == "A"
        assert result[1].name == "B"  # closest to A


class TestDistance:
    """Test distance helper."""

    def setup_method(self):
        self.agent = ItineraryPlannerAgent()

    def test_distance_with_locations(self):
        a = Location(lat=39.9, lng=116.4)
        b = Location(lat=39.91, lng=116.41)
        result = self.agent._distance(a, b)
        assert result > 0
        assert result < 10000  # meters

    def test_distance_none(self):
        assert self.agent._distance(None, Location(lat=0, lng=0)) == float("inf")


class TestBuildSchedule:
    """Test schedule building with meals."""

    def setup_method(self):
        self.agent = ItineraryPlannerAgent()

    def test_build_single_day(self):
        day_pois = [
            [ScoredPOI(name="故宫", category="attraction", score=0.9, location=Location(lat=39.9, lng=116.4), ticket_price=60)],
        ]
        weather = [WeatherDay(date="2026-05-01", condition="晴", temp_high=25, temp_low=15, precipitation_chance=0)]
        profile = UserProfile(destination="北京", food_preferences=["烤鸭"])
        result = self.agent._build_schedule(day_pois, weather, profile)
        assert len(result) == 1
        assert result[0].day_number == 1
        assert any(a.poi_name == "故宫" for a in result[0].activities)

    def test_total_cost_calculation(self):
        day_pois = [
            [
                ScoredPOI(name="A", category="attraction", score=0.5, ticket_price=100),
                ScoredPOI(name="B", category="attraction", score=0.5, ticket_price=50),
            ],
        ]
        weather = []
        profile = UserProfile()
        result = self.agent._build_schedule(day_pois, weather, profile)
        # total = 100 + 50 + lunch meal (80)
        assert result[0].total_cost == 230

    def test_activity_times(self):
        day_pois = [
            [ScoredPOI(name="故宫", category="attraction", score=0.9, location=Location(lat=39.9, lng=116.4))],
        ]
        weather = []
        profile = UserProfile()
        result = self.agent._build_schedule(day_pois, weather, profile)
        activity = result[0].activities[0]
        assert activity.start_time is not None
        assert activity.end_time is not None
        assert activity.duration_min == 120  # attraction default

    def test_restaurant_duration(self):
        day_pois = [
            [ScoredPOI(name="餐厅", category="restaurant", score=0.8)],
        ]
        weather = []
        profile = UserProfile()
        result = self.agent._build_schedule(day_pois, weather, profile)
        activity = result[0].activities[0]
        assert activity.duration_min == 60  # restaurant default


class TestCreateMealActivity:
    """Test meal activity creation."""

    def setup_method(self):
        self.agent = ItineraryPlannerAgent()

    def test_lunch(self):
        profile = UserProfile(food_preferences=["辣"])
        meal = self.agent._create_meal_activity(0, "lunch", profile)
        assert meal.category == "restaurant"
        assert meal.duration_min == 90
        assert meal.meal_cost == 80
        assert "辣" in meal.poi_name

    def test_dinner(self):
        profile = UserProfile()
        meal = self.agent._create_meal_activity(0, "dinner", profile)
        assert meal.category == "restaurant"
        assert "Dinner" in meal.poi_name


class TestMinToTime:
    """Test time formatting."""

    def test_morning(self):
        assert ItineraryPlannerAgent._min_to_time(540) == "09:00"

    def test_noon(self):
        assert ItineraryPlannerAgent._min_to_time(720) == "12:00"

    def test_afternoon(self):
        assert ItineraryPlannerAgent._min_to_time(900) == "15:00"

    def test_evening(self):
        assert ItineraryPlannerAgent._min_to_time(1080) == "18:00"


class TestPlanWithAlgorithm:
    """Test the full fallback algorithm."""

    @pytest.mark.asyncio
    async def test_full_algorithm(self):
        agent = ItineraryPlannerAgent()
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9, location=Location(lat=39.9, lng=116.4), area="东城", ticket_price=60),
            ScoredPOI(name="长城", category="attraction", score=0.9, location=Location(lat=40.4, lng=116.0), area="延庆", ticket_price=40),
        ]
        weather = [WeatherDay(date="2026-05-01", condition="晴", temp_high=25, temp_low=15, precipitation_chance=0)]
        profile = UserProfile(destination="北京", travel_days=1, interests=["历史"])
        result = await agent._plan_with_algorithm(pois, weather, profile)
        assert len(result) == 1
        assert len(result[0].activities) > 0
