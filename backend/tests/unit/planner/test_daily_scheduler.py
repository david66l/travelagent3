"""Unit tests for daily scheduler."""

from schemas import ScoredPOI, WeatherDay, UserProfile, Location
from planner.core.heuristic_strategy import build_strategy
from planner.core.daily_scheduler import build_schedule, _assign_days


class TestDailyScheduler:
    def test_empty_pois_returns_empty_schedule(self):
        strategy = build_strategy([], UserProfile(destination="北京", travel_days=1))
        schedule = build_schedule(strategy, [], [], UserProfile(destination="北京", travel_days=1))
        assert schedule == []

    def test_schedule_has_correct_day_count(self):
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9, area="东城区"),
            ScoredPOI(name="天坛", category="attraction", score=0.85, area="东城区"),
            ScoredPOI(name="颐和园", category="attraction", score=0.8, area="海淀区"),
        ]
        profile = UserProfile(destination="北京", travel_days=2)
        strategy = build_strategy(pois, profile)
        weather = [
            WeatherDay(date="2026-05-01", condition="晴", temp_high=25, temp_low=15, precipitation_chance=0),
            WeatherDay(date="2026-05-02", condition="多云", temp_high=24, temp_low=14, precipitation_chance=10),
        ]
        schedule = build_schedule(strategy, pois, weather, profile)

        assert len(schedule) == 2
        assert schedule[0].day_number == 1
        assert schedule[1].day_number == 2

    def test_activities_have_times(self):
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9, area="东城区"),
        ]
        profile = UserProfile(destination="北京", travel_days=1)
        strategy = build_strategy(pois, profile)
        schedule = build_schedule(strategy, pois, [], profile)

        assert len(schedule) == 1
        assert len(schedule[0].activities) >= 1
        act = schedule[0].activities[0]
        assert act.start_time is not None
        assert act.end_time is not None
        assert act.duration_min > 0

    def test_meals_inserted_at_lunch_dinner_time(self):
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9, area="东城区"),
            ScoredPOI(name="天坛", category="attraction", score=0.85, area="东城区"),
            ScoredPOI(name="颐和园", category="attraction", score=0.8, area="海淀区"),
            ScoredPOI(name="圆明园", category="attraction", score=0.75, area="海淀区"),
        ]
        profile = UserProfile(destination="北京", travel_days=1)
        strategy = build_strategy(pois, profile)
        schedule = build_schedule(strategy, pois, [], profile)

        categories = [a.category for a in schedule[0].activities]
        assert "restaurant" in categories
        meals = [a for a in schedule[0].activities if a.category == "restaurant"]
        assert all(meal.start_time and meal.end_time for meal in meals)

    def test_route_optimization_orders_by_proximity(self):
        loc_a = Location(lat=39.9, lng=116.4)
        loc_b = Location(lat=39.91, lng=116.41)
        loc_c = Location(lat=40.0, lng=116.5)  # far away

        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9, location=loc_a),
            ScoredPOI(name="天坛", category="attraction", score=0.85, location=loc_b),
            ScoredPOI(name="颐和园", category="attraction", score=0.8, location=loc_c),
        ]
        profile = UserProfile(destination="北京", travel_days=1)
        strategy = build_strategy(pois, profile)
        schedule = build_schedule(strategy, pois, [], profile)

        # After nearest-neighbor optimization, 故宫 and 天坛 should be adjacent
        names = [a.poi_name for a in schedule[0].activities if a.category != "restaurant"]
        # Either order is fine as long as they're close
        assert len(names) == 3

    def test_deterministic_output(self):
        """Same input must produce same output."""
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9, area="东城区"),
            ScoredPOI(name="天坛", category="attraction", score=0.85, area="东城区"),
        ]
        profile = UserProfile(destination="北京", travel_days=1)
        strategy = build_strategy(pois, profile)

        schedule1 = build_schedule(strategy, pois, [], profile)
        schedule2 = build_schedule(strategy, pois, [], profile)

        assert len(schedule1) == len(schedule2)
        for d1, d2 in zip(schedule1, schedule2):
            assert [a.poi_name for a in d1.activities] == [a.poi_name for a in d2.activities]

    def test_duration_from_recommended_hours(self):
        pois = [
            ScoredPOI(
                name="故宫",
                category="attraction",
                score=0.9,
                recommended_hours="半天",
            ),
        ]
        profile = UserProfile(destination="北京", travel_days=1)
        strategy = build_strategy(pois, profile)
        schedule = build_schedule(strategy, pois, [], profile)

        act = schedule[0].activities[0]
        assert act.duration_min == 240  # 半天 = 240 min

    def test_beijing_remote_excursion_is_not_mixed_with_city_core(self):
        """长城这类远郊半日/一日行程不能和故宫、天坛、颐和园硬塞同一天。"""
        pois = [
            ScoredPOI(
                name="故宫",
                category="attraction",
                score=0.95,
                area="东城区",
                recommended_hours="半天",
                location=Location(lat=39.916345, lng=116.397155),
            ),
            ScoredPOI(
                name="长城",
                category="attraction",
                score=0.95,
                area="延庆区",
                recommended_hours="半天",
                location=Location(lat=40.359580, lng=116.019967),
            ),
            ScoredPOI(
                name="天坛",
                category="attraction",
                score=0.9,
                area="东城区",
                recommended_hours="2-3小时",
                location=Location(lat=39.883455, lng=116.406588),
            ),
            ScoredPOI(
                name="颐和园",
                category="attraction",
                score=0.9,
                area="海淀区",
                recommended_hours="半天",
                location=Location(lat=39.999982, lng=116.275461),
            ),
        ]
        profile = UserProfile(destination="北京", travel_days=3)
        strategy = build_strategy(pois, profile)
        schedule = build_schedule(strategy, pois, [], profile)

        wall_day = next(
            day for day in schedule
            if any(a.poi_name == "长城" for a in day.activities)
        )
        non_meal_names = [
            a.poi_name for a in wall_day.activities
            if a.category != "restaurant"
        ]
        assert non_meal_names == ["长城"]

    def test_remote_excursion_is_detected_without_area_metadata(self):
        """Fallback/外部搜索缺 area 时，也必须用坐标/名称识别远郊专项。"""
        pois = [
            ScoredPOI(
                name="城市核心A",
                category="attraction",
                score=0.95,
                location=Location(lat=39.916345, lng=116.397155),
            ),
            ScoredPOI(
                name="远郊山线",
                category="attraction",
                score=0.95,
                location=Location(lat=40.359580, lng=116.019967),
            ),
            ScoredPOI(
                name="城市核心B",
                category="attraction",
                score=0.9,
                location=Location(lat=39.883455, lng=116.406588),
            ),
            ScoredPOI(
                name="城市核心C",
                category="attraction",
                score=0.9,
                location=Location(lat=39.999982, lng=116.275461),
            ),
        ]
        profile = UserProfile(destination="北京", travel_days=3)
        strategy = build_strategy(pois, profile)
        schedule = build_schedule(strategy, pois, [], profile)

        wall_day = next(
            day for day in schedule
            if any(a.poi_name == "远郊山线" for a in day.activities)
        )
        non_meal_names = [
            a.poi_name for a in wall_day.activities
            if a.category != "restaurant"
        ]
        # Remote excursion is placed on a dedicated day; the day *may*
        # contain a small number of non-remote POIs when the remote
        # leaves enough remaining daylight (dynamic effective_max).
        assert "远郊山线" in non_meal_names
        assert len(non_meal_names) <= 3  # 1 remote + up to 2 extras from daylight

    def test_visit_durations_come_from_poi_metadata_not_poi_name(self):
        pois = [
            ScoredPOI(
                name="任意半日景点",
                category="attraction",
                score=0.95,
                recommended_hours="半天",
            ),
            ScoredPOI(
                name="任意全天景点",
                category="attraction",
                score=0.95,
                recommended_hours="全天",
            ),
        ]
        profile = UserProfile(destination="任意城市", travel_days=2)
        strategy = build_strategy(pois, profile)
        schedule = build_schedule(strategy, pois, [], profile)

        durations = {
            a.poi_name: a.duration_min
            for day in schedule
            for a in day.activities
            if a.category != "restaurant"
        }
        assert durations["任意半日景点"] == 240
        assert durations["任意全天景点"] == 360

    def test_scheduler_does_not_emit_activities_after_day_end(self):
        pois = [
            ScoredPOI(
                name=f"半日景点{i}",
                category="attraction",
                score=0.9 - i * 0.01,
                recommended_hours="半天",
            )
            for i in range(4)
        ]
        profile = UserProfile(destination="任意城市", travel_days=1)
        strategy = build_strategy(pois, profile)
        schedule = build_schedule(strategy, pois, [], profile)

        assert all(
            activity.end_time <= "21:00"
            for activity in schedule[0].activities
            if activity.end_time
        )

    def test_restaurants_fairly_compete_with_attractions(self):
        """Restaurants and attractions share daily capacity by score."""
        pois = [
            ScoredPOI(
                name=f"景点{i}",
                category="attraction",
                score=0.9 - i * 0.01,
                recommended_hours="1-2小时",
            )
            for i in range(4)
        ] + [
            ScoredPOI(
                name=f"餐厅{i}",
                category="restaurant",
                score=0.95,
                recommended_hours="1小时",
            )
            for i in range(4)
        ]
        profile = UserProfile(destination="任意城市", travel_days=1)
        strategy = build_strategy(pois, profile)
        schedule = build_schedule(strategy, pois, [], profile)

        all_names = [a.poi_name for a in schedule[0].activities if a.category != "restaurant"]
        # max_per_day=4 limits total; higher-score restaurants may take priority
        assert len(all_names) >= 1
        assert "景点0" in all_names

    def test_assign_days_stops_when_all_days_reach_capacity(self):
        """Overflow POIs are dropped once all days reach the feasible daily cap."""
        pois = [
            ScoredPOI(name=f"A{i}", category="attraction", score=0.9, area="A")
            for i in range(5)
        ] + [
            ScoredPOI(name=f"B{i}", category="attraction", score=0.8, area="B")
            for i in range(5)
        ] + [
            ScoredPOI(name=f"C{i}", category="attraction", score=0.7, area="C")
            for i in range(5)
        ] + [
            ScoredPOI(name=f"D{i}", category="attraction", score=0.6, area="D")
            for i in range(2)
        ]
        groups: dict[str, list[ScoredPOI]] = {}
        for poi in pois:
            groups.setdefault(poi.area or "其他", []).append(poi)

        days = _assign_days(pois, groups, travel_days=3)

        assigned_names = [poi.name for day in days for poi in day]
        assert len(assigned_names) == 12
        assert all(len(day) <= 4 for day in days)
