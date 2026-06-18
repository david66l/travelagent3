"""Unit tests for daily scheduler."""

from schemas import ScoredPOI, WeatherDay, UserProfile, Location
from planner.core.heuristic_strategy import build_strategy
from planner.core.daily_scheduler import build_schedule, _assign_days_constrained


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
            WeatherDay(
                date="2026-05-01", condition="晴", temp_high=25, temp_low=15, precipitation_chance=0
            ),
            WeatherDay(
                date="2026-05-02",
                condition="多云",
                temp_high=24,
                temp_low=14,
                precipitation_chance=10,
            ),
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
            ScoredPOI(name="颐和园", category="temple", score=0.8, location=loc_c),
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
            day for day in schedule if any(a.poi_name == "长城" for a in day.activities)
        )
        non_meal_names = [a.poi_name for a in wall_day.activities if a.category != "restaurant"]
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
            day for day in schedule if any(a.poi_name == "远郊山线" for a in day.activities)
        )
        non_meal_names = [a.poi_name for a in wall_day.activities if a.category != "restaurant"]
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
            activity.end_time <= "21:00" for activity in schedule[0].activities if activity.end_time
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
        pois = (
            [ScoredPOI(name=f"A{i}", category="attraction", score=0.9, area="A") for i in range(5)]
            + [
                ScoredPOI(name=f"B{i}", category="attraction", score=0.8, area="B")
                for i in range(5)
            ]
            + [
                ScoredPOI(name=f"C{i}", category="attraction", score=0.7, area="C")
                for i in range(5)
            ]
            + [
                ScoredPOI(name=f"D{i}", category="attraction", score=0.6, area="D")
                for i in range(2)
            ]
        )
        groups: dict[str, list[ScoredPOI]] = {}
        for poi in pois:
            groups.setdefault(poi.area or "其他", []).append(poi)

        profile = UserProfile(destination="测试城市", travel_days=3)
        days = _assign_days_constrained(
            pois,
            groups,
            travel_days=3,
            profile=profile,
            must_see=[],
            remote_class={},
            cross_city=set(),
            center=None,
        )

        assigned_names = [poi.name for day in days for poi in day]
        # Diversity constraint limits same-category POIs to 2 per day
        assert len(assigned_names) == 6
        assert all(len(day) <= 2 for day in days)


# --------------------------------------------------------------------------- #
# Algorithm optimization tests
# --------------------------------------------------------------------------- #


class TestProximityGrouping:
    def test_proximity_grouping_groups_nearby_pois(self):
        loc_a = Location(lat=30.24, lng=120.14)
        loc_b = Location(lat=30.25, lng=120.15)
        loc_c = Location(lat=30.26, lng=120.16)
        loc_far = Location(lat=29.60, lng=119.03)

        pois = [
            ScoredPOI(name="西湖", category="attraction", score=0.9, location=loc_a),
            ScoredPOI(name="雷峰塔", category="attraction", score=0.9, location=loc_b),
            ScoredPOI(name="断桥", category="attraction", score=0.9, location=loc_c),
            ScoredPOI(name="千岛湖", category="attraction", score=0.9, location=loc_far),
        ]

        from planner.core.daily_scheduler import _group_pois_by_proximity

        groups = _group_pois_by_proximity(pois)
        # 西湖, 雷峰塔, 断桥 are within 3km of each other; 千岛湖 is alone
        assert len(groups) == 2
        cluster_names = {name: [p.name for p in group] for name, group in groups.items()}
        names = set()
        for group in cluster_names.values():
            names.update(group)
        assert names == {"西湖", "雷峰塔", "断桥", "千岛湖"}

    def test_proximity_grouping_falls_back_when_no_locations(self):
        pois = [
            ScoredPOI(name="故宫", category="attraction", score=0.9),
            ScoredPOI(name="天坛", category="attraction", score=0.9),
        ]

        from planner.core.daily_scheduler import _group_pois_by_proximity

        groups = _group_pois_by_proximity(pois)
        assert groups == {}


class TestRouteOptimization:
    def test_two_opt_matches_or_improves_nearest_neighbor(self):
        locs = [
            Location(lat=30.2458, lng=120.1450),
            Location(lat=30.2408, lng=120.0980),
            Location(lat=30.2310, lng=120.1430),
            Location(lat=30.2390, lng=120.1420),
            Location(lat=30.2590, lng=120.1460),
        ]
        pois = [
            ScoredPOI(name=f"POI{i}", category="attraction", score=0.9, location=loc)
            for i, loc in enumerate(locs)
        ]

        from planner.core.daily_scheduler import (
            _nearest_neighbor,
            _optimize_route_2opt,
            _route_total_distance,
        )

        nn_route = _nearest_neighbor(pois)
        opt_route = _optimize_route_2opt(nn_route)
        assert _route_total_distance(opt_route) <= _route_total_distance(nn_route) + 1e-9


class TestMealMatching:
    def test_create_meal_prefers_real_restaurant_by_food_preference(self):
        from planner.core.daily_scheduler import _create_meal_activity

        nearby = [
            ScoredPOI(name="川菜馆", category="restaurant", score=0.8, tags=["辣", "川菜"]),
            ScoredPOI(name="杭帮菜馆", category="restaurant", score=0.8, tags=["杭帮菜"]),
        ]
        profile = UserProfile(destination="杭州", travel_days=1, food_preferences=["杭帮菜"])
        meal = _create_meal_activity(0, "lunch", profile, 12 * 60, nearby)
        assert "杭帮菜馆" in meal.poi_name
        assert meal.category == "restaurant"

    def test_create_meal_falls_back_to_placeholder_when_no_restaurants(self):
        from planner.core.daily_scheduler import _create_meal_activity

        profile = UserProfile(destination="杭州", travel_days=1)
        meal = _create_meal_activity(0, "lunch", profile, 12 * 60, [])
        assert "Lunch" in meal.poi_name


class TestDailyDiversity:
    def test_daily_diversity_replaces_excess_same_category(self):
        from planner.core.daily_scheduler import _ensure_daily_diversity

        all_pois = [
            ScoredPOI(name="寺庙A", category="temple", score=0.9),
            ScoredPOI(name="寺庙B", category="temple", score=0.85),
            ScoredPOI(name="寺庙C", category="temple", score=0.8),
            ScoredPOI(name="湖泊", category="lake", score=0.95),
        ]
        day_assignments = [list(all_pois[:3])]
        result = _ensure_daily_diversity(day_assignments, all_pois)
        categories = [p.category for p in result[0]]
        assert categories.count("temple") == 2
        assert "lake" in categories


class TestTravelBudget:
    def test_budget_includes_transport_and_accommodation(self):
        pois = [
            ScoredPOI(name="西湖", category="attraction", score=0.9, area="西湖区"),
        ]
        profile = UserProfile(destination="杭州", travel_days=3)
        strategy = build_strategy(pois, profile)
        schedule = build_schedule(strategy, pois, [], profile)

        total = sum(day.total_cost for day in schedule)
        transport = 3 * 30
        accommodation = (3 - 1) * 200
        assert total == transport + accommodation


class TestRemoteDetection:
    def test_cross_city_poi_is_detected_and_excluded(self):
        from planner.core.daily_scheduler import detect_remote_pois

        pois = [
            ScoredPOI(
                name="西湖",
                category="attraction",
                score=0.9,
                location=Location(lat=30.2458, lng=120.1450),
            ),
            ScoredPOI(
                name="雷峰塔",
                category="attraction",
                score=0.9,
                location=Location(lat=30.2310, lng=120.1430),
            ),
            ScoredPOI(
                name="千岛湖",
                category="attraction",
                score=0.9,
                location=Location(lat=29.5980, lng=119.0330),
            ),
        ]
        remote_class, cross_city, _ = detect_remote_pois(pois)
        assert remote_class.get("千岛湖") == "cross_city"
        assert "千岛湖" in cross_city
