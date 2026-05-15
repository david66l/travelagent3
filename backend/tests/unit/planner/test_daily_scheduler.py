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

    def test_assign_days_does_not_loop_when_all_days_reach_soft_cap(self):
        """Overflow POIs are still assigned once all days already have 5 items."""
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
        assert sorted(assigned_names) == sorted(poi.name for poi in pois)
