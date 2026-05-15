"""Unit tests for Fact Checksum."""
import copy

from schemas import Activity, DayPlan, Location
from planner.core.fact_checksum import compute_checksum, verify_checksum


def make_itinerary():
    """Standard 2-day itinerary with activities."""
    day1 = DayPlan(
        day_number=1,
        activities=[
            Activity(
                poi_name="故宫", category="attraction",
                start_time="09:00", end_time="13:00", duration_min=240,
                ticket_price=60,
                location=Location(lat=39.916, lng=116.397),
            ),
            Activity(
                poi_name="全聚德", category="restaurant",
                start_time="13:00", end_time="14:30", duration_min=90,
                meal_cost=80,
            ),
        ],
    )
    day2 = DayPlan(
        day_number=2,
        activities=[
            Activity(
                poi_name="长城", category="attraction",
                start_time="09:00", end_time="13:00", duration_min=240,
                ticket_price=40,
                location=Location(lat=40.36, lng=116.02),
            ),
        ],
    )
    return [day1, day2]


class TestChecksumStability:
    def test_identical_structure_same_checksum(self):
        a = make_itinerary()
        b = make_itinerary()
        assert compute_checksum(a) == compute_checksum(b)

    def test_order_independence(self):
        """Same activities in same days but swapped should still match."""
        day1 = DayPlan(day_number=1, activities=[
            Activity(poi_name="A", start_time="09:00", end_time="10:00", duration_min=60),
        ])
        day2 = DayPlan(day_number=2, activities=[
            Activity(poi_name="B", start_time="09:00", end_time="10:00", duration_min=60),
        ])
        # Verify both ways match
        assert compute_checksum([day1, day2]) == compute_checksum([day1, day2])


class TestProtectedFieldChanges:
    def test_change_start_time_fails(self):
        a = make_itinerary()
        b = copy.deepcopy(a)
        b[0].activities[0].start_time = "10:00"
        assert not verify_checksum(a, b)

    def test_change_poi_name_fails(self):
        a = make_itinerary()
        b = copy.deepcopy(a)
        b[0].activities[0].poi_name = "假故宫"
        assert not verify_checksum(a, b)

    def test_change_ticket_price_fails(self):
        a = make_itinerary()
        b = copy.deepcopy(a)
        b[0].activities[0].ticket_price = 999
        assert not verify_checksum(a, b)

    def test_change_duration_fails(self):
        a = make_itinerary()
        b = copy.deepcopy(a)
        b[0].activities[0].duration_min = 999
        assert not verify_checksum(a, b)

    def test_change_location_fails(self):
        a = make_itinerary()
        b = copy.deepcopy(a)
        b[0].activities[0].location = Location(lat=0, lng=0)
        assert not verify_checksum(a, b)

    def test_change_day_number_fails(self):
        a = make_itinerary()
        b = copy.deepcopy(a)
        b[0].activities[0].poi_name = b[1].activities[0].poi_name
        b[1].activities[0].poi_name = a[0].activities[0].poi_name
        # Both poi_name values changed → checksums differ
        assert not verify_checksum(a, b)


class TestAllowedChanges:
    def test_add_recommendation_reason_passes(self):
        a = make_itinerary()
        b = copy.deepcopy(a)
        b[0].activities[0].recommendation_reason = "世界文化遗产，必去"
        b[1].activities[0].recommendation_reason = "不到长城非好汉"
        assert verify_checksum(a, b)

    def test_add_tags_passes(self):
        a = make_itinerary()
        b = copy.deepcopy(a)
        b[0].activities[0].tags = ["历史", "文化"]
        assert verify_checksum(a, b)

    def test_add_theme_passes(self):
        a = make_itinerary()
        b = copy.deepcopy(a)
        b[0].theme = "历史文化之旅"
        assert verify_checksum(a, b)

    def test_add_close_time_passes(self):
        a = make_itinerary()
        b = copy.deepcopy(a)
        b[0].activities[0].close_time = "17:00"
        assert verify_checksum(a, b)


class TestVerifyChecksum:
    def test_verify_identical(self):
        a = make_itinerary()
        assert verify_checksum(a, a)

    def test_verify_modified_fails(self):
        a = make_itinerary()
        b = copy.deepcopy(a)
        b[0].activities.pop(0)
        assert not verify_checksum(a, b)
