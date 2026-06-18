"""Verify Nanjing 2-day itinerary improvements."""

import asyncio
import sys

sys.path.insert(0, "src")

from schemas import UserProfile, WeatherDay
from planner.core.heuristic_strategy import build_strategy
from planner.core.daily_scheduler import build_schedule
from planner.core.writer import enrich
from skills.city_data import CITY_DEFAULTS


async def main():
    profile = UserProfile(
        destination="南京",
        travel_days=2,
        travel_dates="2026-06-01",
        travelers_type="情侣",
        budget_range=2000,
        interests=["历史", "文化", "美食"],
        food_preferences=["南京菜"],
        pace="适中",
    )

    # Use a curated subset of Nanjing POIs so that key landmarks appear
    # in the 2-day schedule while still exercising the scheduler.
    nanjing_pois = CITY_DEFAULTS.get("南京", [])
    selected_names = {
        "中山陵",
        "夫子庙",
        "秦淮河",
        "老门东",
        "南京大牌档",
        "盐水鸭",
        "鸭血粉丝汤",
        "小笼包",
    }
    pois = [p for p in nanjing_pois if p.name in selected_names]
    # Ensure all selected POIs exist in city data
    missing = selected_names - {p.name for p in pois}
    if missing:
        print(f"Warning: missing POIs from city data: {missing}")

    weather = [
        WeatherDay(date="2026-06-01", condition="晴", temp_high=28, temp_low=18, precipitation_chance=0),
        WeatherDay(date="2026-06-02", condition="多云", temp_high=26, temp_low=17, precipitation_chance=10),
    ]

    strategy = build_strategy(pois, profile)
    schedule = build_schedule(strategy, pois, weather, profile)
    enriched, proposal = await enrich(schedule, profile)

    print(proposal)
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    generic_count = proposal.count("推荐游览") + proposal.count("口碑推荐")
    print(f"'推荐游览' + '口碑推荐' occurrences: {generic_count}")
    assert generic_count == 0, f"Found generic phrases {generic_count} times"

    # Check consecutive restaurants
    max_consecutive = 0
    for day in enriched:
        consecutive = 0
        for act in day.activities:
            if act.category == "restaurant":
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0
    print(f"Max consecutive restaurant items: {max_consecutive}")
    assert max_consecutive <= 2, f"Too many consecutive restaurants: {max_consecutive}"

    # Check 中山陵 description uses template
    zhongshan_found = False
    for day in enriched:
        for act in day.activities:
            if act.poi_name == "中山陵":
                zhongshan_found = True
                print(f"中山陵 description: {act.recommendation_reason}")
                expected = "中国近代建筑史上第一陵，392级台阶象征三民主义与五权宪法"
                assert expected in act.recommendation_reason, f"Expected template reason for 中山陵, got: {act.recommendation_reason}"
    assert zhongshan_found, "中山陵 not found in itinerary"

    # Check each activity has a non-empty, non-generic reason
    for day in enriched:
        for act in day.activities:
            assert act.recommendation_reason, f"Empty reason for {act.poi_name}"
            assert "推荐游览" not in act.recommendation_reason, f"Generic '推荐游览' reason for {act.poi_name}"
            assert "口碑推荐" not in act.recommendation_reason, f"Generic '口碑推荐' reason for {act.poi_name}"

    # Check budget note
    for day in enriched:
        assert "费用明细" in day.budget_note, f"Missing budget note on day {day.day_number}"
        assert "合计" in day.budget_note, f"Missing total in budget note on day {day.day_number}"

    print("\nAll validations passed!")


if __name__ == "__main__":
    asyncio.run(main())
