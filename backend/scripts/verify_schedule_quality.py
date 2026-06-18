"""Quick quality check for Nanjing/Chengdu schedules after scheduler refactor."""

from schemas import UserProfile, WeatherDay
from planner.core.heuristic_strategy import build_strategy
from planner.core.daily_scheduler import build_schedule, _sanity_check
from skills.city_data import CITY_DEFAULTS


def _time_to_minutes(time_str: str) -> int:
    h, m = map(int, time_str.split(":"))
    return h * 60 + m


def _check_day(day, budget_limit: float):
    issues = []
    if not day.activities:
        issues.append(f"Day {day.day_number}: no activities")
        return issues

    # Time overlap / ordering
    prev_end = None
    for a in day.activities:
        if a.start_time and a.end_time:
            start = _time_to_minutes(a.start_time)
            end = _time_to_minutes(a.end_time)
            if start >= end:
                issues.append(f"Day {day.day_number}: {a.poi_name} start >= end")
            if prev_end is not None and start < prev_end:
                issues.append(
                    f"Day {day.day_number}: overlap {a.poi_name} starts before previous ends"
                )
            prev_end = end

    # Budget
    tickets_meals = sum((a.ticket_price or 0) + (a.meal_cost or 0) for a in day.activities)
    if tickets_meals > budget_limit:
        issues.append(f"Day {day.day_number}: tickets/meals {tickets_meals} > budget {budget_limit}")

    return issues


def verify_city(city: str, days: int, budget: float):
    print(f"\n=== {city} {days} days budget={budget} ===")
    pois = list(CITY_DEFAULTS.get(city, []))
    if not pois:
        print(f"No fallback POIs for {city}")
        return

    profile = UserProfile(
        destination=city,
        travel_days=days,
        budget_range=budget,
        interests=["历史", "文化", "美食"],
        food_preferences=["辣"],
        pace="moderate",
    )
    weather = [
        WeatherDay(
            date=f"2026-05-{i+1:02d}",
            condition="晴",
            temp_high=25,
            temp_low=15,
            precipitation_chance=0,
        )
        for i in range(days)
    ]

    strategy = build_strategy(pois, profile)
    schedule = build_schedule(strategy, pois, weather, profile)

    print(f"Scheduled {len(schedule)} days")
    for day in schedule:
        print(f"\nDay {day.day_number} ({day.date})")
        print(f"  total_cost={day.total_cost:.0f}")
        for a in day.activities:
            print(
                f"  {a.start_time}-{a.end_time} {a.poi_name} "
                f"({a.category}, {a.duration_min}min, ticket={a.ticket_price}, meal={a.meal_cost})"
            )

    sanity = _sanity_check(schedule, profile)
    if sanity:
        print("\nSanity warnings:")
        for w in sanity:
            print(f"  - {w}")

    issues = []
    for day in schedule:
        issues.extend(_check_day(day, budget))
    if issues:
        print("\nQuality issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nNo quality issues detected.")


if __name__ == "__main__":
    verify_city("南京", 3, 1500)
    verify_city("成都", 3, 1500)
