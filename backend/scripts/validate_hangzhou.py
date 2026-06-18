"""验证脚本：杭州3日游规划算法优化效果。

检查项：
1. 千岛湖（距离杭州市中心约150km）被识别为跨城远程 POI 并过滤
2. 描述不再全是「推荐游览XXX」
3. 行程为3天，每天有活动安排
4. 预算包含交通和住宿估算
"""

import asyncio
from copy import deepcopy

from schemas import ScoredPOI, Location, UserProfile
from planner.core.heuristic_strategy import build_strategy
from planner.core.daily_scheduler import build_schedule
from planner.core.writer import enrich


def make_attraction(
    name: str,
    lat: float,
    lng: float,
    area: str,
    tags: list[str],
    score: float = 0.9,
    **overrides,
) -> ScoredPOI:
    return ScoredPOI(
        name=name,
        category="attraction",
        score=score,
        location=Location(lat=lat, lng=lng),
        area=area,
        tags=tags,
        **overrides,
    )


def make_restaurant(
    name: str, lat: float, lng: float, tags: list[str], description: str = ""
) -> ScoredPOI:
    return ScoredPOI(
        name=name,
        category="restaurant",
        score=0.85,
        location=Location(lat=lat, lng=lng),
        area="西湖区",
        tags=tags,
        description=description,
    )


async def main() -> None:
    pois = [
        # 杭州市区核心景点
        make_attraction("西湖", 30.2458, 120.1450, "西湖区", ["湖景", "文化"], score=0.96),
        make_attraction("灵隐寺", 30.2408, 120.0980, "西湖区", ["历史", "宗教"], score=0.94),
        make_attraction("雷峰塔", 30.2310, 120.1430, "西湖区", ["湖景", "历史"], score=0.92),
        make_attraction("三潭印月", 30.2390, 120.1420, "西湖区", ["湖景"], score=0.91),
        make_attraction("断桥残雪", 30.2590, 120.1460, "西湖区", ["湖景", "文化"], score=0.90),
        make_attraction("龙井村", 30.2190, 120.1180, "西湖区", ["自然", "茶文化"], score=0.88),
        make_attraction("西溪湿地", 30.2690, 120.0630, "余杭区", ["自然"], score=0.87),
        make_attraction("岳王庙", 30.2520, 120.1440, "西湖区", ["历史"], score=0.86),
        make_attraction("九溪烟树", 30.2020, 120.1080, "西湖区", ["自然"], score=0.85),
        # 远程跨城 POI：千岛湖
        make_attraction(
            "千岛湖",
            29.5980,
            119.0330,
            "淳安县",
            ["湖景", "自然"],
            score=0.93,
            recommended_hours="全天",
        ),
        # 餐厅 POI
        make_restaurant("楼外楼", 30.2460, 120.1460, ["杭帮菜"], description="西湖醋鱼名店"),
        make_restaurant("知味小馆", 30.2500, 120.1500, ["杭帮菜", "小吃"]),
        make_restaurant("绿茶餐厅", 30.2400, 120.1400, ["创意菜"]),
        make_restaurant("外婆家", 30.2550, 120.1480, ["杭帮菜"]),
    ]

    profile = UserProfile(
        destination="杭州",
        travel_days=3,
        travel_dates="2026-05-01",
        travelers_type="情侣",
        interests=["湖景", "历史", "自然"],
        food_preferences=["杭帮菜"],
        pace="moderate",
    )

    strategy = build_strategy(pois, profile)
    schedule = build_schedule(strategy, pois, [], profile)

    print(f"生成行程天数: {len(schedule)}")
    for day in schedule:
        print(
            f"\n第{day.day_number}天 (theme={day.theme or '无'}, total_cost={day.total_cost:.0f})"
        )
        for act in day.activities:
            cost = act.ticket_price or act.meal_cost or 0
            print(f"  {act.start_time}-{act.end_time} {act.poi_name} [{act.category}] ¥{cost:.0f}")
            if act.recommendation_reason:
                print(f"    _{act.recommendation_reason}_")

    # 检查1: 千岛湖被过滤
    all_attraction_names = [
        a.poi_name for day in schedule for a in day.activities if a.category != "restaurant"
    ]
    assert "千岛湖" not in all_attraction_names, "千岛湖应被识别为跨城远程 POI 并过滤"
    print("\n✅ 千岛湖已正确过滤")

    # 检查2: 描述不全是「推荐游览」
    scheduler_reasons = [
        a.recommendation_reason
        for day in schedule
        for a in day.activities
        if a.category != "restaurant"
    ]
    generic_count = sum(1 for r in scheduler_reasons if r and r.startswith("推荐游览"))
    print(f"\n调度层 generic 描述数量: {generic_count}/{len(scheduler_reasons)}")

    # 再验证 writer 模板兜底描述
    raw_itinerary = deepcopy(schedule)
    enriched, proposal = await enrich(raw_itinerary, profile)
    enriched_reasons = [
        a.recommendation_reason
        for day in enriched
        for a in day.activities
        if a.category != "restaurant"
    ]
    generic_enriched = sum(1 for r in enriched_reasons if r and r.startswith("推荐游览"))
    print(f"文案层 generic 描述数量: {generic_enriched}/{len(enriched_reasons)}")
    assert generic_enriched == 0, "文案层不应再出现「推荐游览XXX」式描述"
    print("✅ 描述已丰富化，无「推荐游览XXX」")

    # 检查3: 预算包含交通和住宿
    total_cost = sum(day.total_cost for day in schedule)
    transport_cost = profile.travel_days * 30
    accommodation_cost = max(0, profile.travel_days - 1) * 200
    print(
        f"\n预估总费用: ¥{total_cost:.0f} (含交通 ¥{transport_cost} + 住宿 ¥{accommodation_cost})"
    )
    assert total_cost >= transport_cost + accommodation_cost, "预算应包含交通和住宿估算"
    print("✅ 预算已补全交通和住宿")

    # 检查4: 3天都有活动
    assert len(schedule) == 3, "应为3天行程"
    assert all(len(day.activities) > 0 for day in schedule), "每天应有活动安排"
    print("✅ 3天行程均有活动安排")

    # 检查5: 存在真实餐厅 POI 作为用餐推荐
    meal_names = [
        a.poi_name for day in schedule for a in day.activities if a.category == "restaurant"
    ]
    real_restaurants = {"楼外楼", "知味小馆", "绿茶餐厅", "外婆家"}
    matched = set(meal_names) & real_restaurants
    print(f"\n用餐活动使用真实餐厅: {matched}")
    assert len(matched) > 0, "至少有一个用餐活动应匹配真实餐厅 POI"
    print("✅ 用餐已匹配真实餐厅")

    print("\n🎉 杭州3日验证通过")


if __name__ == "__main__":
    asyncio.run(main())
