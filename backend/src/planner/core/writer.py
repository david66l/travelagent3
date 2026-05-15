"""Phase 2C Writer — enrich itinerary with prose without mutating facts.

Every enrichment runs through fact_checksum.verify_checksum() before
being accepted.  Writer can only decorate; never alter structural fields.
"""

from copy import deepcopy
from typing import Optional

from schemas import DayPlan, UserProfile
from planner.core.fact_checksum import compute_checksum, verify_checksum


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def enrich(
    itinerary: list[DayPlan],
    profile: UserProfile,
) -> tuple[list[DayPlan], str]:
    """Enrich itinerary with themes and recommendation reasons.

    Returns (enriched_itinerary, proposal_text).  If enrichment would
    alter any protected field, the original itinerary is returned
    unchanged with a fallback proposal.
    """
    checksum_before = compute_checksum(itinerary)
    enriched = deepcopy(itinerary)

    try:
        _assign_day_themes(enriched, profile)
        _enrich_reasons(enriched, profile)
    except Exception:
        return itinerary, _fallback_proposal(itinerary, profile)

    if not verify_checksum(itinerary, enriched):
        return itinerary, _fallback_proposal(itinerary, profile)

    proposal = _build_proposal(enriched, profile)
    return enriched, proposal


def enrich_safe(
    itinerary: list[DayPlan],
    profile: UserProfile,
) -> tuple[list[DayPlan], str]:
    """Same as enrich() but guaranteed to never alter itinerary facts."""
    enriched, proposal = enrich(itinerary, profile)
    return enriched, proposal


# --------------------------------------------------------------------------- #
# Day themes
# --------------------------------------------------------------------------- #


def _assign_day_themes(itinerary: list[DayPlan], profile: UserProfile) -> None:
    """Assign a human-readable theme to each day based on its activities."""
    for day in itinerary:
        if day.theme:
            continue

        categories = {a.category for a in day.activities}
        tags = {t for a in day.activities for t in (a.tags or [])}
        area = {a.poi_name for a in day.activities if a.category == "attraction"}

        if "历史" in tags and "文化" in tags:
            day.theme = "历史文化之旅"
        elif "园林" in tags or "湖景" in tags:
            day.theme = "园林湖景休闲"
        elif "登山" in tags:
            day.theme = "户外探索"
        elif "文艺" in tags or "艺术" in tags:
            day.theme = "文艺漫游"
        elif "美食" in tags:
            day.theme = "美食寻味"
        elif "夜景" in tags:
            day.theme = "都市夜色"
        elif "购物" in tags:
            day.theme = "购物休闲"
        else:
            day.theme = f"{profile.destination or '目的地'}探索"


# --------------------------------------------------------------------------- #
# Activity reasons
# --------------------------------------------------------------------------- #


_REASON_TEMPLATES: dict[str, str] = {
    "故宫": "世界文化遗产，明清两代皇宫，中华文明的象征",
    "天坛": "明清代帝王祭天场所，建筑学杰作",
    "颐和园": "中国现存最大的皇家园林，昆明湖与万寿山相映成趣",
    "长城": "世界七大奇迹之一，中华民族的脊梁",
    "外滩": "万国建筑博览群，上海最经典的城市名片",
    "东方明珠": "上海地标性建筑，俯瞰浦江两岸的绝佳位置",
    "豫园": "明代江南园林代表作，感受老上海的风雅",
    "南京路步行街": "中华商业第一街，购物者的天堂",
    "田子坊": "石库门里弄里的创意艺术区，适合闲逛拍照",
    "新天地": "石库门与现代时尚的完美融合",
    "上海博物馆": "馆藏丰富的综合性博物馆，历史爱好者必去",
    "南锣鼓巷": "老北京胡同文化代表，美食与文艺小店林立",
    "798艺术区": "工业遗址改造的当代艺术圣地",
    "鸟巢": "2008年奥运会主体育场，现代建筑奇迹",
    "雍和宫": "北京最大的藏传佛教寺院，香火鼎盛",
}


def _enrich_reasons(itinerary: list[DayPlan], profile: UserProfile) -> None:
    """Add recommendation_reason to activities that lack one."""
    food_hint = (
        f"品尝{','.join(profile.food_preferences)}的好去处"
        if profile.food_preferences
        else "口碑推荐"
    )

    for day in itinerary:
        for act in day.activities:
            if act.recommendation_reason:
                continue

            if act.poi_name in _REASON_TEMPLATES:
                act.recommendation_reason = _REASON_TEMPLATES[act.poi_name]
            elif act.category == "restaurant":
                act.recommendation_reason = food_hint
            elif act.category == "attraction":
                act.recommendation_reason = f"推荐游览{act.poi_name}"
            else:
                act.recommendation_reason = f"体验{act.poi_name}"


# --------------------------------------------------------------------------- #
# Proposal text
# --------------------------------------------------------------------------- #


def _build_proposal(
    itinerary: list[DayPlan], profile: UserProfile
) -> str:
    """Build a rich markdown proposal from the enriched itinerary."""
    lines: list[str] = []
    dest = profile.destination or "目的地"
    days = profile.travel_days or len(itinerary)

    # Header
    lines.append(f"# {dest} {days}日游行程方案\n")

    if profile.travel_dates:
        lines.append(f"**出行日期**: {profile.travel_dates}")
    if profile.travelers_type:
        lines.append(f"**出行类型**: {profile.travelers_type}")
    if profile.interests:
        lines.append(f"**兴趣偏好**: {'、'.join(profile.interests)}")
    lines.append("")

    # Budget
    total_cost = sum(day.total_cost for day in itinerary)
    lines.append(f"**预估总费用**: ¥{total_cost:.0f}")
    if profile.budget_range:
        ratio = total_cost / profile.budget_range
        if ratio <= 1.0:
            lines.append(f"✅ 在预算 ¥{profile.budget_range:.0f} 之内")
        elif ratio <= 1.2:
            lines.append(f"⚠️ 略超预算 ¥{profile.budget_range:.0f}（+{ratio - 1:.0%}）")
        else:
            lines.append(f"❌ 超出预算 ¥{profile.budget_range:.0f} 的 20%")
    lines.append("")

    # Day-by-day
    for day in itinerary:
        lines.append(f"## 第{day.day_number}天" + (f" — {day.theme}" if day.theme else ""))
        if day.date:
            lines.append(f"**日期**: {day.date}")
        lines.append("")

        for i, act in enumerate(day.activities, 1):
            time_str = ""
            if act.start_time and act.end_time:
                time_str = f" ({act.start_time}-{act.end_time})"

            cost_str = ""
            if act.ticket_price:
                cost_str = f" — ¥{act.ticket_price:.0f}"
            elif act.meal_cost:
                cost_str = f" — ¥{act.meal_cost:.0f}"

            lines.append(f"{i}. **{act.poi_name}**{time_str}{cost_str}")
            if act.recommendation_reason:
                lines.append(f"   _{act.recommendation_reason}_")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*由 TravelAgent 自动生成 · {dest} {days}日行程*")
    lines.append("")

    return "\n".join(lines)


def _fallback_proposal(
    itinerary: list[DayPlan], profile: UserProfile
) -> str:
    """Minimal proposal used when enrichment fails checksum validation."""
    dest = profile.destination or "目的地"
    days = profile.travel_days or len(itinerary)
    lines = [
        f"# {dest} {days}日游行程方案\n",
        f"**预估总费用**: ¥{sum(d.total_cost for d in itinerary):.0f}\n",
    ]
    for day in itinerary:
        lines.append(f"## 第{day.day_number}天")
        for act in day.activities:
            t = f" ({act.start_time}-{act.end_time})" if act.start_time else ""
            lines.append(f"- **{act.poi_name}**{t}")
        lines.append("")
    return "\n".join(lines)
