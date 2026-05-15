"""Deterministic heuristic strategy engine — zero LLM dependency."""

from collections import Counter

from schemas import ScoredPOI, UserProfile
from planner.core.models import Strategy, DayTheme, AreaGroup


# Landmark keywords for must-see detection
_LANDMARK_TAGS = {
    "博物馆",
    "世界遗产",
    "地标",
    "历史",
    "文化",
    "古迹",
    "标志性建筑",
    "全国重点文物保护单位",
}


def build_strategy(pois: list[ScoredPOI], profile: UserProfile) -> Strategy:
    """Build a deterministic strategy from POIs and user profile.

    Runs in < 200ms, no external API calls, no LLM.
    """
    if not pois:
        return Strategy()

    travel_days = profile.travel_days or 1

    # 1. Group POIs by area
    area_groups = _group_by_area(pois)

    # 2. Assign top N area groups to days
    assigned_groups = area_groups[:travel_days]
    remaining_groups = area_groups[travel_days:]

    # 3. Build day themes from assigned groups
    day_themes = _build_day_themes(assigned_groups, remaining_groups, travel_days)

    # 4. Detect must-see POIs
    must_see = _detect_must_see(pois, profile)

    return Strategy(
        day_themes=day_themes,
        area_groups=area_groups,
        must_see=must_see,
    )


def _group_by_area(pois: list[ScoredPOI]) -> list[AreaGroup]:
    """Group POIs by area, sorted by density (descending)."""
    groups: dict[str, list[ScoredPOI]] = {}
    for poi in pois:
        area = poi.area or "其他"
        groups.setdefault(area, []).append(poi)

    # Sort by group size descending, then by average score
    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (len(item[1]), sum(p.score for p in item[1]) / len(item[1])),
        reverse=True,
    )

    return [
        AreaGroup(
            area=area,
            poi_names=[p.name for p in pois_in_area],
            priority_score=sum(p.score for p in pois_in_area) / len(pois_in_area),
        )
        for area, pois_in_area in sorted_groups
    ]


def _build_day_themes(
    assigned_groups: list[AreaGroup],
    remaining_groups: list[AreaGroup],
    travel_days: int,
) -> list[DayTheme]:
    """Infer day themes from area groups and tags."""
    day_themes = []

    # Build a pool of all POI names from remaining groups (for overflow days)
    overflow_names = []
    for g in remaining_groups:
        overflow_names.extend(g.poi_names)

    for day_idx in range(travel_days):
        day_number = day_idx + 1

        if day_idx < len(assigned_groups):
            group = assigned_groups[day_idx]
            area_focus = group.area
            # Use overflow names to fill empty days
            names = group.poi_names[:]
        else:
            area_focus = "综合"
            names = overflow_names[:5]
            overflow_names = overflow_names[5:]

        # Theme = most frequent tag among these POIs (placeholder; scheduler fills details)
        theme = f"{area_focus}探索"

        day_themes.append(
            DayTheme(
                day_number=day_number,
                theme=theme,
                area_focus=area_focus,
                primary_tags=[],
            )
        )

    return day_themes


def _detect_must_see(pois: list[ScoredPOI], profile: UserProfile) -> list[str]:
    """Detect must-see POIs: user mentions + high-score landmarks."""
    must_see: set[str] = set()

    # User-mentioned POIs (from special_requests or interests)
    user_text = " ".join(profile.special_requests + profile.interests).lower()
    for poi in pois:
        if poi.name.lower() in user_text:
            must_see.add(poi.name)

    # High-score landmarks
    for poi in pois:
        if poi.score >= 0.85 and (
            poi.category == "museum" or (set(poi.tags) & _LANDMARK_TAGS)
        ):
            must_see.add(poi.name)

    return list(must_see)
