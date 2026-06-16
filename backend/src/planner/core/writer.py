"""Phase 2C Writer — enrich itinerary with LLM-generated prose.

Phase 2C uses a day-batch enrichment strategy: one LLM call generates the day
theme and all activity recommendation reasons for that day.  This is cheaper
than per-activity calls because the shared context (system prompt, user
profile) is paid for once per day.

If the day-batch call fails or returns invalid data, the pipeline falls back to
per-activity LLM enrichment (with per-activity validation and retry).  If a
single activity still cannot be enriched, a template fallback is used for that
activity only — other activities keep their enriched prose.

Protected fields (poi_name, start_time, end_time, duration_min, ticket_price,
location lat/lng) can never be mutated by enrichment.
"""

import asyncio
import logging
from copy import deepcopy
from typing import Optional

from core.llm_client import llm
from schemas import Activity, DayPlan, UserProfile
from planner.core.fact_guard import activity_fields_match, protected_field_differences

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Template fallback (single-activity, formerly _REASON_TEMPLATES)
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


def _template_reason(poi_name: str, category: str) -> str:
    """Single-activity fallback when LLM enrichment fails."""
    if poi_name in _REASON_TEMPLATES:
        return _REASON_TEMPLATES[poi_name]
    if category == "restaurant":
        return "口碑推荐"
    if category == "attraction":
        return f"推荐游览{poi_name}"
    return f"体验{poi_name}"


def _template_theme(day_activities: list[Activity]) -> str:
    """Rule-based day theme when LLM is unavailable."""
    tags: set[str] = set()
    for a in day_activities:
        for t in a.tags or []:
            tags.add(t)

    if "历史" in tags and "文化" in tags:
        return "历史文化之旅"
    if "园林" in tags or "湖景" in tags:
        return "园林湖景休闲"
    if "登山" in tags:
        return "户外探索"
    if "文艺" in tags or "艺术" in tags:
        return "文艺漫游"
    if "美食" in tags:
        return "美食寻味"
    if "夜景" in tags:
        return "都市夜色"
    if "购物" in tags:
        return "购物休闲"
    return "精彩探索"


# --------------------------------------------------------------------------- #
# Day-batch LLM enrichment
# --------------------------------------------------------------------------- #

_BATCH_ENRICHMENT_TIMEOUT = 30.0  # seconds — batch output is longer

_BUILD_DAY_ENRICHMENT_PROMPT = """请为以下一日行程生成主题和每个景点的推荐语。

用户画像：{travelers_type}，兴趣是{interests}，节奏偏好{pace}

一日行程：
{activities}

要求：
1. theme: 4-8个字的中文主题名，能概括当天行程特色
2. 每个景点必须原样返回 poi_name，并给出 recommendation_reason（20-40字推荐语）和 tags（2-3个标签）
3. 推荐语根据用户画像个性化——比如亲子游强调"适合带孩子"，情侣游强调"浪漫"，历史爱好者强调"文化底蕴"
4. 仅输出 JSON，不要其他内容

返回 JSON 格式：
{{
  "theme": "...",
  "activities": [
    {{"poi_name": "...", "recommendation_reason": "...", "tags": ["...", "..."]}},
    {{"poi_name": "...", "recommendation_reason": "...", "tags": ["...", "..."]}}
  ]
}}"""


async def _llm_enrich_day_batch(
    day: DayPlan,
    profile: UserProfile,
) -> Optional[dict]:
    """Call LLM to enrich a whole day (theme + all activities). Returns parsed JSON dict or None."""
    if not day.activities:
        return None

    interests = "、".join(profile.interests) if profile.interests else "无特殊偏好"
    travelers_type = profile.travelers_type or "普通游客"
    pace = profile.pace or "适中"

    activity_lines = []
    for act in day.activities:
        time_str = f" {act.start_time}-{act.end_time}" if act.start_time and act.end_time else ""
        activity_lines.append(f"- {time_str} {act.poi_name} [{act.category}]")

    prompt = _BUILD_DAY_ENRICHMENT_PROMPT.format(
        travelers_type=travelers_type,
        interests=interests,
        pace=pace,
        activities="\n".join(activity_lines),
    )

    try:
        result = await asyncio.wait_for(
            llm.json_chat(
                messages=[
                    {"role": "system", "content": "你是一个专业旅行文案写手，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1024,
            ),
            timeout=_BATCH_ENRICHMENT_TIMEOUT,
        )
        if not isinstance(result, dict):
            return None
        return result
    except asyncio.TimeoutError:
        logger.warning("LLM day-batch enrichment timeout for day %d", day.day_number)
        return None
    except Exception as exc:
        logger.warning("LLM day-batch enrichment failed for day %d: %s", day.day_number, exc)
        return None


async def _enrich_day_batch(
    day: DayPlan,
    profile: UserProfile,
) -> bool:
    """Try to enrich an entire day with one LLM call.

    On success, mutates ``day`` in place and returns True.
    On failure, leaves ``day`` unchanged and returns False so the caller can
    fall back to per-activity enrichment.
    """
    batch_result = await _llm_enrich_day_batch(day, profile)
    if not batch_result:
        return False

    # Theme
    theme = str(batch_result.get("theme", "")).strip()
    if theme and 2 <= len(theme) <= 12:
        day.theme = theme

    # Map results by poi_name so LLM reordering does not break matching
    raw_items = batch_result.get("activities", [])
    if not isinstance(raw_items, list):
        return False

    result_by_name: dict[str, dict] = {}
    for item in raw_items:
        if isinstance(item, dict) and item.get("poi_name"):
            result_by_name[str(item["poi_name"]).strip()] = item

    new_activities: list[Activity] = []
    batch_valid = True

    for activity in day.activities:
        item = result_by_name.get(activity.poi_name)
        if not item:
            batch_valid = False
            # Missing enrichment for this POI — fall back to per-activity
            new_activities.append(await _enrich_activity_with_retry(activity, profile))
            continue

        reason = str(item.get("recommendation_reason", "")).strip()
        raw_tags = item.get("tags", [])
        tags = list(raw_tags) if isinstance(raw_tags, list) else []

        if not reason:
            batch_valid = False
            new_activities.append(await _enrich_activity_with_retry(activity, profile))
            continue

        candidate = deepcopy(activity)
        candidate.recommendation_reason = reason
        candidate.tags = list(set((activity.tags or []) + tags))

        # Fact Guard: LLM must not mutate protected fields
        if not activity_fields_match(activity, candidate):
            changed_fields = protected_field_differences(activity, candidate)
            logger.warning(
                "Day-batch Fact Guard failed for %s fields=%s",
                activity.poi_name,
                ",".join(changed_fields) or "unknown",
            )
            batch_valid = False
            new_activities.append(await _enrich_activity_with_retry(activity, profile))
            continue

        new_activities.append(candidate)

    if new_activities:
        day.activities = new_activities

    return batch_valid


# --------------------------------------------------------------------------- #
# Per-activity LLM enrichment (fallback)
# --------------------------------------------------------------------------- #

_ENRICHMENT_TIMEOUT = 10.0  # seconds — single-activity fallback

_BUILD_ENRICHMENT_PROMPT = """请为以下景点写一句简短的中文推荐语。

景点：{poi_name}
类型：{category}
用户画像：{travelers_type}，兴趣是{interests}，节奏偏好{pace}

要求：
1. 仅输出一句推荐语（20-40字），不提价格或具体时间
2. 根据用户画像个性化——比如亲子游强调"适合带孩子"，情侣游强调"浪漫"
3. 如果该景点适合特定时间段（如夜景、早茶），可在推荐语中自然提及

返回 JSON 格式：
{{"recommendation_reason": "...", "tags": ["标签1", "标签2"]}}"""


async def _llm_enrich_activity(
    activity: Activity,
    profile: UserProfile,
) -> Optional[tuple[str, list[str]]]:
    """Call LLM to enrich a single activity.  Returns (reason, tags) or None on failure."""
    interests = "、".join(profile.interests) if profile.interests else "无特殊偏好"
    travelers_type = profile.travelers_type or "普通游客"
    pace = profile.pace or "适中"

    prompt = _BUILD_ENRICHMENT_PROMPT.format(
        poi_name=activity.poi_name,
        category=activity.category,
        travelers_type=travelers_type,
        interests=interests,
        pace=pace,
    )

    try:
        result = await asyncio.wait_for(
            llm.json_chat(
                messages=[
                    {"role": "system", "content": "你是一个专业旅行文案写手，只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=256,
            ),
            timeout=_ENRICHMENT_TIMEOUT,
        )
        reason = str(result.get("recommendation_reason", "")).strip()
        tags = list(result.get("tags", [])) if isinstance(result.get("tags"), list) else []
        if not reason:
            return None
        return reason, tags
    except asyncio.TimeoutError:
        logger.warning("LLM enrichment timeout for %s", activity.poi_name)
        return None
    except Exception as exc:
        logger.warning("LLM enrichment failed for %s: %s", activity.poi_name, exc)
        return None


async def _enrich_activity_with_retry(
    activity: Activity,
    profile: UserProfile,
    max_retries: int = 2,
) -> Activity:
    """Enrich a single activity with LLM, validate, retry on mutation, fallback."""
    # LLM not available or enrichment skipped — use template
    enriched = deepcopy(activity)

    for attempt in range(max_retries + 1):
        llm_result = await _llm_enrich_activity(activity, profile)
        if llm_result is None:
            # LLM call failed — try again or fallback
            if attempt < max_retries:
                continue
            enriched.recommendation_reason = _template_reason(activity.poi_name, activity.category)
            return enriched

        reason, tags = llm_result
        enriched.recommendation_reason = reason
        enriched.tags = list(set((activity.tags or []) + tags))

        # Validate: LLM must not mutate protected fields
        if activity_fields_match(activity, enriched):
            return enriched

        changed_fields = protected_field_differences(activity, enriched)
        logger.warning(
            "Fact Guard failed for %s fields=%s (attempt %d/%d)",
            activity.poi_name,
            ",".join(changed_fields) or "unknown",
            attempt + 1,
            max_retries + 1,
        )
        # Reset for retry
        enriched = deepcopy(activity)

    # All retries exhausted — single-activity template fallback
    enriched.recommendation_reason = _template_reason(activity.poi_name, activity.category)
    return enriched


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


async def enrich(
    itinerary: list[DayPlan],
    profile: UserProfile,
) -> tuple[list[DayPlan], str]:
    """Enrich itinerary with LLM-generated themes and recommendation reasons.

    Strategy:
    1. Try to enrich each day with a single LLM call (theme + all activities).
       This is cheaper because the shared context is paid for once per day.
    2. If the day-batch call fails for a day, fall back to per-activity LLM
       enrichment with per-activity validation and retry.
    3. If a single activity still cannot be enriched, use a template fallback
       for that activity only.

    Returns (enriched_itinerary, proposal_text).  If the entire enrichment
    process fails, returns the original itinerary with a fallback proposal.
    """
    try:
        enriched = deepcopy(itinerary)

        for day in enriched:
            # Primary path: one LLM call per day
            batch_ok = await _enrich_day_batch(day, profile)

            if not batch_ok:
                # Fallback path: per-activity LLM + rule-based theme
                if not day.theme:
                    day.theme = _template_theme(day.activities)
                for i, activity in enumerate(day.activities):
                    day.activities[i] = await _enrich_activity_with_retry(activity, profile)

        proposal = _build_proposal(enriched, profile)
        return enriched, proposal

    except Exception:
        logger.exception("Enrichment failed — returning original itinerary")
        return itinerary, _fallback_proposal(itinerary, profile)


# --------------------------------------------------------------------------- #
# Proposal text
# --------------------------------------------------------------------------- #


def _build_proposal(itinerary: list[DayPlan], profile: UserProfile) -> str:
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


def _fallback_proposal(itinerary: list[DayPlan], profile: UserProfile) -> str:
    """Minimal proposal used when enrichment fails entirely."""
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
