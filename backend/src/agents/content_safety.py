"""Content safety engine for travel itineraries."""

from __future__ import annotations

from typing import Any

from schemas import Activity, DayPlan, ValidationResult


class ContentSafetyEngine:
    """Detect shopping trips, illegal routes and unsafe activities."""

    SHOPPING_KEYWORDS: tuple[str, ...] = (
        "购物团",
        "零负团费",
        "强制购物",
        "必进购物店",
    )

    ILLEGAL_ROUTE_KEYWORDS: tuple[str, ...] = (
        "军事禁区",
        "未开放边境",
        "非法穿越",
        "缅北",
    )

    UNSAFE_ACTIVITY_KEYWORDS: tuple[str, ...] = (
        "徒手攀岩",
        "无保护潜水",
        "野泳",
        "未开发洞穴",
        "极限跳伞",
    )

    @classmethod
    def detect_shopping_trip(
        cls,
        itinerary: list[DayPlan] | list[dict[str, Any]] | None,
        user_input: str | None,
    ) -> tuple[bool, float, str]:
        """Return (flagged, score, suggestion)."""
        user_input = user_input or ""

        text = user_input
        if _contains_any(text, cls.SHOPPING_KEYWORDS):
            return True, 0.0, "检测到购物团/强制购物关键词，拒绝生成行程"

        if not itinerary:
            return False, 1.0, ""

        activities = _flatten_activities(itinerary)
        if not activities:
            return False, 1.0, ""

        for activity in activities:
            note = _activity_text(activity)
            if _contains_any(note, cls.SHOPPING_KEYWORDS):
                return True, 0.0, "检测到购物团/强制购物关键词，拒绝生成行程"

        shopping_count = sum(1 for a in activities if _is_shopping_activity(a))
        shopping_ratio = shopping_count / len(activities)

        if shopping_ratio > 0.4:
            total_budget = _estimate_total_budget(activities)
            day_count = len(itinerary) if itinerary else 1
            avg_daily_budget = total_budget / day_count if day_count > 0 else 0.0
            if avg_daily_budget < 200:
                return (
                    True,
                    0.0,
                    "购物项目占比过高且日均预算极低，疑似购物团",
                )

        score = 1.0 - shopping_ratio * 0.5
        return False, max(score, 0.0), ""

    @classmethod
    def detect_illegal_route(
        cls,
        itinerary: list[DayPlan] | list[dict[str, Any]] | None,
    ) -> tuple[bool, float, str]:
        """Return (flagged, score, suggestion)."""
        if not itinerary:
            return False, 1.0, ""

        activities = _flatten_activities(itinerary)
        for activity in activities:
            text = _activity_text(activity)
            if _contains_any(text, cls.ILLEGAL_ROUTE_KEYWORDS):
                return True, 0.0, f"检测到违规路线关键词：{text[:30]}"

        return False, 1.0, ""

    @classmethod
    def detect_unsafe_activity(
        cls,
        itinerary: list[DayPlan] | list[dict[str, Any]] | None,
    ) -> tuple[bool, float, str]:
        """Return (flagged, score, suggestion)."""
        if not itinerary:
            return False, 1.0, ""

        activities = _flatten_activities(itinerary)
        for activity in activities:
            text = _activity_text(activity)
            if _contains_any(text, cls.UNSAFE_ACTIVITY_KEYWORDS):
                return True, 0.0, f"检测到高风险活动：{text[:30]}"

        return False, 1.0, ""

    @classmethod
    def check(cls, state: dict[str, Any]) -> ValidationResult:
        """Run all safety detectors and return a ValidationResult."""
        user_input = state.get("user_input", "") if isinstance(state, dict) else ""
        itinerary = state.get("itinerary") if isinstance(state, dict) else None

        # Defensive: missing data means nothing to block.
        if not itinerary and not user_input:
            return ValidationResult(passed=True)

        shopping_flag, shopping_score, shopping_suggestion = cls.detect_shopping_trip(
            itinerary, user_input
        )
        illegal_flag, illegal_score, illegal_suggestion = cls.detect_illegal_route(
            itinerary
        )
        unsafe_flag, unsafe_score, unsafe_suggestion = cls.detect_unsafe_activity(
            itinerary
        )

        scores = {
            "shopping_trip": shopping_score,
            "illegal_route": illegal_score,
            "unsafe_activity": unsafe_score,
        }
        total_score = round(sum(scores.values()) / len(scores), 2)

        critical_failures: list[str] = []
        suggestions: list[str] = []

        if shopping_flag:
            critical_failures.append("shopping_trip")
            suggestions.append(shopping_suggestion)
        if illegal_flag:
            critical_failures.append("illegal_route")
            suggestions.append(illegal_suggestion)
        if unsafe_flag:
            critical_failures.append("unsafe_activity")
            suggestions.append(unsafe_suggestion)

        passed = not critical_failures

        return ValidationResult(
            passed=passed,
            scores=scores,
            total_score=total_score,
            critical_failures=critical_failures,
            improvement_suggestions=suggestions,
        )


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    """Check if any keyword appears in text."""
    return any(keyword in text for keyword in keywords)


def _flatten_activities(
    itinerary: list[DayPlan] | list[dict[str, Any]],
) -> list[Activity] | list[dict[str, Any]]:
    """Collect activities from a list of DayPlans or dicts."""
    activities: list[Any] = []
    for day in itinerary:
        if isinstance(day, DayPlan):
            activities.extend(day.activities or [])
        elif isinstance(day, dict):
            for activity in day.get("activities") or []:
                activities.append(activity)
    return activities


def _is_shopping_activity(activity: Activity | dict[str, Any]) -> bool:
    """Return True if activity is shopping-related."""
    category = _get_field(activity, "category", "").lower()
    tags = _get_field(activity, "tags", [])
    if category == "shopping":
        return True
    return any("购物" in tag for tag in tags)


def _activity_text(activity: Activity | dict[str, Any]) -> str:
    """Concatenate textual fields of an activity."""
    parts = [
        _get_field(activity, "poi_name", ""),
        _get_field(activity, "recommendation_reason", ""),
        _get_field(activity, "note", ""),
    ]
    tags = _get_field(activity, "tags", [])
    parts.append(" ".join(str(tag) for tag in tags))
    return " ".join(part for part in parts if part)


def _estimate_total_budget(activities: list[Activity] | list[dict[str, Any]]) -> float:
    """Sum known cost fields across activities."""
    total = 0.0
    for activity in activities:
        for field in ("ticket_price", "meal_cost", "transport_cost"):
            value = _get_field(activity, field, None)
            if isinstance(value, (int, float)):
                total += value
    return total


def _get_field(activity: Activity | dict[str, Any], field: str, default: Any) -> Any:
    """Read an attribute or dict key from an activity."""
    if isinstance(activity, Activity):
        return getattr(activity, field, default)
    if isinstance(activity, dict):
        return activity.get(field, default)
    return default
