"""PRD-aligned Redis cache key builders (§4.8.2)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from schemas import UserProfile

CACHE_VERSION = "v2"


def _query_hash(parts: list[str]) -> str:
    payload = "|".join(sorted(parts))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def poi_key(city: str, category: str | None, keywords: list[str]) -> str:
    cat = category or "all"
    qh = _query_hash([str(k) for k in keywords or []])
    return f"poi:{city}:{cat}:{qh}:{CACHE_VERSION}"


def weather_key(city: str, start_date: str, end_date: str) -> str:
    if start_date and end_date and start_date != end_date:
        date_part = f"{start_date}_{end_date}"
    else:
        date_part = start_date or end_date or "unknown"
    return f"weather:{city}:{date_part}:{CACHE_VERSION}"


def price_key(poi_id: str, date: str, price_type: str = "default") -> str:
    return f"price:{poi_id}:{date}:{price_type}:{CACHE_VERSION}"


def route_key(origin: dict[str, Any], destination: dict[str, Any], mode: str) -> str:
    payload = json.dumps(
        {"origin": origin, "destination": destination, "mode": mode}, sort_keys=True
    )
    h = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return f"route:{h}:{CACHE_VERSION}"


def itinerary_draft_key(profile: UserProfile | dict[str, Any]) -> str:
    if not isinstance(profile, UserProfile):
        profile = UserProfile(**profile)
    city = profile.destination or "unknown"
    days = profile.travel_days or 0
    budget = int(profile.budget_range or 0)
    theme_parts = [str(x) for x in profile.interests or []]
    theme_parts += [str(x) for x in profile.food_preferences or []]
    theme_parts.append(profile.pace or "")
    if profile.travelers_type:
        theme_parts.append(profile.travelers_type)
    if profile.accommodation_preference:
        theme_parts.append(profile.accommodation_preference)
    theme = _query_hash(theme_parts)
    return f"itinerary:{city}:{theme}:{days}:{budget}:{CACHE_VERSION}"


def tool_cache_key(tool_name: str, params: dict[str, Any]) -> str:
    """Map tool invocations to PRD keys; unknown tools keep legacy hashed keys."""
    if tool_name == "poi_search":
        return poi_key(
            str(params.get("city", "")),
            params.get("category"),
            list(params.get("keywords") or []),
        )
    if tool_name == "weather":
        return weather_key(
            str(params.get("city", "")),
            str(params.get("start_date", "")),
            str(params.get("end_date", "")),
        )
    if tool_name == "route":
        return route_key(
            dict(params.get("origin") or {}),
            dict(params.get("destination") or {}),
            str(params.get("mode", "transit")),
        )
    if tool_name in ("price", "price_query"):
        from datetime import date

        poi_id = str(params.get("poi_name") or params.get("poi_id") or "")
        day = str(params.get("date") or date.today().isoformat())
        price_type = str(params.get("price_type") or "default")
        return price_key(poi_id, day, price_type)
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    return f"tool:{tool_name}:{digest}"
