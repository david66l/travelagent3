"""Tool executor for OpenAI-style tool calls.

Parses `tool_calls`, routes to the appropriate handler, and returns a list of
`ToolResult` objects. Each handler failure is isolated: a single failing tool
does not block other tools in the batch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
import unicodedata
from datetime import UTC, datetime, timedelta
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

import httpx

from planner.transport_router import HaversineFallback, MapServiceRouter
from agentic.observations import ObservationEnvelope
from agentic.guard import GuardContext, GuardDecision, ToolGuard
from core.settings import settings
from core.city_names import canonical_city_name
from evaluation.validator import ItineraryValidator
from schemas import ToolResult
from tools.tool_definitions import TOOL_NAME_TO_SCHEMA

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[ToolResult]]


_EXTERNAL_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_EXTERNAL_INVISIBLE_CHARS_RE = re.compile(
    "[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\ufe00-\ufe0f\U000e0000-\U000e007f]"
)
_INSTRUCTION_LIKE_EXTERNAL_CONTENT_RE = re.compile(
    r"(?i)(?:ignore\s+(?:all\s+)?(?:previous|above)\s+instructions|"
    r"system\s*override|<\s*system\s*>|\[\s*system\s*\]|"
    r"you\s+are\s+now|jailbreak|prompt\s*injection)"
)


def _normalize_external_text(value: Any, *, max_length: int) -> tuple[str, list[str]]:
    """Normalize untrusted provider text without treating it as instructions."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    flags: list[str] = []
    if _EXTERNAL_INVISIBLE_CHARS_RE.search(text):
        flags.append("invisible_unicode_removed")
        text = _EXTERNAL_INVISIBLE_CHARS_RE.sub("", text)
    if _EXTERNAL_CONTROL_CHARS_RE.search(text):
        flags.append("control_characters_removed")
        text = _EXTERNAL_CONTROL_CHARS_RE.sub(" ", text)
    text = " ".join(text.split())[:max_length]
    if _INSTRUCTION_LIKE_EXTERNAL_CONTENT_RE.search(text):
        flags.append("instruction_like_content")
    return text, flags


def _normalize_external_url(value: Any) -> str | None:
    """Allow only credential-free HTTP(S) evidence URLs."""
    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path, parsed.query, ""))


def _classify_tool_exception(exc: Exception) -> tuple[str, bool, dict[str, Any]]:
    """Map provider failures to a stable retry contract."""
    details: dict[str, Any] = {"error_type": type(exc).__name__}
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
        return "TOOL_TIMEOUT", True, details
    if isinstance(exc, httpx.HTTPStatusError):
        status = int(exc.response.status_code)
        details["status_code"] = status
        if status in {408, 429} or status >= 500:
            return "TOOL_HTTP_TRANSIENT", True, details
        if status in {401, 403}:
            return "TOOL_AUTH_ERROR", False, details
        return "TOOL_HTTP_PERMANENT", False, details
    if isinstance(exc, PermissionError):
        return "TOOL_AUTH_ERROR", False, details
    if isinstance(exc, (ConnectionError, OSError, httpx.NetworkError)):
        return "TOOL_NETWORK_ERROR", True, details
    if isinstance(exc, (TypeError, ValueError, KeyError)):
        return "TOOL_RESPONSE_INVALID", False, details
    return "TOOL_EXECUTION_ERROR", False, details


def _normalize_poi_name(value: Any) -> str:
    """Return a conservative identity key for POI entity matching."""
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\s·・,，。.!！?？:：;；()（）\[\]【】'\"“”‘’_-]+", "", normalized)


def _extract_transport_options(results: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    """Extract only source-backed schedules with an explicit depart/arrive pair."""
    options: list[dict[str, Any]] = []
    for item in results:
        corpus = f"{item.get('title') or ''} {item.get('snippet') or ''}"
        times = list(dict.fromkeys(re.findall(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d", corpus)))
        if len(times) < 2 or not item.get("url"):
            continue
        service_match = re.search(
            r"\b(?:[GDCZTK]\d{1,4}|[A-Z0-9]{2}\d{3,4})\b",
            corpus,
            flags=re.IGNORECASE,
        )
        options.append(
            {
                "service_code": service_match.group(0).upper() if service_match else None,
                "mode": mode,
                "departure_time": times[0],
                "arrival_time": times[1],
                "source_url": item["url"],
                "source_title": item.get("title"),
                "source_score": float(item.get("score") or 0),
            }
        )
    return sorted(options, key=lambda item: item["source_score"], reverse=True)


_ATTRACTION_PREFERENCE_TERMS: dict[str, tuple[str, ...]] = {
    "history_culture": (
        "历史",
        "文化",
        "人文",
        "博物馆",
        "纪念馆",
        "遗址",
        "古城",
        "古镇",
        "古街",
        "陵",
        "故居",
        "文物",
        "城墙",
        "寺",
        "庙",
        "宫",
        "书院",
        "园林",
    ),
    "nature": (
        "自然",
        "山水",
        "湖",
        "山",
        "森林",
        "湿地",
        "公园",
        "海",
        "瀑布",
    ),
    "art": ("艺术", "美术馆", "画廊", "剧院", "文艺", "创意"),
    "family": ("亲子", "儿童", "乐园", "欢乐谷", "动物园", "海洋馆", "科技馆"),
    "shopping": ("购物", "商场", "步行街", "商业街", "市集"),
}


def _rank_poi_candidates(
    items: list[dict[str, Any]], keywords: list[str] | None
) -> list[dict[str, Any]]:
    """Apply deterministic preference relevance without narrowing supply.

    Provider/local popularity remains part of the score.  Preference relevance
    only reorders the broad candidate pool, so a weak or unusual preference can
    never make the downstream solver lose all feasible choices.
    """
    normalized_keywords = [
        _normalize_poi_name(keyword) for keyword in (keywords or []) if str(keyword).strip()
    ]
    ranked: list[dict[str, Any]] = []
    for original in items:
        item = dict(original)
        base = max(0.0, min(1.0, float(item.get("score") or 0.5)))
        category = str(item.get("category") or "attraction").lower()
        if not normalized_keywords or category != "attraction":
            item["score"] = round(base, 3)
            ranked.append(item)
            continue

        haystack = _normalize_poi_name(
            " ".join(
                [
                    str(item.get("name") or ""),
                    str(item.get("description") or ""),
                    *[str(tag) for tag in (item.get("tags") or [])],
                ]
            )
        )
        direct = any(keyword in haystack for keyword in normalized_keywords)
        concept_hits = 0
        for terms in _ATTRACTION_PREFERENCE_TERMS.values():
            normalized_terms = tuple(_normalize_poi_name(term) for term in terms)
            if any(term in keyword for term in normalized_terms for keyword in normalized_keywords):
                if any(term in haystack for term in normalized_terms):
                    concept_hits += 1
        relevance = min(1.0, (0.55 if direct else 0.0) + 0.45 * concept_hits)
        item["score"] = round(min(1.0, 0.45 * base + 0.35 + 0.25 * relevance), 3)
        item["preference_relevance"] = round(relevance, 3)
        ranked.append(item)

    return sorted(
        ranked,
        key=lambda item: (
            float(item.get("score") or 0),
            float(item.get("preference_relevance") or 0),
            str(item.get("name") or ""),
        ),
        reverse=True,
    )


def _has_plannable_poi(items: list[dict[str, Any]]) -> bool:
    return any(
        str(item.get("category") or "attraction").lower()
        not in {"restaurant", "meal", "hotel", "transport"}
        for item in items
        if isinstance(item, dict)
    )


def _has_requested_poi_supply(items: list[dict[str, Any]], category: str | None) -> bool:
    if category is None:
        return _has_plannable_poi(items)
    requested = category.lower()
    return any(
        str(item.get("category") or "attraction").lower() == requested
        for item in items
        if isinstance(item, dict)
    )


def _merge_poi_candidates(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*first, *second]:
        key = str(item.get("id") or _normalize_poi_name(item.get("name")))
        if key and key not in seen:
            seen.add(key)
            merged.append(item)
    return merged


class ToolExecutor:
    """Execute a batch of OpenAI-format tool_calls.

    Usage:
        results = await ToolExecutor().execute(tool_calls)
    """

    def __init__(self) -> None:
        from skills.poi_search import POISearchSkill
        from skills.weather_query import WeatherQuerySkill

        self._handlers: dict[str, Handler] = {
            "get_weather": self._handle_get_weather,
            "check_reservation": self._handle_check_reservation,
            "get_route": self._handle_get_route,
            "find_restaurants": self._handle_find_restaurants,
            "find_hotels": self._handle_find_hotels,
            "get_queue_time": self._handle_get_queue_time,
            "get_ticket_link": self._handle_get_ticket_link,
            "retrieve_city_knowledge": self._handle_retrieve_city_knowledge,
            "search_current_info": self._handle_search_current_info,
            "search_transport": self._handle_search_transport,
            "get_emergency_services": self._handle_get_emergency_services,
            "get_poi_detail": self._handle_get_poi_detail,
            "update_user_profile": self._handle_update_user_profile,
            "search_pois": self._handle_search_pois,
            "get_route_matrix": self._handle_get_route_matrix,
            "solve_itinerary": self._handle_solve_itinerary,
            "validate_itinerary": self._handle_validate_itinerary,
        }
        self._weather = WeatherQuerySkill()
        self._poi = POISearchSkill()
        self._router = MapServiceRouter()
        self._guard = ToolGuard(
            mode=settings.agentic_guard_mode,
            max_calls=settings.agentic_tool_call_budget,
        )
        self._tool_timeout_seconds = float(settings.agentic_tool_timeout_seconds)

    @property
    def available_tools(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function schemas."""
        return list(TOOL_NAME_TO_SCHEMA.values())

    async def execute(
        self,
        tool_calls: list[dict[str, Any]],
        guard_context: GuardContext | dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute each tool call and return results keyed by tool_call_id."""
        results: list[dict[str, Any]] = []
        decisions = self._guard.evaluate_batch(tool_calls, guard_context)
        for call, guard_decision in zip(tool_calls, decisions):
            started = time.monotonic()
            tool_call_id = call.get("id", "")
            function = call.get("function", {}) or {}
            name = function.get("name", "")
            raw_args = function.get("arguments", "{}")

            if not guard_decision.allowed:
                first = guard_decision.violations[0]
                result = ToolResult(
                    data=None,
                    data_source="unavailable",
                    fallback_reason=first.message,
                )
                observation = ObservationEnvelope.failure(
                    tool=name,
                    code=first.code,
                    message=first.message,
                    retryable=False,
                    tool_call_id=tool_call_id,
                    details={
                        "guard_mode": guard_decision.mode,
                        "violations": [v.model_dump() for v in guard_decision.violations],
                    },
                )
                results.append(
                    self._result_record(tool_call_id, name, result, observation, guard_decision)
                )
                continue
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError as exc:
                logger.warning("Invalid tool arguments for %s: %s", name, exc)
                result = ToolResult(
                    data=None,
                    data_source="unavailable",
                    fallback_reason="tool arguments are not valid JSON",
                )
                observation = ObservationEnvelope.failure(
                    tool=name,
                    code="INVALID_ARGUMENTS",
                    message="tool arguments are not valid JSON",
                    retryable=False,
                    tool_call_id=tool_call_id,
                    details={"error_type": type(exc).__name__},
                )
                results.append(
                    self._result_record(tool_call_id, name, result, observation, guard_decision)
                )
                continue

            handler = self._handlers.get(name)
            if handler is None:
                result = ToolResult(
                    data=None,
                    data_source="unavailable",
                    is_fallback=True,
                    fallback_reason=f"unknown tool: {name}",
                )
                observation = ObservationEnvelope.failure(
                    tool=name,
                    code="UNKNOWN_TOOL",
                    message=f"unknown tool: {name}",
                    retryable=False,
                    tool_call_id=tool_call_id,
                )
            else:
                try:
                    async with asyncio.timeout(self._tool_timeout_seconds):
                        result = await handler(args)
                    if result.latency_ms == 0:
                        result.latency_ms = int((time.monotonic() - started) * 1000)
                    observation = ObservationEnvelope.from_tool_result(
                        tool=name,
                        result=result,
                        tool_call_id=tool_call_id,
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("Tool %s failed: %s", name, exc)
                    error_code, retryable, error_details = _classify_tool_exception(exc)
                    result = ToolResult(
                        data=None,
                        data_source="unavailable",
                        is_fallback=True,
                        fallback_reason=f"{type(exc).__name__}: {exc}",
                    )
                    observation = ObservationEnvelope.failure(
                        tool=name,
                        code=error_code,
                        message=str(exc),
                        retryable=retryable,
                        tool_call_id=tool_call_id,
                        latency_ms=int((time.monotonic() - started) * 1000),
                        details=error_details,
                    )

            results.append(
                self._result_record(tool_call_id, name, result, observation, guard_decision)
            )
        return results

    @staticmethod
    def _result_record(
        tool_call_id: str,
        name: str,
        result: ToolResult,
        observation: ObservationEnvelope,
        guard_decision: GuardDecision,
    ) -> dict[str, Any]:
        """Keep the legacy result during migration and add the shared contract."""
        return {
            "tool_call_id": tool_call_id,
            "name": name,
            "result": result.model_dump(),
            "observation": observation.model_dump(),
            "guard": guard_decision.model_dump(),
        }

    async def _handle_get_weather(self, args: dict[str, Any]) -> ToolResult:
        city = args.get("city", "")
        date = args.get("date") or datetime.now().strftime("%Y-%m-%d")
        try:
            parsed = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            parsed = datetime.now()
        end = (parsed + timedelta(days=2)).strftime("%Y-%m-%d")
        days = await self._weather.query(city, date, end)
        return ToolResult(
            data=[d.model_dump() for d in days],
            data_source="fallback" if any(d.is_fallback for d in days) else "api",
            confidence=0.8,
            is_fallback=any(d.is_fallback for d in days),
            fallback_reason=(
                "weather provider unavailable; using estimated conditions"
                if any(d.is_fallback for d in days)
                else None
            ),
        )

    async def _handle_check_reservation(self, args: dict[str, Any]) -> ToolResult:
        poi = args.get("poi_name", "")
        city = args.get("city") or "未知城市"
        need_reserve = any(
            keyword in poi
            for keyword in ("故宫", "国博", "国家博物馆", "兵马俑", "莫高窟", "迪士尼", "环球")
        )
        return ToolResult(
            data={
                "poi": poi,
                "city": city,
                "need_reserve": need_reserve,
                "notice": "建议提前 1-7 天预约" if need_reserve else "通常可现场购票",
                "channels": ["官方小程序", "携程", "美团"] if need_reserve else ["现场窗口", "OTA"],
            },
            data_source="built_in",
            confidence=0.7,
        )

    async def _handle_get_route(self, args: dict[str, Any]) -> ToolResult:
        origin = args.get("origin", "")
        destination = args.get("destination", "")
        mode = args.get("mode", "transit")
        fallback = HaversineFallback()
        # Mock coordinates based on hash for deterministic tests.
        olat, olng = self._coords_from_name(origin)
        dlat, dlng = self._coords_from_name(destination)
        minutes, cost = fallback.estimate(olat, olng, dlat, dlng, mode)
        return ToolResult(
            data={
                "origin": origin,
                "destination": destination,
                "mode": mode,
                "minutes": minutes,
                "cost_cny": cost,
                "polyline": None,
            },
            data_source="fallback",
            confidence=0.6,
            is_fallback=True,
            fallback_reason="using haversine estimation; real routing needs coordinates",
        )

    async def _handle_find_restaurants(self, args: dict[str, Any]) -> ToolResult:
        city = args.get("city", "")
        area = args.get("area")
        cuisine = args.get("cuisine")
        budget = args.get("budget_per_person")
        candidates = await self._poi.search_pois(
            city=city,
            keywords=[area, cuisine, "餐厅"] if (area or cuisine) else ["餐厅"],
            category="restaurant",
        )
        results = []
        for p in candidates[:5]:
            price = budget or random.randint(60, 200)
            results.append(
                {
                    "name": p.name,
                    "category": cuisine or "本地菜",
                    "area": area or "市中心",
                    "price_per_person": price,
                    "rating": p.score,
                    "tags": p.tags or [],
                }
            )
        if not results:
            results = [{"name": f"{city}推荐餐厅", "category": "本地菜", "price_per_person": 100}]
        return ToolResult(data=results, data_source="built_in", confidence=0.7)

    async def _handle_find_hotels(self, args: dict[str, Any]) -> ToolResult:
        city = args.get("city", "")
        area = args.get("area") or "市中心"
        budget = args.get("budget_per_night")
        candidates = await self._poi.search_pois(
            city=city,
            keywords=[area, "酒店"],
            category="hotel",
        )
        results = []
        for p in candidates[:5]:
            price = budget or random.randint(300, 800)
            results.append(
                {
                    "name": p.name,
                    "area": area,
                    "price_per_night": price,
                    "rating": p.score,
                    "tags": p.tags or [],
                }
            )
        if not results:
            results = [
                {"name": f"{city}{area}酒店", "area": area, "price_per_night": budget or 500}
            ]
        return ToolResult(data=results, data_source="built_in", confidence=0.7)

    async def _handle_get_queue_time(self, args: dict[str, Any]) -> ToolResult:
        poi = args.get("poi_name", "")
        return ToolResult(
            data={
                "poi": poi,
                "current_minutes": random.randint(5, 90),
                "status": random.choice(["畅通", "适中", "拥挤"]),
                "best_time": "上午 9:00 前",
            },
            data_source="fallback",
            confidence=0.5,
            is_fallback=True,
            fallback_reason="queue time API not configured",
        )

    async def _handle_get_ticket_link(self, args: dict[str, Any]) -> ToolResult:
        poi = args.get("poi_name", "")
        return ToolResult(
            data={
                "poi": poi,
                "official_url": f"https://example.com/ticket/{poi}",
                "ota_urls": ["https://www.ctrip.com", "https://www.meituan.com"],
                "tip": "请以官方渠道为准",
            },
            data_source="fallback",
            confidence=0.5,
            is_fallback=True,
            fallback_reason="ticket link API not configured",
        )

    async def _handle_retrieve_city_knowledge(self, args: dict[str, Any]) -> ToolResult:
        """Read stable city/POI facts from the canonical local knowledge store."""
        city = canonical_city_name(args.get("city", ""))
        topic = str(args.get("topic") or "").strip()
        items = await self._search_local_pois(city, category="attraction", limit=20)
        if not items:
            try:
                from skills.city_data import CITY_DEFAULTS

                items = [
                    item.model_dump(mode="json")
                    for item in (CITY_DEFAULTS.get(city) or [])
                    if item.category == "attraction"
                ]
            except Exception as exc:
                logger.warning("Built-in city knowledge failed for %s: %s", city, exc)
        if topic:
            ranked = _rank_poi_candidates(items, [topic])
        else:
            ranked = items
        if not ranked:
            # An empty local index is a valid knowledge-base observation, not
            # a transient tool outage.  The ReAct loop can continue with
            # grounded POI/live-search tools while preserving this explicit
            # coverage gap in the research bundle.
            return ToolResult(
                data={
                    "city": city,
                    "topic": topic or None,
                    "pois": [],
                    "record_count": 0,
                    "availability": "not_indexed",
                    "retrieved_at": datetime.now(UTC).isoformat(),
                },
                data_source="built_in",
                confidence=1.0,
            )
        return ToolResult(
            data={
                "city": city,
                "topic": topic or None,
                "pois": ranked[:12],
                "record_count": len(ranked),
                "retrieved_at": datetime.now(UTC).isoformat(),
            },
            data_source="built_in",
            confidence=0.95,
        )

    @staticmethod
    async def _search_web_evidence(query: str, *, top_n: int = 6) -> list[dict[str, Any]]:
        from skills.tavily_search import UnifiedSearchSkill

        results = await UnifiedSearchSkill().search(query, top_n=top_n)
        evidence: list[dict[str, Any]] = []
        for item in results:
            url = _normalize_external_url(item.url)
            title, title_flags = _normalize_external_text(item.title, max_length=240)
            snippet, snippet_flags = _normalize_external_text(item.snippet, max_length=800)
            if not url or not (title or snippet):
                continue
            evidence.append(
                {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "score": float(item.score or 0),
                    "trust_tier": "untrusted_external",
                    "security_flags": list(dict.fromkeys([*title_flags, *snippet_flags])),
                }
            )
        return evidence

    async def _handle_search_current_info(self, args: dict[str, Any]) -> ToolResult:
        city = canonical_city_name(args.get("city") or "")
        date = str(args.get("date") or "").strip()
        info_type = str(args.get("info_type") or "general")
        raw_query = str(args.get("query") or "").strip()
        search_suffix = "时间 地点 场馆 官方" if info_type == "event" else "官方 最新"
        query = " ".join(part for part in (city, raw_query, date, search_suffix) if part)
        results = await self._search_web_evidence(query, top_n=8 if info_type == "event" else 6)
        if not results:
            return ToolResult(
                data={
                    "query": query,
                    "info_type": info_type,
                    "date": date or None,
                    "results": [],
                    "availability": "no_results",
                    "queried_at": datetime.now(UTC).isoformat(),
                },
                data_source="api",
                confidence=1.0,
            )
        payload: dict[str, Any] = {
            "query": query,
            "info_type": info_type,
            "date": date or None,
            "results": results,
            "queried_at": datetime.now(UTC).isoformat(),
        }
        if info_type != "event":
            return ToolResult(data=payload, data_source="api", confidence=0.75)

        corpus = "\n".join(f"{item['title']} {item['snippet']}" for item in results[:5])
        date_match = re.search(r"20\d{2}[-年/.]\d{1,2}[-月/.]\d{1,2}日?", corpus)
        time_matches = re.findall(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d", corpus)
        venue_match = re.search(
            r"([\u4e00-\u9fa5A-Za-z0-9·（）()]{2,30}(?:体育场|体育馆|剧院|音乐厅|中心|场馆))",
            corpus,
        )
        venue = venue_match.group(1) if venue_match else None
        if venue:
            venue = re.sub(r"^(?:演出)?(?:地点|场馆|地址)[:：]?", "", venue)
        structured = {
            "name": raw_query,
            "city": city or None,
            "date": date_match.group(0) if date_match else (date or None),
            "start_time": time_matches[0] if time_matches else None,
            "end_time": time_matches[1] if len(time_matches) > 1 else None,
            "venue": venue,
        }
        if structured["venue"] and city and settings.amap_key:
            from data.collectors.amap import AmapCollector

            collector = AmapCollector(settings.amap_key)
            try:
                venue_items = await collector.search_pois(
                    city,
                    keywords=str(structured["venue"]),
                    types="",
                    limit=3,
                )
                if venue_items:
                    structured["venue"] = venue_items[0].name
                    structured["lat"] = venue_items[0].lat
                    structured["lng"] = venue_items[0].lng
            except Exception as exc:
                logger.warning("AMap event venue grounding failed for %s: %s", raw_query, exc)
            finally:
                await collector.close()
        required = ("date", "start_time", "venue")
        structured["complete"] = all(structured.get(key) for key in required)
        payload["event"] = structured
        return ToolResult(
            data=payload,
            data_source="api",
            confidence=0.8 if structured["complete"] else 0.65,
        )

    async def _handle_search_transport(self, args: dict[str, Any]) -> ToolResult:
        origin = canonical_city_name(args.get("origin") or "")
        destination = canonical_city_name(args.get("destination") or "")
        date = str(args.get("date") or "").strip()
        return_date = str(args.get("return_date") or "").strip()
        mode = str(args.get("mode") or "both")
        mode_text = {"flight": "航班 机票", "train": "火车 高铁 车次"}.get(mode, "航班 火车 高铁")
        leg_queries = [
            ("inbound", origin, destination, date),
        ]
        if return_date:
            leg_queries.append(("outbound", destination, origin, return_date))
        legs: list[dict[str, Any]] = []
        all_results: list[dict[str, Any]] = []
        for direction, leg_origin, leg_destination, leg_date in leg_queries:
            query = (
                f"{leg_date} {leg_origin} 到 {leg_destination} {mode_text} 时刻表 官方"
            ).strip()
            results = await self._search_web_evidence(query, top_n=8)
            all_results.extend(results)
            options = _extract_transport_options(results, mode=mode)
            legs.append(
                {
                    "direction": direction,
                    "origin": leg_origin,
                    "destination": leg_destination,
                    "date": leg_date or None,
                    "query": query,
                    "options": options,
                    "selected_option": options[0] if options else None,
                }
            )
        if not all_results:
            return ToolResult(
                data={
                    "origin": origin,
                    "destination": destination,
                    "date": date or None,
                    "return_date": return_date or None,
                    "mode": mode,
                    "legs": legs,
                    "results": [],
                    "availability": "no_results",
                    "queried_at": datetime.now(UTC).isoformat(),
                },
                data_source="api",
                confidence=1.0,
            )
        return ToolResult(
            data={
                "origin": origin,
                "destination": destination,
                "date": date or None,
                "return_date": return_date or None,
                "mode": mode,
                "legs": legs,
                "results": all_results,
                "queried_at": datetime.now(UTC).isoformat(),
                "notice": "班次、票价和余票具有时效性，最终以承运方或官方售票平台为准。",
            },
            data_source="api",
            confidence=0.8 if all(leg["selected_option"] for leg in legs) else 0.6,
        )

    async def _handle_get_emergency_services(self, args: dict[str, Any]) -> ToolResult:
        city = args.get("city", "")
        return ToolResult(
            data={
                "hospitals": [f"{city}第一人民医院", f"{city}中心医院"],
                "police": f"{city}公安局",
                "consulate": "请查询外交部领事司官网",
                "emergency_number": "120/110",
            },
            data_source="built_in",
            confidence=0.8,
        )

    async def _handle_get_poi_detail(self, args: dict[str, Any]) -> ToolResult:
        poi = args.get("poi_name", "")
        city = args.get("city") or ""
        local_candidates = await self._search_local_pois(city, category="attraction")
        local_match = next(
            (
                item
                for item in local_candidates
                if _normalize_poi_name(item.get("name")) == _normalize_poi_name(poi)
            ),
            None,
        )
        if local_match is not None:
            return ToolResult(
                data={
                    "name": local_match["name"],
                    "city": city,
                    "open_hours": (f"{local_match['open_time']}-{local_match['close_time']}"),
                    "ticket_price": local_match.get("ticket_price"),
                    "suggested_hours": round(
                        float(local_match.get("duration_minutes") or 120) / 60, 2
                    ),
                    "tags": local_match.get("tags") or [],
                    "description": local_match.get("description") or "",
                },
                data_source="built_in",
                confidence=0.95,
            )
        candidates = await self._poi.search_pois(
            city=city,
            keywords=[poi],
            category="attraction",
        )
        requested_identity = _normalize_poi_name(poi)
        best = next(
            (
                candidate
                for candidate in candidates
                if _normalize_poi_name(getattr(candidate, "name", "")) == requested_identity
            ),
            None,
        )
        if best is not None:
            # Use actual POI data -- no random values
            open_time = getattr(best, "open_time", None)
            close_time = getattr(best, "close_time", None)
            if open_time and close_time:
                open_hours = open_time + "-" + close_time
            elif open_time:
                open_hours = open_time
            else:
                open_hours = "09:00-17:00"

            # Map recommended_hours string to a float
            hours_map = {
                "1小时": 1.0,
                "1.5小时": 1.5,
                "1-2小时": 1.5,
                "2小时": 2.0,
                "2-3小时": 2.5,
                "3小时": 3.0,
                "3-4小时": 3.5,
                "半天": 4.0,
                "全天": 8.0,
            }
            raw_hours = getattr(best, "recommended_hours", None)
            suggested_hours = hours_map.get(raw_hours or "", 2.0)

            return ToolResult(
                data={
                    "name": best.name,
                    "city": city,
                    "open_hours": open_hours,
                    "ticket_price": best.ticket_price,
                    "suggested_hours": suggested_hours,
                    "tags": best.tags or [],
                    "description": best.description or "",
                },
                data_source="built_in",
                confidence=0.7,
            )
        return ToolResult(
            data={
                "name": poi,
                "city": city,
                "open_hours": "09:00-17:00",
                "ticket_price": None,
                "suggested_hours": 2.0,
                "tags": [],
                "description": "",
            },
            data_source="fallback",
            confidence=0.5,
            is_fallback=True,
            fallback_reason="POI detail not found",
        )

    async def _handle_update_user_profile(self, args: dict[str, Any]) -> ToolResult:
        key = args.get("key", "")
        value = args.get("value")
        return ToolResult(
            data={"updated": {key: value}},
            data_source="built_in",
            confidence=1.0,
        )

    async def _handle_search_pois(self, args: dict[str, Any]) -> ToolResult:
        keywords = args.get("keywords") or []
        city = canonical_city_name(args["city"])
        category = args.get("category")
        candidates = await self._search_local_pois(city, category=category)
        if _has_requested_poi_supply(candidates, category):
            return ToolResult(
                data=_rank_poi_candidates(candidates, keywords),
                data_source="built_in",
                confidence=0.95,
            )

        # A mixed Agent search must always contain something the itinerary
        # solver can schedule. Providers occasionally ignore the requested
        # type and return restaurants for an attraction query, so validate the
        # normalized category before treating a non-empty payload as success.
        supply_category = category or "attraction"
        amap = await self._search_amap_pois(
            city,
            keywords=keywords,
            category=supply_category,
        )
        candidates = _merge_poi_candidates(candidates, amap)
        if _has_requested_poi_supply(candidates, category):
            return ToolResult(
                data=_rank_poi_candidates(candidates, keywords),
                data_source="api",
                confidence=0.9,
            )

        fallback_supply = await self._poi.run(
            {
                "city": city,
                "keywords": keywords,
                "category": supply_category,
            }
        )
        fallback_items = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in (fallback_supply.data or [])
            if isinstance(item, dict) or hasattr(item, "model_dump")
        ]
        candidates = _merge_poi_candidates(candidates, fallback_items)
        if _has_requested_poi_supply(candidates, category):
            return ToolResult(
                data=_rank_poi_candidates(candidates, keywords),
                data_source=fallback_supply.data_source,
                confidence=fallback_supply.confidence,
                is_fallback=fallback_supply.is_fallback,
                fallback_reason=fallback_supply.fallback_reason,
            )
        return ToolResult(
            data=[],
            data_source="unavailable",
            confidence=0.0,
            is_fallback=True,
            fallback_reason="no POIs matched the required planning category",
        )

    @staticmethod
    async def _search_local_pois(
        city: str, *, category: str | None = None, limit: int = 30
    ) -> list[dict[str, Any]]:
        """Read the application's canonical PostgreSQL POI stores first.

        The API/external search skill remains a degradation path for cities that
        are not yet present locally.  Keeping this conversion here gives Agent
        tools the same grounded entities used by the deterministic workflow.
        """
        if not city:
            return []
        results: list[dict[str, Any]] = []
        if category in (None, "attraction"):
            try:
                from data.retrieval_repository import retrieval_repo

                attractions = await retrieval_repo.search_structured(city, limit=limit)
                results.extend(
                    {
                        "id": item.spot_id,
                        "name": item.spot_name,
                        "category": "attraction",
                        "score": round(float(item.rating or 4) / 5, 3),
                        "location": {"lat": item.lat, "lng": item.lng},
                        "lat": item.lat,
                        "lng": item.lng,
                        "description": item.description or "",
                        "tags": item.tags,
                        "ticket_price": item.ticket_price,
                        "open_time": item.open_time,
                        "close_time": item.close_time,
                        "duration_minutes": item.duration_minutes,
                        "recommended_hours": str(item.duration_minutes / 60),
                        "data_source": "built_in",
                        "confidence": 0.95,
                        "is_fallback": False,
                    }
                    for item in attractions
                )
            except Exception as exc:
                logger.warning("Local attraction search failed for %s: %s", city, exc)
        if category in (None, "restaurant"):
            try:
                from data.repository import repo

                restaurants = await repo.search_restaurants(city, limit=limit)
                results.extend(
                    {
                        "id": str(item.get("id") or item.get("name")),
                        "name": str(item.get("name") or ""),
                        "category": "restaurant",
                        "score": round(float(item.get("rating") or 4) / 5, 3),
                        "location": {
                            "lat": float(item.get("lat") or 0),
                            "lng": float(item.get("lng") or 0),
                        },
                        "lat": float(item.get("lat") or 0),
                        "lng": float(item.get("lng") or 0),
                        "tags": list(item.get("tags") or []),
                        "ticket_price": float(item.get("avg_price") or 0),
                        "open_time": str(item.get("open_time") or "10:00")[:5],
                        "close_time": str(item.get("close_time") or "22:00")[:5],
                        "duration_minutes": 90,
                        "recommended_hours": "1.5",
                        "data_source": "built_in",
                        "confidence": 0.95,
                        "is_fallback": False,
                    }
                    for item in restaurants
                    if item.get("name")
                )
            except Exception as exc:
                logger.warning("Local restaurant search failed for %s: %s", city, exc)
        return results

    @staticmethod
    async def _search_amap_pois(
        city: str, *, keywords: list[str], category: str | None = None
    ) -> list[dict[str, Any]]:
        """Use the configured map provider when the canonical store misses."""
        if not city or not settings.amap_key:
            return []
        from data.collectors.amap import AmapCollector

        type_map = {
            "attraction": "风景名胜",
            "restaurant": "餐饮服务",
            "hotel": "住宿服务",
            "shopping": "购物服务",
        }
        collector = AmapCollector(settings.amap_key)
        try:
            if category:
                raw_items = await collector.search_pois(
                    city,
                    keywords=(
                        " ".join(str(item) for item in keywords[:3])
                        if category != "attraction"
                        else ""
                    ),
                    types=type_map[category],
                    limit=30,
                )
            else:
                attractions, restaurants = await asyncio.gather(
                    collector.search_pois(
                        city, keywords="", types=type_map["attraction"], limit=20
                    ),
                    collector.search_pois(
                        city,
                        keywords=" ".join(str(item) for item in keywords[:3]),
                        types=type_map["restaurant"],
                        limit=10,
                    ),
                )
                raw_items = [*attractions, *restaurants]
        except Exception as exc:
            logger.warning("AMap POI search failed for %s: %s", city, exc)
            return []
        finally:
            await collector.close()
        return [
            {
                "id": f"amap:{item.category}:{index}:{item.name}",
                "name": item.name,
                "category": item.category,
                "score": 0.7,
                "location": {"lat": item.lat, "lng": item.lng},
                "lat": item.lat,
                "lng": item.lng,
                "tags": item.tags,
                "ticket_price": float(item.ticket_price or 0),
                "open_time": item.open_time or "08:00",
                "close_time": item.close_time or "18:00",
                "duration_minutes": 120 if item.category == "attraction" else 90,
                "recommended_hours": "2" if item.category == "attraction" else "1.5",
                "data_source": "api",
                "confidence": 0.9,
                "is_fallback": False,
            }
            for index, item in enumerate(raw_items[:30])
        ]

    async def _handle_get_route_matrix(self, args: dict[str, Any]) -> ToolResult:
        from planner.preprocessing.transport_selector import TransportSelector
        from vrp_solver_service.models import ConstraintsInput, POIInput

        pois = [POIInput(**item) for item in args["pois"]]
        # TravelVRPSolver injects a virtual hotel before consuming supplied
        # matrices. Build the artifact with the same leading node so its shape
        # and indexes remain identical at solve time.
        matrix_pois = [
            POIInput(
                id="__hotel",
                name="Hotel",
                category="hotel",
                duration_minutes=0,
                open_time="00:00",
                close_time="23:59",
                walk_intensity=0,
            ),
            *pois,
        ]
        constraints = ConstraintsInput(**(args.get("constraints") or {}))
        dist, costs = TransportSelector().build_matrices(
            matrix_pois, constraints, args.get("amap_minutes")
        )
        return ToolResult(
            data={
                "poi_ids": [poi.id for poi in matrix_pois],
                "time_minutes": dist,
                "transport_cost": costs,
            },
            data_source="built_in",
            confidence=1.0,
        )

    async def _handle_solve_itinerary(self, args: dict[str, Any]) -> ToolResult:
        from vrp_solver_service.models import SolverRequest
        from vrp_solver_service.solver import TravelVRPSolver

        request = SolverRequest(**args)
        response = await asyncio.to_thread(TravelVRPSolver().solve, request)
        usable = response.status not in {"infeasible", "error"} and bool(response.days)
        return ToolResult(
            data=response.model_dump(),
            data_source="built_in" if usable else "unavailable",
            confidence=1.0 if usable else 0.0,
            is_fallback=response.status == "fallback",
            fallback_reason=response.message if not usable else None,
            latency_ms=response.solve_time_ms,
        )

    async def _handle_validate_itinerary(self, args: dict[str, Any]) -> ToolResult:
        report = ItineraryValidator().validate(
            args.get("itinerary") or [],
            constraints=args.get("constraints") or {},
            facts=args.get("facts") or [],
        )
        return ToolResult(
            data=report.model_dump(),
            data_source="built_in",
            confidence=1.0,
        )

    @staticmethod
    def _coords_from_name(name: str) -> tuple[float, float]:
        """Return deterministic pseudo-coordinates for a place name."""
        h = hash(name) % 10000
        return 30.0 + (h % 100) / 100.0, 110.0 + (h // 100) / 100.0


# Lazy singleton — avoid import-time init (circular import with skills.poi_search).
_executor: ToolExecutor | None = None


def get_tool_executor() -> ToolExecutor:
    global _executor
    if _executor is None:
        _executor = ToolExecutor()
    return _executor


class _ToolExecutorProxy:
    """Backward-compatible lazy proxy for ``tool_executor.execute(...)``."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_tool_executor(), name)


tool_executor = _ToolExecutorProxy()

__all__ = ["ToolExecutor", "tool_executor", "get_tool_executor"]
