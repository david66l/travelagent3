"""Async node implementations shared by the LangGraph orchestration layer.

These helpers were originally part of backend/src/agent/graph.py. They are kept
here as pure async implementations so the old graph can be removed while the new
graph in backend/src/graph continues to use them.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.langsmith_trace import traceable_step

logger = logging.getLogger(__name__)

# Transport/scenery hubs that get miscategorised as dining (e.g. a ferry pier).
# Used to keep them out of the meal-naming pool so a lunch is never labelled with
# a "码头" or "车站".
_NON_DINING_KW = (
    "码头",
    "游船",
    "游轮",
    "邮轮",
    "轮渡",
    "渡口",
    "客运",
    "车站",
    "地铁站",
    "机场",
    "停车",
    "口岸",
    "缆车",
    "索道",
)


def _is_dining_venue(name: str | None) -> bool:
    return not any(kw in (name or "") for kw in _NON_DINING_KW)


# ---------------------------------------------------------------------------
# ① DemandParserAgent — 需求解析
# ---------------------------------------------------------------------------


@traceable_step("planning/weather_check", run_type="chain")
async def _weather_check_async(state: dict) -> dict:
    """Fetch weather for destination + dates BEFORE planning.

    Uses the same AMap API / Open-Meteo / geography fallback chain
    as WeatherQuerySkill, but imported here to avoid the circular
    import between skills.weather_query → tools.base → tool_executor.
    """
    import asyncio as aio
    from datetime import datetime

    slots = state.get("slots") or {}
    destination = slots.get("destination") or ""
    travel_dates = slots.get("travel_dates") or ""
    travel_days = slots.get("travel_days") or 1

    if not destination:
        return {"weather": [], "weather_fetched": False}

    today = datetime.now().strftime("%Y-%m-%d")
    try:
        from datetime import timedelta

        end_date = (datetime.now() + timedelta(days=travel_days + 1)).strftime("%Y-%m-%d")
    except Exception:
        end_date = today

    start_date = today
    if travel_dates:
        parts = travel_dates.replace(" to ", "|").replace("~", "|").split("|")
        try:
            d = datetime.strptime(parts[0].strip()[:10], "%Y-%m-%d")
            start_date = d.strftime("%Y-%m-%d")
        except (ValueError, IndexError):
            pass

    # Try AMap weather first (free, already have the key)
    from core.settings import settings as _s

    weather: list[dict] = []

    if _s.amap_key:
        try:
            weather = await aio.wait_for(
                _fetch_amap_weather(destination, start_date, end_date, _s.amap_key),
                timeout=5.0,
            )
        except Exception:
            pass

    if not weather:
        # Fallback: geography estimation
        from skills.weather_query import (
            CITY_COORDS,
            _estimate_temp,
            _estimate_condition,
            _estimate_precip,
        )
        import random

        coords = CITY_COORDS.get(destination, CITY_COORDS.get(destination.rstrip("市省")))
        lat = coords[0] if coords else 30.0
        try:
            s = datetime.strptime(start_date, "%Y-%m-%d")
            e = datetime.strptime(end_date, "%Y-%m-%d")
            cur = s
            while cur <= e:
                high, low = _estimate_temp(lat, cur.month)
                high += random.randint(-3, 3)
                low += random.randint(-2, 2)
                weather.append(
                    {
                        "date": cur.strftime("%Y-%m-%d"),
                        "condition": _estimate_condition(lat, cur.month),
                        "temp_high": high,
                        "temp_low": low,
                        "precipitation_chance": _estimate_precip(lat, cur.month),
                        "recommendation": "（估算数据）",
                        "data_source": "fallback",
                        "is_fallback": True,
                    }
                )
                cur += timedelta(days=1)
        except ValueError:
            pass

    return {
        "weather": weather,
        "weather_fetched": len(weather) > 0,
        "weather_start": start_date,
        "weather_end": end_date,
    }


async def _fetch_amap_weather(
    destination: str, start_date: str, end_date: str, key: str
) -> list[dict]:
    """Fetch weather from AMap API for weather_check_node."""
    from datetime import datetime
    import httpx

    # City → adcode mapping (subset of weather_query's _CITY_ADCODE)
    ADCODE: dict[str, str] = {
        "北京": "110000",
        "上海": "310000",
        "广州": "440100",
        "深圳": "440300",
        "成都": "510100",
        "杭州": "330100",
        "西安": "610100",
        "重庆": "500000",
        "苏州": "320500",
        "南京": "320100",
        "厦门": "350200",
        "青岛": "370200",
        "大理": "532901",
        "丽江": "530700",
        "三亚": "460200",
        "长沙": "430100",
        "武汉": "420100",
        "昆明": "530100",
        "桂林": "450300",
        "拉萨": "540100",
        "济南": "370100",
        "郑州": "410100",
        "天津": "120000",
        "合肥": "340100",
        "哈尔滨": "230100",
        "长春": "220100",
        "沈阳": "210100",
    }
    adcode = ADCODE.get(destination, ADCODE.get(destination.rstrip("市省"), ""))
    if not adcode:
        return []

    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(url, params={"key": key, "city": adcode, "extensions": "all"})
        resp.raise_for_status()
        data = resp.json()

    if data.get("status") != "1":
        return []

    try:
        s = datetime.strptime(start_date, "%Y-%m-%d")
        e = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        return []

    results: list[dict] = []
    for fc in data.get("forecasts", []):
        for cast in fc.get("casts", []):
            try:
                d = datetime.strptime(cast["date"], "%Y-%m-%d")
            except (ValueError, KeyError):
                continue
            if s <= d <= e:
                results.append(
                    {
                        "date": cast["date"],
                        "condition": cast.get("dayweather", "多云"),
                        "temp_high": int(float(cast.get("daytemp_float", cast.get("daytemp", 25)))),
                        "temp_low": int(
                            float(cast.get("nighttemp_float", cast.get("nighttemp", 15)))
                        ),
                        "precipitation_chance": 70 if "雨" in cast.get("dayweather", "") else 10,
                        "data_source": "api",
                        "is_fallback": False,
                    }
                )
    return results


# ---------------------------------------------------------------------------
# ② UserMemoryRecallAgent — 用户画像记忆
# ---------------------------------------------------------------------------


@traceable_step("planning/user_memory", run_type="chain")
async def _user_memory_async(state: dict) -> dict:
    profile = state.get("profile") or {}
    user_id = state.get("user_id", "")
    stage = state.get("stage", "")

    # 行程结束 → 写入记忆
    if stage == "completed" and user_id and user_id != "anonymous":
        try:
            from data.profile_service import profile_service

            itinerary = state.get("itinerary", [])
            destination = profile.get("destination", "")
            visited = [destination] if destination else []
            await profile_service.update_profile(
                user_id,
                visited_cities=visited,
                trip_budget=sum(d.get("total_cost", 0) for d in itinerary) / max(len(itinerary), 1),
            )
            logger.info("Trip end: updated profile for user %s", user_id)
        except Exception as exc:
            logger.warning("Memory update failed: %s", exc)
        return {"stage": "memory_updated"}

    # 正常流程 → 加载画像
    if user_id and user_id != "anonymous":
        try:
            from data.profile_service import profile_service

            stored = await profile_service.get_profile(user_id)
            if stored:
                for key in (
                    "visited_cities",
                    "favorite_spots",
                    "liked_foods",
                    "avoided_foods",
                    "avg_daily_budget",
                ):
                    if stored.get(key) and not profile.get(key):
                        profile[key] = stored[key]
        except Exception as exc:
            logger.warning("Profile load failed: %s", exc)

    # Preserve terminal stages set by upstream nodes (e.g. after booking) so the
    # graph router can end the loop instead of cycling back to retrieve.
    existing_stage = state.get("stage")
    if existing_stage in ("completed", "memory_updated"):
        return {"profile": profile, "stage": existing_stage}
    # Do not write `stage` here — profile_recall runs in parallel with weather_check
    # and a non-reducer stage write would crash the graph superstep.
    return {"profile": profile}


# ---------------------------------------------------------------------------
# ③ TravelRetrievalRAGAgent — 知识库检索
# ---------------------------------------------------------------------------


@traceable_step("planning/rag_retrieval", run_type="chain")
async def _rag_async(state: dict) -> dict:
    from agents.rag_retrieval import TravelRetrievalRAGAgent
    from models.travel_slots import TravelSlots

    profile = state.get("profile") or {}
    slots_raw = state.get("slots") or {}

    destination = slots_raw.get("destination") or profile.get("destination", "")
    if not destination:
        return {
            "knowledge_results": [],
            "poi_candidates": [],
            "retrieval_query": "",
            "retrieval_empty": True,
        }

    # Normalize raw slots into the structured TravelSlots model
    try:
        slots = TravelSlots(**slots_raw)
    except Exception:
        slots = TravelSlots(destination=destination)

    try:
        agent = TravelRetrievalRAGAgent()
        result = await agent.retrieve(slots, profile=profile, top_k=30)
    except Exception as exc:
        logger.warning("RAG retrieval failed: %s", exc)
        result = {
            "poi_candidates": [],
            "retrieval_query": "",
            "retrieval_empty": True,
            "retrieval_stats": {},
        }

    poi_candidates = result.get("poi_candidates") or []

    # Map POI fields to the downstream itinerary planner schema
    knowledge = [
        {
            "name": p.get("spot_name", ""),
            "category": p.get("spot_type", "attraction"),
            "price": p.get("ticket_price"),
            "tags": p.get("tags") or [],
            "lat": p.get("lat"),
            "lng": p.get("lng"),
            "duration_minutes": p.get("duration_minutes"),
            "walk_intensity": p.get("walk_intensity"),
            "need_reservation": p.get("need_reservation"),
            "reservation_reminder": p.get("reservation_reminder"),
        }
        for p in poi_candidates
        if p.get("spot_name")
    ]

    # Append text knowledge (travel guides) when available
    if destination:
        try:
            from data.repository import repo

            query = f"{destination} {' '.join(profile.get('interests') or [])} 旅游攻略"
            tips = await repo.search_knowledge(query, city=destination, top_k=3)
            knowledge.extend(tips)
        except Exception:
            pass

    return {
        "knowledge_results": knowledge,
        "poi_candidates": poi_candidates,
        "retrieval_query": result.get("retrieval_query", ""),
        "retrieval_empty": result.get("retrieval_empty", True),
        "retrieval_stats": result.get("retrieval_stats", {}),
    }


# ---------------------------------------------------------------------------
# ④ ItineraryPlannerAgent — 行程规划求解
# ---------------------------------------------------------------------------


@traceable_step("planning/amap_matrix", run_type="chain")
async def _trace_build_amap_matrix(poi_inputs: list[Any], api_key: str) -> dict[str, int] | None:
    from planner.preprocessing.amap_distance import build_amap_minutes_map

    return await build_amap_minutes_map(poi_inputs, api_key)


@traceable_step("planning/vrp_solve", run_type="chain")
async def _trace_vrp_solve(request: Any) -> Any:
    from vrp_solver_service.client import VRPSolverClient

    client = VRPSolverClient()
    return await client.solve(request)


@traceable_step("planning/vrp_solve_local", run_type="chain")
def _trace_vrp_solve_local(request: Any) -> Any:
    from vrp_solver_service.solver import TravelVRPSolver

    return TravelVRPSolver().greedy_solve(request)


@traceable_step("planning/planner", run_type="chain")
async def _planner_async(state: dict) -> dict:
    from core.conversation_state import flatten_profile

    profile_raw = state.get("profile") or {}
    slots = state.get("slots") or {}
    knowledge = state.get("knowledge_results") or []

    flat_profile = flatten_profile(profile_raw)
    merged = {**flat_profile, **{k: v for k, v in slots.items() if v is not None and v != []}}

    destination = merged.get("destination", "")
    if not destination:
        return {"next_action": "clarify", "warnings": ["Missing destination"]}

    from schemas import UserProfile
    from planner.core.enhancements import PersonaRules, feasibility_check
    from vrp_solver_service.models import POIInput, ConstraintsInput, SolverRequest

    profile_obj = UserProfile(
        destination=destination,
        travel_days=merged.get("travel_days") or 1,
        travelers_type=merged.get("travelers_type"),
        budget_range=merged.get("budget_range"),
        food_preferences=merged.get("food_preferences") or [],
        interests=merged.get("interests") or [],
        pace=merged.get("pace") or "moderate",
        has_elderly=merged.get("has_elderly", False),
        has_children=merged.get("has_children", False),
        max_walk_minutes=merged.get("max_walk_minutes", 180),
        max_transit_minutes=merged.get("max_transit_minutes", 120),
    )

    # 人群规则调整
    profile_obj = PersonaRules.adjust_profile(profile_obj)

    # 可行性校验
    conflicts = feasibility_check(profile_obj)
    if conflicts:
        logger.warning("Feasibility conflicts: %s", conflicts)

    # Build POI inputs for VRP service
    poi_inputs = [
        POIInput(
            id=k.get("id") or f"poi-{i}",
            name=k.get("name", f"POI-{i}"),
            category=k.get("category", "attraction"),
            lat=k.get("lat") or 0.0,
            lng=k.get("lng") or 0.0,
            score=k.get("score", 0.5),
            ticket_price=k.get("price") or 0.0,
            duration_minutes=k.get("duration_minutes") or 120,
            open_time=k.get("open_time") or "08:00",
            close_time=k.get("close_time") or "18:00",
            walk_intensity=k.get("walk_intensity") or 1,
            tags=k.get("tags") or [],
        )
        for i, k in enumerate(knowledge[:30])
        if k.get("name")
    ]

    if not poi_inputs:
        # fallback: use existing POI query
        from agents.realtime_query import RealtimeQueryAgent
        from skills.city_data import CITY_DEFAULTS

        qa = RealtimeQueryAgent()
        try:
            import asyncio as aio

            pois = await aio.wait_for(
                qa.query_pois(destination, profile_obj.interests + profile_obj.food_preferences),
                timeout=3.0,
            )
        except Exception:
            pois = list(CITY_DEFAULTS.get(destination, []))

        if pois:
            # Convert legacy ScoredPOI to POIInput
            poi_inputs = [
                POIInput(
                    id=getattr(p, "name", f"poi-{i}"),
                    name=getattr(p, "name", f"POI-{i}"),
                    category=getattr(p, "category", "attraction"),
                    lat=getattr(p.location, "lat", 0.0) if getattr(p, "location", None) else 0.0,
                    lng=getattr(p.location, "lng", 0.0) if getattr(p, "location", None) else 0.0,
                    score=getattr(p, "score", 0.5),
                    ticket_price=getattr(p, "ticket_price", 0.0) or 0.0,
                    duration_minutes=120,
                    tags=getattr(p, "tags", []) or [],
                )
                for i, p in enumerate(pois)
            ]
        else:
            # Last resort: AMap API for cities not covered by built-in data
            min_attr = max(6, profile_obj.travel_days * 2)
            include_restaurant = slots.get("include_restaurant", False)
            min_total = max(8, profile_obj.travel_days * 3) if include_restaurant else min_attr
            amap_pois = await _amap_supplement(destination, min_attr, min_total, include_restaurant)
            if amap_pois:
                poi_inputs = amap_pois
            else:
                return {"next_action": "respond", "warnings": ["No POIs found"]}

    poi_inputs = await _ensure_sufficient_pois(
        poi_inputs,
        destination,
        profile_obj.travel_days,
        include_restaurant=slots.get("include_restaurant", False),
    )

    # Real restaurants (from RAG / AMap) are not scheduled as sightseeing stops.
    # They are kept aside as a naming source so each injected meal block can be
    # labelled with a concrete nearby restaurant instead of a bland "用餐".
    # Filter out venues that are transport/scenery hubs miscategorised as dining
    # (e.g. "十六铺游船码头" became a lunch spot) — naming a meal after a ferry
    # pier reads wrong and isn't a real restaurant.
    restaurant_pois = [
        p
        for p in poi_inputs
        if p.category == "restaurant" and (p.lat or p.lng) and _is_dining_venue(p.name)
    ]
    poi_inputs = [p for p in poi_inputs if p.category != "restaurant"]

    # Cap candidate attractions to what a trip of this length can actually
    # schedule (MAX_POI_PER_DAY = 5). RAG returns a fixed top-30, so a 2-day trip
    # would otherwise compute the AMap matrix and solve over ~3x more POIs than
    # it can ever use. Keep must-visit POIs plus the highest-ranked candidates
    # (poi_candidates arrive in RAG fusion order, best first).
    #
    # Sized at 3/day + a small buffer: full-day landmarks and far suburbs each
    # consume a whole day, so a 5-day trip realistically seats ~15 stops, not 25.
    # Over-supplying candidates bloats the O(n²) CP-SAT model and — at real scale
    # with AMap road times — pushed it past the 18s budget into a sub-optimal
    # FEASIBLE stop (observed: 25 POIs → FEASIBLE@18s). ~20 keeps the model small
    # enough for single-worker CP-SAT to reach OPTIMAL while staying generous.
    _max_usable = profile_obj.travel_days * 3 + max(5, profile_obj.travel_days)
    if len(poi_inputs) > _max_usable:
        _must = {m for m in (slots.get("must_visit") or []) if m}
        _kept = [p for p in poi_inputs if p.name in _must]
        for p in poi_inputs:
            if len(_kept) >= _max_usable:
                break
            if p.name not in _must:
                _kept.append(p)
        logger.info(
            "Capped candidate POIs %d → %d for %d-day trip",
            len(poi_inputs),
            len(_kept),
            profile_obj.travel_days,
        )
        poi_inputs = _kept

    # If dining is requested but the knowledge base returned no restaurants,
    # fetch a few real venues so meal blocks can name an actual restaurant.
    if slots.get("include_restaurant", True) and not restaurant_pois:
        restaurant_pois = await _fetch_restaurants(destination)

    max_walk_minutes = getattr(profile_obj, "max_walk_minutes", None) or 120
    travelers_type = _map_travelers_type(getattr(profile_obj, "travelers_type", None) or "adult")

    # Real road-network travel times from AMap (coord-keyed; the solver applies
    # them while building its matrix, with haversine fallback per missing edge).
    # Realistic durations are what make the solver pack each day into a compact
    # geographic cluster instead of zig-zagging across the city.
    amap_minutes = None
    from core.settings import settings as _settings

    if _settings.amap_key:
        try:
            amap_minutes = await _trace_build_amap_matrix(poi_inputs, _settings.amap_key)
            if amap_minutes:
                logger.info("Using AMap road-network times for %d POIs", len(poi_inputs))
        except Exception as exc:
            logger.warning("AMap times unavailable, using haversine fallback: %s", exc)

    # Default meals on: a city itinerary without lunch/dinner is not usable.
    include_restaurant = slots.get("include_restaurant", True)
    meals_per_day = slots.get("meals_per_day", 2 if include_restaurant else 0)

    # Real weekday of each travel day → enables 周一闭馆 constraints in the solver.
    # Parse the start date from the trip dates; skip silently when absent or
    # unparseable so we never guess which day is Monday.
    day_weekdays: list[int] = []
    _dates_raw = merged.get("travel_dates") or slots.get("travel_dates") or ""
    if _dates_raw:
        from datetime import datetime as _dt, timedelta as _td

        _start_str = (
            str(_dates_raw)
            .replace(" to ", "|")
            .replace("~", "|")
            .replace("至", "|")
            .replace("—", "|")
            .split("|")[0]
            .strip()[:10]
        )
        try:
            _start = _dt.strptime(_start_str, "%Y-%m-%d")
            day_weekdays = [
                (_start + _td(days=k)).weekday() for k in range(profile_obj.travel_days)
            ]
        except ValueError:
            day_weekdays = []

    request = SolverRequest(
        pois=poi_inputs,
        constraints=ConstraintsInput(
            travel_days=profile_obj.travel_days,
            day_weekdays=day_weekdays,
            total_budget=profile_obj.budget_range or 0.0,
            max_walk_km=max(1, int(max_walk_minutes / 60 * 4.5)),
            max_transit_minutes=getattr(profile_obj, "max_transit_minutes", None) or 120,
            interests=profile_obj.interests,
            must_visit=slots.get("must_visit") or [],
            play_mode=getattr(profile_obj, "play_mode", None) or "standard",
            include_restaurant=include_restaurant,
            meals_per_day=meals_per_day,
            travelers_type=travelers_type,
        ),
        amap_minutes=amap_minutes,
    )

    # Per-meal spend cap so the venue picker can gate out budget-busting venues
    # (高空酒吧/西餐). Allocate ~35% of the trip budget to food, split per meal,
    # with a 2x tolerance and an ¥80 floor so only true outliers are excluded.
    # 0 (no budget given) disables the gate.
    _budget = profile_obj.budget_range or 0.0
    _mpd = meals_per_day or 0
    meal_budget_cap = (
        max(80.0, _budget * 0.35 / (max(1, profile_obj.travel_days) * _mpd) * 2)
        if _budget and _mpd
        else 0.0
    )

    # Try standalone VRP service first (non-blocking for LangGraph main thread)
    try:
        response = await _trace_vrp_solve(request)
        itinerary_json = _vrp_response_to_itinerary(
            response, restaurant_pois, meal_budget_cap, _dates_raw
        )
        return {
            "itinerary": itinerary_json,
            "warnings": conflicts,
            "next_action": "fact_check",
            "stage": "planned",
            "solve_status": response.status,
            "solve_time_ms": response.solve_time_ms,
        }
    except Exception as exc:
        logger.warning("VRP service unavailable, falling back to local solver: %s", exc)

    # Local greedy fallback (kept for resilience, does not block other threads)
    fallback_response = _trace_vrp_solve_local(request)
    itinerary_json = _vrp_response_to_itinerary(
        fallback_response, restaurant_pois, meal_budget_cap, _dates_raw
    )

    return {
        "itinerary": itinerary_json,
        "warnings": conflicts,
        "next_action": "fact_check",
        "stage": "planned",
        "solve_status": "local_fallback",
    }


def _map_travelers_type(raw: str | None) -> str:
    """Map free-form companion type to the VRP ConstraintsInput enum."""
    mapping = {
        "alone": "solo",
        "solo": "solo",
        "couple": "couple",
        "family": "family_kid",
        "parents": "family_elder",
        "friends": "friends",
        "colleagues": "adult",
        "adult": "adult",
        "young": "young",
    }
    if not raw:
        return "adult"
    return mapping.get(raw.lower(), "adult")


def _parse_recommended_hours(hours_str: str | None) -> int:
    """把 '2-3小时' / '半天' 等建议时长解析为分钟。"""
    if not hours_str:
        return 120
    s = str(hours_str)
    m = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)", s)
    if m:
        return int((float(m.group(1)) + float(m.group(2))) / 2 * 60)
    m = re.search(r"(\d+(?:\.\d+)?)\s*小时?", s)
    if m:
        return int(float(m.group(1)) * 60)
    if "半天" in s:
        return 240
    if "全天" in s:
        return 480
    return 120


def _scored_poi_to_poi_input(poi: Any, idx: int = 0) -> Any:
    """把内置的 ScoredPOI 转成 VRP 的 POIInput。"""
    from vrp_solver_service.models import POIInput

    location = getattr(poi, "location", None)
    lat = getattr(location, "lat", 0.0) if location else 0.0
    lng = getattr(location, "lng", 0.0) if location else 0.0
    duration = _parse_recommended_hours(getattr(poi, "recommended_hours", None))

    best_time = getattr(poi, "best_time", None)
    open_time, close_time = "08:00", "22:00"
    if best_time == "上午":
        open_time, close_time = "08:00", "14:00"
    elif best_time == "下午":
        open_time, close_time = "12:00", "21:00"
    elif best_time == "傍晚":
        open_time, close_time = "16:00", "22:00"
    elif best_time == "晚上":
        open_time, close_time = "18:00", "23:00"
    elif best_time == "全天":
        open_time, close_time = "08:00", "23:00"

    return POIInput(
        id=getattr(poi, "name", f"poi-{idx}"),
        name=getattr(poi, "name", f"POI-{idx}"),
        category=getattr(poi, "category", "attraction"),
        lat=lat,
        lng=lng,
        score=getattr(poi, "score", 0.5),
        ticket_price=getattr(poi, "ticket_price", 0.0) or 0.0,
        duration_minutes=duration,
        open_time=open_time,
        close_time=close_time,
        walk_intensity=1,
        tags=list(getattr(poi, "tags", []) or []),
    )


@traceable_step("planning/poi_supplement", run_type="chain")
async def _ensure_sufficient_pois(
    poi_inputs: list[Any],
    destination: str,
    travel_days: int,
    include_restaurant: bool = False,
) -> list[Any]:
    """当检索到的 POI 不足以支撑多天行程时，补充内置城市默认值，仍不足时尝试高德 API。"""
    min_attractions = max(6, travel_days * 2)
    min_total = max(8, travel_days * 3) if include_restaurant else min_attractions
    attractions = [p for p in poi_inputs if getattr(p, "category", "") == "attraction"]
    total = len(poi_inputs)

    if len(attractions) >= min_attractions and total >= min_total:
        return poi_inputs

    try:
        from skills.city_data import CITY_DEFAULTS
    except Exception:
        CITY_DEFAULTS = {}

    existing_names = {getattr(p, "name", "") for p in poi_inputs}

    # Layer 1: built-in CITY_DEFAULTS
    defaults = CITY_DEFAULTS.get(destination, [])
    added: list[Any] = []
    if defaults:
        for i, poi in enumerate(defaults):
            if poi.name in existing_names:
                continue
            if not include_restaurant and poi.category != "attraction":
                continue
            added.append(_scored_poi_to_poi_input(poi, i))
            existing_names.add(poi.name)

    # Layer 2: AMap API if still insufficient and key configured
    needed_attr = max(
        0,
        min_attractions - len(attractions) - len([a for a in added if a.category == "attraction"]),
    )
    needed_total = max(0, min_total - total - len(added))
    if (needed_attr > 0 or needed_total > 0) and (
        amap_pois := await _amap_supplement(
            destination, needed_attr, needed_total, include_restaurant
        )
    ):
        for i, poi in enumerate(amap_pois):
            if poi.name in existing_names:
                continue
            added.append(poi)
            existing_names.add(poi.name)

    # Select exactly how many we still need
    final_needed_attr = max(0, min_attractions - len(attractions))
    final_needed_total = max(0, min_total - total)
    selected: list[Any] = []
    for poi in added:
        if poi.category == "attraction" and final_needed_attr > 0:
            selected.append(poi)
            final_needed_attr -= 1
        elif poi.category != "attraction" and final_needed_total > 0:
            selected.append(poi)
            final_needed_total -= 1
        if final_needed_attr <= 0 and final_needed_total <= 0:
            break

    if selected:
        logger.warning(
            "POI supply low for %s: %d attractions, %d total; supplementing %d fallback POIs",
            destination,
            len(attractions),
            total,
            len(selected),
        )
    return poi_inputs + selected


async def _amap_supplement(
    destination: str,
    needed_attr: int,
    needed_total: int,
    include_restaurant: bool,
) -> list[Any]:
    """Try AMap API to fetch extra POIs when built-in data is not enough."""
    from core.settings import settings

    if not settings.amap_key:
        return []

    try:
        from data.collectors.amap import AmapCollector
    except Exception as exc:
        logger.debug("AmapCollector unavailable: %s", exc)
        return []

    collector = AmapCollector(settings.amap_key)
    results: list[Any] = []
    try:
        if needed_attr > 0:
            raw = await collector.search_pois(
                destination, types="风景名胜|公园广场|寺庙道观|纪念馆"
            )
            for i, r in enumerate(raw):
                results.append(_raw_poi_to_poi_input(r, f"amap-attr-{i}"))
        if include_restaurant and needed_total > 0:
            raw = await collector.search_pois(destination, types="中餐厅|外国餐厅|小吃快餐店")
            for i, r in enumerate(raw):
                results.append(_raw_poi_to_poi_input(r, f"amap-rest-{i}"))
    except Exception as exc:
        logger.warning("AMAP supplement failed for %s: %s", destination, exc)
    finally:
        await collector.close()

    return results


@traceable_step("planning/fetch_restaurants", run_type="chain")
async def _fetch_restaurants(destination: str) -> list[Any]:
    """Fetch real restaurants for naming meal blocks (one AMap call, low QPS)."""
    from core.settings import settings
    from vrp_solver_service.models import POIInput

    if not settings.amap_key:
        return []
    try:
        from data.collectors.amap import AmapCollector
    except Exception as exc:
        logger.debug("AmapCollector unavailable for restaurants: %s", exc)
        return []

    collector = AmapCollector(settings.amap_key)
    out: list[Any] = []
    try:
        raw = await collector.search_pois(destination, types="中餐厅|外国餐厅|小吃快餐店")
        for i, r in enumerate(raw):
            if not (getattr(r, "lat", 0) or getattr(r, "lng", 0)):
                continue
            if not _is_dining_venue(getattr(r, "name", "")):
                continue
            out.append(
                POIInput(
                    id=f"{r.name}-rest-{i}",
                    name=r.name,
                    category="restaurant",
                    lat=getattr(r, "lat", 0.0) or 0.0,
                    lng=getattr(r, "lng", 0.0) or 0.0,
                    score=0.6,
                    # AMap returns per-capita spend in biz_ext.cost (collector maps
                    # it to ticket_price). Keep it so the picker can gate out venues
                    # that blow the daily food budget (高空酒吧/西餐). 0 = unknown.
                    ticket_price=getattr(r, "ticket_price", 0.0) or 0.0,
                    duration_minutes=60,
                    tags=getattr(r, "tags", []) or [],
                )
            )
    except Exception as exc:
        logger.warning("Restaurant fetch failed for %s: %s", destination, exc)
    finally:
        await collector.close()
    return out


def _raw_poi_to_poi_input(raw: Any, idx: str) -> Any:
    """Convert AmapCollector RawPOI to VRP POIInput."""
    from vrp_solver_service.models import POIInput

    return POIInput(
        id=f"{raw.name}-{idx}",
        name=raw.name,
        category=raw.category or "attraction",
        lat=getattr(raw, "lat", 0.0) or 0.0,
        lng=getattr(raw, "lng", 0.0) or 0.0,
        score=0.7,
        ticket_price=getattr(raw, "ticket_price", 0.0) or 0.0,
        duration_minutes=120,
        open_time="08:00",
        close_time="18:00",
        walk_intensity=2,
        tags=list(getattr(raw, "tags", []) or []),
    )


def _vrp_response_to_itinerary(
    response,
    restaurant_pois=None,
    meal_budget: float = 0.0,
    travel_dates: str | None = None,
) -> list[dict]:
    """Convert VRP SolverResponse to the itinerary dict used by downstream nodes.

    Injected meal blocks (``午餐``/``晚餐``) are matched to a concrete nearby
    restaurant — the one closest to the attraction visited just before the meal
    — so the plan names a real venue instead of a generic placeholder.

    ``meal_budget`` is the per-meal spend cap (¥); venues whose per-capita cost
    exceeds it are skipped while a cheaper option exists, so a high-end 西餐/酒吧
    cannot blow the daily food budget. 0 disables the gate.
    """
    import re as _re

    restaurant_pois = list(restaurant_pois or [])

    # Local Shanghai-style cuisine tags get a bias so the trip has a 本帮 main
    # line instead of drifting into repeated chain 寿喜烧/日料.
    _LOCAL_TAGS = {
        "本帮",
        "本帮菜",
        "上海菜",
        "沪菜",
        "小笼",
        "小笼包",
        "生煎",
        "面馆",
        "小吃",
        "弄堂",
        "面馆",
    }
    # Pricey/non-local venue types that repeatedly broke the budget (FLAIR 高空
    # 酒吧, 夏朵花园 西餐). Downranked when a per-meal budget is set.
    _PRICEY_TAGS = {
        "酒吧",
        "bar",
        "西餐",
        "西餐厅",
        "法餐",
        "日本料理",
        "自助餐",
        "buffet",
        "高档",
        "星级",
    }

    def _brand(name: str) -> str:
        """Strip a trailing branch suffix so chain outlets dedup as one brand.

        "牛New寿喜烧(松江店)" and "牛New寿喜烧(上海中心店)" → "牛New寿喜烧".
        """
        return _re.sub(r"[（(].*?[)）]\s*$", "", name or "").strip() or (name or "")

    def _cuisine_key(r) -> str:
        """A coarse cuisine signature for diversity (first meaningful tag)."""
        for t in r.tags or []:
            return t
        return _brand(r.name)

    def _over_budget(r) -> bool:
        """True iff the venue's known per-capita cost exceeds the meal budget."""
        price = getattr(r, "ticket_price", 0.0) or 0.0
        return bool(meal_budget) and price > meal_budget

    def _pick_restaurant(lat: float, lng: float, used_brands: set[str], used_cuisines: set[str]):
        def _score(r, skip_used: bool, allow_over: bool):
            if not (r.lat or r.lng):
                return None
            if skip_used and _brand(r.name) in used_brands:
                return None
            if not allow_over and _over_budget(r):
                return None
            # Squared planar distance is enough for ranking nearby venues.
            dist2 = (r.lat - lat) ** 2 + (r.lng - lng) ** 2 if (lat or lng) else 0.0
            # Higher score = better pick. Distance dominates (must stay on-route),
            # then unused-cuisine diversity, then a local-cuisine bias, then a soft
            # penalty for pricey venue types when a budget is in force.
            score = -dist2 * 1e4
            if _cuisine_key(r) not in used_cuisines:
                score += 0.6
            if _LOCAL_TAGS & set(r.tags or []):
                score += 0.6
            if meal_budget and (_PRICEY_TAGS & {t.lower() for t in (r.tags or [])}):
                score -= 0.8
            return score

        def _best(skip_used: bool, allow_over: bool):
            best, best_s = None, float("-inf")
            for r in restaurant_pois:
                s = _score(r, skip_used, allow_over)
                if s is not None and s > best_s:
                    best, best_s = r, s
            return best

        # Prefer an unused, within-budget brand; relax the budget, then the
        # brand-dedup, only if nothing else qualifies.
        return (
            _best(skip_used=True, allow_over=False)
            or _best(skip_used=False, allow_over=False)
            or _best(skip_used=True, allow_over=True)
            or _best(skip_used=False, allow_over=True)
        )

    # Dedup restaurants across the *whole trip* at brand level (chain outlets
    # count as one) and steer toward cuisine variety + a local main line.
    used_brands: set[str] = set()
    used_cuisines: set[str] = set()
    start_date = None
    if travel_dates:
        from datetime import datetime as _dt

        raw_start = (
            str(travel_dates)
            .replace(" to ", "|")
            .replace("~", "|")
            .replace("至", "|")
            .replace("—", "|")
            .split("|")[0]
            .strip()[:10]
        )
        try:
            start_date = _dt.strptime(raw_start, "%Y-%m-%d")
        except ValueError:
            start_date = None

    itinerary = []
    for day in response.days:
        activities = []
        last_lat, last_lng = 0.0, 0.0
        previous_end: int | None = None
        total_transit = 0
        for a in day.activities:
            poi_name = a.poi_name
            tags = a.tags
            is_meal = a.category == "restaurant" and poi_name in ("午餐", "晚餐", "用餐")
            if is_meal:
                venue = _pick_restaurant(last_lat, last_lng, used_brands, used_cuisines)
                if venue is not None:
                    used_brands.add(_brand(venue.name))
                    used_cuisines.add(_cuisine_key(venue))
                    label = poi_name if poi_name in ("午餐", "晚餐") else "用餐"
                    poi_name = f"{label} · {venue.name}"
                    tags = venue.tags or tags
            else:
                # Track the most recent sightseeing location to anchor the meal.
                last_lat, last_lng = a.lat, a.lng
            start_parts = str(a.start_time).split(":")
            end_parts = str(a.end_time).split(":")
            start_min = int(start_parts[0]) * 60 + int(start_parts[1])
            end_min = int(end_parts[0]) * 60 + int(end_parts[1])
            transit_min = max(0, start_min - previous_end) if previous_end is not None else 0
            total_transit += transit_min
            activities.append(
                {
                    "poi_name": poi_name,
                    "category": a.category,
                    "start_time": a.start_time,
                    "end_time": a.end_time,
                    "duration_min": a.duration_min,
                    "ticket_price": a.ticket_price,
                    "transport_cost": a.transport_cost,
                    "transit_from_prev": (
                        {"mode": "transit", "duration_min": transit_min}
                        if previous_end is not None
                        else None
                    ),
                    "location": {"lat": a.lat, "lng": a.lng},
                    "tags": tags,
                }
            )
            previous_end = end_min
        day_date = None
        if start_date is not None:
            from datetime import timedelta as _td

            day_date = (start_date + _td(days=day.day_number - 1)).strftime("%Y-%m-%d")
        itinerary.append(
            {
                "day_number": day.day_number,
                "date": day_date,
                "activities": activities,
                "total_cost": day.total_cost,
                "transport_cost": day.transport_cost,
                "total_transit_time_min": total_transit,
                "total_walking_steps": 0,
            }
        )
    return itinerary


# ---------------------------------------------------------------------------
# ⑤ FactCheckAgent — 事实校验 & 幻觉拦截
# ---------------------------------------------------------------------------


@traceable_step("planning/factcheck", run_type="chain")
async def _fact_check_async(state: dict) -> dict:
    itinerary = state.get("itinerary", [])
    if not itinerary:
        return {"stage": "fact_check_done"}

    # Shadow-mode deterministic validation: publish the versioned report now,
    # but preserve the current routing until Phase 1 rollout explicitly enables
    # the hard completion gate.
    from core.conversation_state import flatten_profile
    from evaluation.validator import ItineraryValidator

    profile = flatten_profile(state.get("profile") or {})
    validation_report = (
        ItineraryValidator()
        .validate(
            itinerary,
            constraints={
                "travel_days": profile.get("travel_days"),
                "total_budget": profile.get("budget_range"),
                "max_transit_minutes": profile.get("max_transit_minutes"),
                "interests": profile.get("interests") or [],
            },
            facts=state.get("poi_candidates") or [],
        )
        .model_dump()
    )
    from agentic.termination import CompletionGuard
    from core.settings import settings

    completion_decision = (
        CompletionGuard(mode=settings.agentic_completion_guard_mode)
        .evaluate(validation_report)
        .model_dump()
    )

    conflicts = []
    try:
        from sqlalchemy import text

        from core.database import async_session_maker

        async with async_session_maker() as db:
            for day in itinerary:
                for act in day.get("activities", []):
                    poi_name = act.get("poi_name", "")
                    if not poi_name:
                        continue
                    # 查数据库校验
                    result = await db.execute(
                        text(
                            "SELECT ticket_price, open_time, close_time, status "
                            "FROM attractions WHERE name = :name"
                        ),
                        {"name": poi_name},
                    )
                    row = result.mappings().first()
                    if not row:
                        continue
                    if row["status"] == "deprecated":
                        conflicts.append(f"{poi_name} 已永久关闭")
                    if row["ticket_price"] is not None and act.get("ticket_price"):
                        db_price = float(row["ticket_price"])
                        act_price = float(act["ticket_price"])
                        if abs(db_price - act_price) > 50:
                            conflicts.append(
                                f"{poi_name} 门票价格冲突: 行程¥{act_price} vs 数据库¥{db_price}"
                            )
    except Exception as exc:
        logger.warning("FactCheck failed: %s", exc)

    if conflicts:
        loops = state.get("loop_count", 0)
        max_loops = state.get("max_loops", 3)
        if loops < max_loops:
            # Replan is worthwhile and within budget — advance the counter here
            # (routers must stay pure; their mutations are not persisted).
            # warnings is a reducer field → return only the delta.
            return {
                "validation_report": validation_report,
                "completion_decision": completion_decision,
                "warnings": conflicts,
                "loop_count": loops + 1,
                "next_action": "planner",
                "stage": "fact_check_failed",
            }
        # Budget exhausted → keep current plan, surface the unresolved conflict.
        return {
            "validation_report": validation_report,
            "completion_decision": completion_decision,
            "warnings": conflicts + ["事实校验冲突未解决，仍按当前方案输出。"],
            "next_action": "factcheck_done",
            "stage": "fact_check_exhausted",
        }

    return {
        "stage": "fact_check_done",
        "validation_report": validation_report,
        "completion_decision": completion_decision,
    }


# ---------------------------------------------------------------------------
# ⑥ Output&DocAgent — 多模态输出 & 文档
# ---------------------------------------------------------------------------


@traceable_step("planning/respond", run_type="chain")
async def _respond_async(state: dict) -> str:
    """Answer non-planning turns (query_info / chitchat) with the chat LLM.

    For query_info we attach a light RAG context from the knowledge base; for
    other small talk we fall back to a friendly capability hint on failure.
    """
    from core.conversation_state import flatten_profile
    from core.llm_client import llm

    user_input = state.get("user_input", "")
    intent = state.get("intent", "")
    flat = flatten_profile(state.get("profile") or {})
    destination = (state.get("slots") or {}).get("destination") or flat.get("destination") or ""

    context = ""
    if intent == "query_info" and destination:
        try:
            from data.repository import repo

            tips = await repo.search_knowledge(user_input, city=destination, top_k=3)
            snippets = []
            for t in tips or []:
                if isinstance(t, dict):
                    snippets.append(t.get("content") or t.get("tip") or t.get("text") or "")
                else:
                    snippets.append(str(t))
            context = "\n".join(s for s in snippets if s)
        except Exception as exc:
            logger.debug("Respond RAG lookup failed: %s", exc)

    system = "你是专业的旅行助手，用简洁、友好的中文回答用户的旅行相关问题，控制在 3-5 句内。"
    user_msg = user_input
    if context:
        user_msg = f"参考资料：\n{context}\n\n用户问题：{user_input}"

    try:
        answer = await llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
            task_type="chat",
        )
        return (
            answer.strip()
            or "我可以帮你规划行程、推荐景点和美食。告诉我你想去哪、玩几天，我来帮你安排。"
        )
    except Exception as exc:
        logger.warning("Respond generation failed: %s", exc)
        return "我可以帮你规划行程、推荐景点和美食。告诉我你想去哪、玩几天，我来帮你安排。"


@traceable_step("planning/output", run_type="chain")
async def _output_async(state: dict) -> dict:
    next_action = state.get("next_action", "respond")

    if next_action == "clarify":
        from agents.demand_parser import DemandParserAgent
        from core.conversation_state import flatten_profile

        questions = DemandParserAgent.build_clarification_questions(
            flatten_profile(state.get("profile") or {}),
            existing=state.get("clarification_questions"),
        )
        # Join all questions into a single natural message
        content = "\n".join(questions) if questions else "请问还有什么需要补充的吗？"
        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "assistant",
                    "content": content,
                    "type": "clarification",
                    "questions": questions,
                    "missing_slots": state.get("missing_slots", []),
                }
            ],
            "stage": "gathering",
            "next_action": "clarify",
            "profile": state.get("profile"),
            "clarification_questions": questions,
            "missing_slots": state.get("missing_slots", []),
            "conversation_sync": state.get("conversation_sync"),
        }

    # Infeasible request: surface the hard conflicts instead of planning.
    if next_action == "infeasible":
        issues = (state.get("feasibility_report") or {}).get("issues") or []
        body = "\n".join(f"• {i}" for i in issues) or "当前需求存在可行性冲突。"
        content = f"你的需求暂时不太可行，主要问题：\n{body}\n\n方便的话调整一下（比如放宽预算、减少天数或放缓节奏），我再帮你规划。"
        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "assistant",
                    "content": content,
                    "type": "infeasible",
                }
            ],
            "stage": "infeasible",
            "next_action": "infeasible",
        }

    # Non-planning intents (query_info / chitchat / update_preferences / view_history).
    if next_action == "respond":
        answer = await _respond_async(state)
        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "assistant",
                    "content": answer,
                    "type": "text",
                }
            ],
            "stage": "responded",
            "next_action": "respond",
        }

    itinerary = state.get("itinerary", [])
    if not itinerary:
        return {
            "messages": state.get("messages", [])
            + [
                {
                    "role": "assistant",
                    "content": "抱歉，暂时无法生成行程。",
                }
            ],
            "stage": "completed",
        }

    from core.conversation_state import flatten_profile

    profile_raw = flatten_profile(state.get("profile") or {})
    profile_raw.update(
        {
            key: value
            for key, value in (state.get("slots") or {}).items()
            if value not in (None, "", [])
        }
    )
    from schemas import DayPlan, UserProfile
    from planner.core.writer import enrich as enrich_writer

    profile_obj = UserProfile(
        destination=profile_raw.get("destination"),
        travel_days=profile_raw.get("travel_days") or 1,
        travel_dates=profile_raw.get("travel_dates"),
        travelers_count=profile_raw.get("travelers_count"),
        travelers_type=profile_raw.get("travelers_type"),
        budget_range=profile_raw.get("budget_range"),
        interests=profile_raw.get("interests") or [],
        food_preferences=profile_raw.get("food_preferences") or [],
        food_taboos=profile_raw.get("food_taboos") or [],
        pace=profile_raw.get("pace") or "moderate",
        has_elderly=bool(profile_raw.get("has_elderly", False)),
        has_children=bool(profile_raw.get("has_children", False)),
    )

    try:
        days = [DayPlan(**d) for d in itinerary]
        enriched, proposal_text = await enrich_writer(days, profile_obj)
        itinerary_enriched = [day.model_dump() for day in enriched]
    except Exception as exc:
        logger.warning("Writer failed: %s", exc)
        proposal_text = _format_simple(itinerary, profile_raw)
        itinerary_enriched = itinerary

    return {
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": proposal_text,
                "type": "itinerary",
                "itinerary": itinerary_enriched,
                "warnings": state.get("warnings", []),
            }
        ],
        "itinerary": itinerary_enriched,
        "stage": "awaiting_booking",
    }


def _format_simple(itinerary: list[dict], profile: dict) -> str:
    dest = profile.get("destination", "目的地")
    days = len(itinerary)
    lines = [f"# {dest} {days}日游行程方案\n"]
    ticket_food_total = 0.0
    day_total = 0.0
    for day in itinerary:
        lines.append(f"## 第{day.get('day_number', '?')}天")
        for act in day.get("activities", []):
            cost = act.get("ticket_price", 0) or act.get("meal_cost", 0) or 0
            ticket_food_total += cost
            time_str = (
                f"{act.get('start_time', '')}-{act.get('end_time', '')}"
                if act.get("start_time")
                else ""
            )
            cost_str = f" — ¥{cost:.0f}" if cost else ""
            lines.append(f"  {time_str} {act.get('poi_name', '?')}{cost_str}")
        # day total_cost (from solver) already includes tickets + food + city transit
        day_total += day.get("total_cost", 0) or 0

    # Prefer the solver's per-day total (includes city transit); fall back to the
    # ticket+meal sum. Either way, label the scope so the number isn't mistaken
    # for an all-in trip cost (it excludes hotel and inter-city transport).
    est = day_total if day_total > 0 else ticket_food_total
    lines.append(
        f"\n预估费用（门票+餐饮+市内交通）: 约 ¥{est:.0f}"
        f"\n> 不含住宿与往返大交通，实际以官方渠道为准"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ⑦ BookingToolAgent — 预订工具搜索 & 汇总
# ---------------------------------------------------------------------------


@traceable_step("planning/booking", run_type="chain")
async def _booking_tool_async(state: dict) -> dict:
    """
    预订 Agent：自动搜索机票/酒店/门票/餐厅，汇总展示。
    所有数据标注 source=mock，待后续接入真实 API。
    """
    itinerary = state.get("itinerary", [])
    from core.conversation_state import flatten_profile

    profile = flatten_profile(state.get("profile") or {})

    if not itinerary:
        return {"stage": "completed"}

    destination = profile.get("destination", "")
    origin = profile.get("origin", "")
    travel_days = len(itinerary)
    travelers_count = int(profile.get("travelers_count") or 1)
    budget_range = profile.get("budget_range")

    booking = {"flights": [], "hotels": [], "tickets": [], "restaurants": [], "source": "mock"}

    # ── 机票 ──
    if origin and destination:
        flight_key = f"{origin}-{destination}"
        _MOCK_FLIGHTS = {
            "北京-成都": [
                {"no": "CA4101", "dep": "07:30", "arr": "10:15", "price": 680},
                {"no": "MU5210", "dep": "14:00", "arr": "16:45", "price": 520},
                {"no": "CZ8842", "dep": "19:30", "arr": "22:15", "price": 380},
            ],
        }
        flights = _MOCK_FLIGHTS.get(
            flight_key,
            [
                {"no": "CA0001", "dep": "08:00", "arr": "11:00", "price": 600},
                {"no": "MU0002", "dep": "14:00", "arr": "17:00", "price": 480},
            ],
        )
        booking["flights"] = flights

    # ── 酒店 ──
    _MOCK_HOTELS: dict[str, list[dict]] = {
        "成都": [
            {"name": "春熙路亚朵酒店", "district": "锦江区", "price": 350, "rating": 4.7},
            {"name": "宽窄巷子全季酒店", "district": "青羊区", "price": 280, "rating": 4.5},
            {"name": "天府广场汉庭酒店", "district": "锦江区", "price": 180, "rating": 4.2},
            {"name": "成都希尔顿酒店", "district": "高新区", "price": 680, "rating": 4.9},
        ],
    }
    hotels = _MOCK_HOTELS.get(
        destination,
        [
            {
                "name": f"{destination}经济酒店（示例）",
                "district": "市中心",
                "price": 220,
                "rating": 4.2,
            },
            {
                "name": f"{destination}舒适酒店（示例）",
                "district": "市中心",
                "price": 360,
                "rating": 4.5,
            },
            {
                "name": f"{destination}品质酒店（示例）",
                "district": "市中心",
                "price": 520,
                "rating": 4.7,
            },
        ],
    )
    if budget_range:
        daily = budget_range / max(travel_days, 1) * 0.35  # 住宿占35%
        hotels = [h for h in hotels if h["price"] <= daily] or hotels[:1]
    booking["hotels"] = hotels

    # ── 门票 ──
    for day in itinerary:
        for act in day.get("activities", []):
            if act.get("category") not in {"attraction", "spot", "museum", "park"}:
                continue
            poi_name = act.get("poi_name", "")
            if poi_name and poi_name not in {t.get("poi_name") for t in booking["tickets"]}:
                booking["tickets"].append(
                    {
                        "poi_name": poi_name,
                        "price": act.get("ticket_price") or 0,
                        "need_reserve": bool(act.get("need_reservation")),
                    }
                )

    # ── 餐厅 ──
    restaurant_pool = {
        "成都": [
            {"name": "蜀大侠火锅", "cuisine": "川菜", "per_person": 80, "tags": ["辣", "火锅"]},
            {"name": "陈麻婆豆腐", "cuisine": "川菜", "per_person": 45, "tags": ["麻辣", "经典"]},
            {"name": "龙抄手", "cuisine": "小吃", "per_person": 30, "tags": ["清淡", "面食"]},
            {"name": "大蓉和", "cuisine": "川菜", "per_person": 120, "tags": ["高端", "宴请"]},
        ],
    }
    restaurants = restaurant_pool.get(
        destination,
        [{"name": f"{destination}本地菜馆", "cuisine": "本地菜", "per_person": 60, "tags": []}],
    )
    booking["restaurants"] = restaurants[:3]

    # ── 格式化 ──
    city_transport = (
        sum(
            float(day.get("transport_cost") or 0)
            or sum(float(act.get("transport_cost") or 0) for act in day.get("activities", []))
            for day in itinerary
        )
        * travelers_count
    )
    msg, budget_breakdown = _format_booking_summary(
        booking, destination, origin, travel_days, travelers_count, city_transport
    )

    return {
        "messages": state.get("messages", [])
        + [
            {
                "role": "assistant",
                "content": msg,
                "type": "booking",
                "booking_results": booking,
            }
        ],
        "booking_results": booking,
        "budget_breakdown": budget_breakdown,
        "stage": "completed",
    }


def _format_booking_summary(
    booking: dict,
    destination: str,
    origin: str,
    days: int,
    travelers_count: int = 1,
    city_transport: float = 0,
) -> tuple[str, dict]:
    """格式化预订摘要为 Markdown 文案。"""
    lines = ["# 📋 预订参考\n"]
    total_est = 0

    # 机票
    if booking["flights"]:
        lines.append("## ✈️ 机票")
        for f in booking["flights"]:
            lines.append(f"- {f['no']} {f['dep']}-{f['arr']}  ¥{f['price']}")
        best = min(booking["flights"], key=lambda x: x["price"])
        transport_est = best["price"] * 2 * travelers_count
        total_est += transport_est
        lines.append(f"\n> 推荐 {best['no']}，{travelers_count} 人往返约 ¥{transport_est}\n")
    else:
        transport_est = 0

    # 酒店
    if booking["hotels"]:
        lines.append("## 🏨 酒店")
        nights = max(days - 1, 0)
        rooms = max(1, (travelers_count + 1) // 2) if nights else 0
        for h in booking["hotels"][:3]:
            cost = h["price"] * nights * rooms
            lines.append(
                f"- {h['name']}（{h['district']}）⭐{h['rating']}  "
                f"¥{h['price']}/间夜 × {rooms}间 × {nights}晚 = ¥{cost}"
            )
        if nights:
            best_h = booking["hotels"][0]
            hotel_est = best_h["price"] * nights * rooms
            total_est += hotel_est
            lines.append(f"\n> 推荐 {best_h['name']}，{rooms}间 × {nights}晚约 ¥{hotel_est}\n")
        else:
            hotel_est = 0
    else:
        hotel_est = 0

    # 门票
    if booking["tickets"]:
        lines.append("## 🎫 门票")
        tix_total = 0
        for t in booking["tickets"]:
            reserve = "⚠️需预约" if t.get("need_reserve") else "免预约"
            lines.append(f"- {t['poi_name']}  ¥{t['price']} {reserve}")
            tix_total += t["price"]
        ticket_est = tix_total * travelers_count
        total_est += ticket_est
        lines.append(f"\n> {travelers_count} 人门票合计约 ¥{ticket_est}\n")
    else:
        ticket_est = 0

    if city_transport > 0:
        total_est += city_transport
        lines.append(f"## 🚇 市内交通\n- 按行程路线估算：约 ¥{city_transport:.0f}\n")

    # 餐厅
    if booking["restaurants"]:
        lines.append("## 🍜 推荐餐厅")
        for r in booking["restaurants"][:3]:
            tags_str = " · ".join(r.get("tags", []))
            lines.append(f"- {r['name']}（{r['cuisine']}）人均 ¥{r['per_person']} {tags_str}")
        lines.append("")

    # 总预估
    if total_est > 0:
        food_est = 100 * days * travelers_count
        total_est += food_est
        lines.append(f"---\n💰 **预估总费用：约 ¥{total_est:,}**（机票+酒店+门票+餐饮）")
        if origin:
            lines.append("\n> ⚠️ 以上为模拟参考价，来源标注 mock，实际请以官方渠道为准")

    breakdown = {
        "source": "mock_estimate",
        "travelers_count": travelers_count,
        "intercity_transport": transport_est,
        "local_transport": city_transport,
        "accommodation": hotel_est,
        "tickets": ticket_est,
        "food": 100 * days * travelers_count,
        "total": total_est,
    }
    return "\n".join(lines), breakdown
