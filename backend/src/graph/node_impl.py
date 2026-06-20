"""Async node implementations shared by the LangGraph orchestration layer.

These helpers were originally part of backend/src/agent/graph.py. They are kept
here as pure async implementations so the old graph can be removed while the new
graph in backend/src/graph continues to use them.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ① DemandParserAgent — 需求解析
# ---------------------------------------------------------------------------


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
    import httpx

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
        from skills.weather_query import CITY_COORDS, _estimate_temp, _estimate_condition, _estimate_precip
        import random
        coords = CITY_COORDS.get(destination, CITY_COORDS.get(destination.rstrip("市省")))
        lat = coords[0] if coords else 30.0
        try:
            s = datetime.strptime(start_date, "%Y-%m-%d")
            e = datetime.strptime(end_date, "%Y-%m-%d")
            cur = s
            while cur <= e:
                h, l = _estimate_temp(lat, cur.month)
                h += random.randint(-3, 3)
                l += random.randint(-2, 2)
                weather.append({
                    "date": cur.strftime("%Y-%m-%d"),
                    "condition": _estimate_condition(lat, cur.month),
                    "temp_high": h, "temp_low": l,
                    "precipitation_chance": _estimate_precip(lat, cur.month),
                    "recommendation": "（估算数据）",
                    "data_source": "fallback",
                    "is_fallback": True,
                })
                cur += timedelta(days=1)
        except ValueError:
            pass

    return {
        "weather": weather,
        "weather_fetched": len(weather) > 0,
        "weather_start": start_date,
        "weather_end": end_date,
    }


async def _fetch_amap_weather(destination: str, start_date: str, end_date: str, key: str) -> list[dict]:
    """Fetch weather from AMap API for weather_check_node."""
    from datetime import datetime
    import httpx

    # City → adcode mapping (subset of weather_query's _CITY_ADCODE)
    ADCODE: dict[str, str] = {
        "北京": "110000", "上海": "310000", "广州": "440100", "深圳": "440300",
        "成都": "510100", "杭州": "330100", "西安": "610100", "重庆": "500000",
        "苏州": "320500", "南京": "320100", "厦门": "350200", "青岛": "370200",
        "大理": "532901", "丽江": "530700", "三亚": "460200", "长沙": "430100",
        "武汉": "420100", "昆明": "530100", "桂林": "450300", "拉萨": "540100",
        "济南": "370100", "郑州": "410100", "天津": "120000", "合肥": "340100",
        "哈尔滨": "230100", "长春": "220100", "沈阳": "210100",
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
                results.append({
                    "date": cast["date"],
                    "condition": cast.get("dayweather", "多云"),
                    "temp_high": int(float(cast.get("daytemp_float", cast.get("daytemp", 25)))),
                    "temp_low": int(float(cast.get("nighttemp_float", cast.get("nighttemp", 15)))),
                    "precipitation_chance": 70 if "雨" in cast.get("dayweather", "") else 10,
                    "data_source": "api",
                    "is_fallback": False,
                })
    return results


# ---------------------------------------------------------------------------
# ② UserMemoryRecallAgent — 用户画像记忆
# ---------------------------------------------------------------------------


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
                trip_budget=sum(
                    d.get("total_cost", 0) for d in itinerary
                ) / max(len(itinerary), 1),
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
                for key in ("visited_cities", "favorite_spots", "liked_foods",
                            "avoided_foods", "avg_daily_budget"):
                    if stored.get(key) and not profile.get(key):
                        profile[key] = stored[key]
        except Exception as exc:
            logger.warning("Profile load failed: %s", exc)

    # Preserve terminal stages set by upstream nodes (e.g. after booking) so the
    # graph router can end the loop instead of cycling back to retrieve.
    existing_stage = state.get("stage")
    if existing_stage in ("completed", "memory_updated"):
        return {"profile": profile, "stage": existing_stage}
    return {"profile": profile, "stage": "memory_loaded"}


# ---------------------------------------------------------------------------
# ③ TravelRetrievalRAGAgent — 知识库检索
# ---------------------------------------------------------------------------


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
            "stage": "rag_done",
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
        "stage": "rag_done",
    }


# ---------------------------------------------------------------------------
# ④ ItineraryPlannerAgent — 行程规划求解
# ---------------------------------------------------------------------------


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
    from vrp_solver_service.client import VRPSolverClient
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
            walk_intensity=k.get("walk_intensity") or 3,
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
            amap_pois = await _amap_supplement(
                destination, min_attr, min_total, include_restaurant
            )
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

    max_walk_minutes = getattr(profile_obj, "max_walk_minutes", None) or 120
    travelers_type = _map_travelers_type(getattr(profile_obj, "travelers_type", None) or "adult")
    request = SolverRequest(
        pois=poi_inputs,
        constraints=ConstraintsInput(
            travel_days=profile_obj.travel_days,
            total_budget=profile_obj.budget_range or 0.0,
            max_walk_km=max(1, int(max_walk_minutes / 60 * 4.5)),
            max_transit_minutes=getattr(profile_obj, "max_transit_minutes", None) or 120,
            interests=profile_obj.interests,
            must_visit=slots.get("must_visit") or [],
            play_mode=getattr(profile_obj, "play_mode", None) or "standard",
            include_restaurant=slots.get("include_restaurant", False),
            meals_per_day=slots.get("meals_per_day", 0),
            travelers_type=travelers_type,
        ),
    )

    # Try standalone VRP service first (non-blocking for LangGraph main thread)
    try:
        client = VRPSolverClient()
        response = await client.solve(request)
        itinerary_json = _vrp_response_to_itinerary(response)
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
    from vrp_solver_service.solver import TravelVRPSolver

    fallback_response = TravelVRPSolver().greedy_solve(request)
    itinerary_json = _vrp_response_to_itinerary(fallback_response)

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
        walk_intensity=3,
        tags=list(getattr(poi, "tags", []) or []),
    )


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
    needed_attr = max(0, min_attractions - len(attractions) - len([a for a in added if a.category == "attraction"]))
    needed_total = max(0, min_total - total - len(added))
    if (needed_attr > 0 or needed_total > 0) and (amap_pois := await _amap_supplement(destination, needed_attr, needed_total, include_restaurant)):
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
            raw = await collector.search_pois(destination, types="风景名胜|公园广场|寺庙道观|纪念馆")
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


def _vrp_response_to_itinerary(response) -> list[dict]:
    """Convert VRP SolverResponse to the itinerary dict used by downstream nodes."""
    itinerary = []
    for day in response.days:
        activities = []
        for a in day.activities:
            activities.append({
                "poi_name": a.poi_name,
                "category": a.category,
                "start_time": a.start_time,
                "end_time": a.end_time,
                "duration_min": a.duration_min,
                "ticket_price": a.ticket_price,
                "location": {"lat": a.lat, "lng": a.lng},
                "tags": a.tags,
            })
        itinerary.append({
            "day_number": day.day_number,
            "activities": activities,
            "total_cost": day.total_cost,
        })
    return itinerary


# ---------------------------------------------------------------------------
# ⑤ FactCheckAgent — 事实校验 & 幻觉拦截
# ---------------------------------------------------------------------------


async def _fact_check_async(state: dict) -> dict:
    itinerary = state.get("itinerary", [])
    if not itinerary:
        return {"stage": "fact_check_done"}

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
        return {
            "warnings": state.get("warnings", []) + conflicts,
            "next_action": "planner",  # 回 planner 重规划
            "stage": "fact_check_failed",
        }

    return {"stage": "fact_check_done"}


# ---------------------------------------------------------------------------
# ⑥ Output&DocAgent — 多模态输出 & 文档
# ---------------------------------------------------------------------------


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
            "messages": state.get("messages", []) + [{
                "role": "assistant", "content": content, "type": "clarification",
                "questions": questions, "missing_slots": state.get("missing_slots", []),
            }],
            "stage": "gathering",
            "next_action": "clarify",
            "profile": state.get("profile"),
            "clarification_questions": questions,
            "missing_slots": state.get("missing_slots", []),
            "conversation_sync": state.get("conversation_sync"),
        }

    itinerary = state.get("itinerary", [])
    if not itinerary:
        return {
            "messages": state.get("messages", []) + [{
                "role": "assistant", "content": "抱歉，暂时无法生成行程。",
            }],
            "stage": "completed",
        }

    from core.conversation_state import flatten_profile
    profile_raw = flatten_profile(state.get("profile") or {})
    from schemas import DayPlan, UserProfile
    from planner.core.writer import enrich as enrich_writer

    profile_obj = UserProfile(
        destination=profile_raw.get("destination"),
        travel_days=profile_raw.get("travel_days") or 1,
        interests=profile_raw.get("interests") or [],
        food_preferences=profile_raw.get("food_preferences") or [],
        pace=profile_raw.get("pace") or "moderate",
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
        "messages": state.get("messages", []) + [{
            "role": "assistant", "content": proposal_text, "type": "itinerary",
            "itinerary": itinerary_enriched, "warnings": state.get("warnings", []),
        }],
        "itinerary": itinerary_enriched,
        "stage": "awaiting_booking",
    }


def _format_simple(itinerary: list[dict], profile: dict) -> str:
    dest = profile.get("destination", "目的地")
    days = len(itinerary)
    lines = [f"# {dest} {days}日游行程方案\n"]
    total = 0
    for day in itinerary:
        lines.append(f"## 第{day.get('day_number','?')}天")
        for act in day.get("activities", []):
            cost = act.get("ticket_price", 0) or act.get("meal_cost", 0) or 0
            total += cost
            time_str = f"{act.get('start_time','')}-{act.get('end_time','')}" if act.get("start_time") else ""
            cost_str = f" — ¥{cost}" if cost else ""
            lines.append(f"  {time_str} {act.get('poi_name','?')}{cost_str}")
    lines.append(f"\n预估总费用: ¥{total}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ⑦ BookingToolAgent — 预订工具搜索 & 汇总
# ---------------------------------------------------------------------------


async def _booking_tool_async(state: dict) -> dict:
    """
    预订 Agent：自动搜索机票/酒店/门票/餐厅，汇总展示。
    所有数据标注 source=mock，待后续接入真实 API。
    """
    itinerary = state.get("itinerary", [])
    profile = state.get("profile") or {}

    if not itinerary:
        return {"stage": "completed"}

    destination = profile.get("destination", "")
    origin = profile.get("origin", "")
    travel_days = len(itinerary)
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
                {"no": f"CA{random.randint(1000,9999)}", "dep": "08:00", "arr": "11:00", "price": round(random.uniform(300, 900))},
                {"no": f"MU{random.randint(1000,9999)}", "dep": "14:00", "arr": "17:00", "price": round(random.uniform(250, 700))},
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
            {"name": f"{destination}舒适酒店", "district": "市中心", "price": round(random.uniform(150, 500)), "rating": round(random.uniform(4.0, 4.8), 1)}
            for _ in range(3)
        ],
    )
    if budget_range:
        daily = budget_range / max(travel_days, 1) * 0.35  # 住宿占35%
        hotels = [h for h in hotels if h["price"] <= daily] or hotels[:1]
    booking["hotels"] = hotels

    # ── 门票 ──
    for day in itinerary:
        for act in day.get("activities", []):
            poi_name = act.get("poi_name", "")
            if poi_name and poi_name not in {t.get("poi_name") for t in booking["tickets"]}:
                booking["tickets"].append({
                    "poi_name": poi_name,
                    "price": act.get("ticket_price") or round(random.uniform(30, 120)),
                    "need_reserve": random.random() > 0.5,
                })

    # ── 餐厅 ──
    food_prefs = profile.get("food_preferences") or profile.get("interests") or []
    restaurant_pool = {
        "成都": [
            {"name": "蜀大侠火锅", "cuisine": "川菜", "per_person": 80, "tags": ["辣", "火锅"]},
            {"name": "陈麻婆豆腐", "cuisine": "川菜", "per_person": 45, "tags": ["麻辣", "经典"]},
            {"name": "龙抄手", "cuisine": "小吃", "per_person": 30, "tags": ["清淡", "面食"]},
            {"name": "大蓉和", "cuisine": "川菜", "per_person": 120, "tags": ["高端", "宴请"]},
        ],
    }
    restaurants = restaurant_pool.get(destination, [
        {"name": f"{destination}本地菜馆", "cuisine": "本地菜", "per_person": 60, "tags": []}
    ])
    booking["restaurants"] = restaurants[:3]

    # ── 格式化 ──
    msg = _format_booking_summary(booking, destination, origin, travel_days)

    return {
        "messages": state.get("messages", []) + [{
            "role": "assistant",
            "content": msg,
            "type": "booking",
            "booking_results": booking,
        }],
        "booking_results": booking,
        "stage": "completed",
    }


def _format_booking_summary(booking: dict, destination: str, origin: str, days: int) -> str:
    """格式化预订摘要为 Markdown 文案。"""
    lines = ["# 📋 预订参考\n"]
    total_est = 0

    # 机票
    if booking["flights"]:
        lines.append("## ✈️ 机票")
        for f in booking["flights"]:
            lines.append(f"- {f['no']} {f['dep']}-{f['arr']}  ¥{f['price']}")
        best = min(booking["flights"], key=lambda x: x["price"])
        total_est += best["price"] * 2  # 往返
        lines.append(f"\n> 推荐 {best['no']} 往返约 ¥{best['price'] * 2}\n")

    # 酒店
    if booking["hotels"]:
        lines.append("## 🏨 酒店")
        for h in booking["hotels"][:3]:
            cost = h["price"] * days
            lines.append(f"- {h['name']}（{h['district']}）⭐{h['rating']}  ¥{h['price']}/晚 × {days}晚 = ¥{cost}")
        if booking["hotels"]:
            best_h = booking["hotels"][0]
            total_est += best_h["price"] * days
            lines.append(f"\n> 推荐 {best_h['name']}，{days}晚约 ¥{best_h['price'] * days}\n")

    # 门票
    if booking["tickets"]:
        lines.append("## 🎫 门票")
        tix_total = 0
        for t in booking["tickets"]:
            reserve = "⚠️需预约" if t.get("need_reserve") else "免预约"
            lines.append(f"- {t['poi_name']}  ¥{t['price']} {reserve}")
            tix_total += t["price"]
        total_est += tix_total
        lines.append(f"\n> 门票合计约 ¥{tix_total}\n")

    # 餐厅
    if booking["restaurants"]:
        lines.append("## 🍜 推荐餐厅")
        for r in booking["restaurants"][:3]:
            tags_str = " · ".join(r.get("tags", []))
            lines.append(f"- {r['name']}（{r['cuisine']}）人均 ¥{r['per_person']} {tags_str}")
        lines.append("")

    # 总预估
    if total_est > 0:
        food_est = 100 * days * 1  # 餐饮每人每天100
        total_est += food_est
        lines.append(f"---\n💰 **预估总费用：约 ¥{total_est:,}**（机票+酒店+门票+餐饮）")
        if origin:
            lines.append(f"\n> ⚠️ 以上为模拟参考价，来源标注 mock，实际请以官方渠道为准")

    return "\n".join(lines)
