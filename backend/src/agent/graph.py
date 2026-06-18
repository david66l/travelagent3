"""
TravelAgent LangGraph StateGraph — 6 Agent 多智能体架构。

架构 (蓝图 0.5):
  ① DemandParserAgent → ② UserMemoryRecallAgent → ③ TravelRetrievalRAGAgent
  → ④ ItineraryPlannerAgent → ⑤ FactCheckAgent
  → HumanInterrupt → ⑥ Output&DocAgent → TripEnd (画像更新)

所有子 Agent 无独立调度权，统一由 Graph 流转驱动。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class TravelAgentState(dict):
    """LangGraph 全局状态。"""
    pass


# ---------------------------------------------------------------------------
# ① DemandParserAgent — 需求解析
# ---------------------------------------------------------------------------


def demand_parser_node(state: dict) -> dict:
    """异步包装：调用现有 IntentRecognitionAgent。"""
    import asyncio

    return asyncio.get_event_loop().run_until_complete(_demand_parser_async(state))


async def _demand_parser_async(state: dict) -> dict:
    from agents.intent_recognition import IntentRecognitionAgent
    from core.conversation_turn import entities_to_patch, flatten_profile
    from schemas import UserProfile

    agent = IntentRecognitionAgent()
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in messages[-10:]
        if m.get("role") and m.get("content")
    ]

    existing_profile = state.get("profile") or {}
    flat = flatten_profile(existing_profile)
    profile_kwargs = {k: v for k, v in flat.items() if v is not None and v != []}
    user_profile = UserProfile(**profile_kwargs) if profile_kwargs else None

    result = await agent.recognize(user_input, history, user_profile)

    merged_profile = dict(existing_profile)
    patch = entities_to_patch(result.user_entities)
    if patch.set:
        merged_profile.update(patch.set)
    if patch.add:
        for k, v in patch.add.items():
            existing = merged_profile.get(k, [])
            merged_profile[k] = list(set((existing if isinstance(existing, list) else []) + v))

    if result.missing_required and result.intent == "generate_itinerary":
        next_action = "clarify"
    elif result.intent in ("generate_itinerary", "modify_itinerary"):
        next_action = "plan"
    else:
        next_action = "respond"

    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "slots": result.user_entities,
        "missing_slots": result.missing_required,
        "clarification_questions": result.clarification_questions,
        "profile": merged_profile,
        "next_action": next_action,
        "stage": "demand_parsed",
    }


# ---------------------------------------------------------------------------
# ② UserMemoryRecallAgent — 用户画像记忆
# ---------------------------------------------------------------------------


def user_memory_node(state: dict) -> dict:
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_user_memory_async(state))


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

    return {"profile": profile, "stage": "memory_loaded"}


# ---------------------------------------------------------------------------
# ③ TravelRetrievalRAGAgent — 知识库检索
# ---------------------------------------------------------------------------


def rag_retrieval_node(state: dict) -> dict:
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_rag_async(state))


async def _rag_async(state: dict) -> dict:
    profile = state.get("profile") or {}
    destination = profile.get("destination", "")
    interests = profile.get("interests") or []

    if not destination:
        return {"knowledge_results": [], "stage": "rag_done"}

    # 从本地库检索 POI
    try:
        from data.repository import repo
        results = await repo.search_attractions(destination, limit=30)
        knowledge = [{"name": r.name, "category": r.category, "price": r.ticket_price, "tags": r.tags} for r in results]
    except Exception:
        knowledge = []

    # 尝试 RAG 向量检索
    try:
        query = f"{destination} {' '.join(interests)} 旅游攻略"
        tips = await repo.search_knowledge(query, city=destination, top_k=3)
        knowledge.extend(tips)
    except Exception:
        pass

    return {"knowledge_results": knowledge, "stage": "rag_done"}


# ---------------------------------------------------------------------------
# ④ ItineraryPlannerAgent — 行程规划求解
# ---------------------------------------------------------------------------


def itinerary_planner_node(state: dict) -> dict:
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_planner_async(state))


async def _planner_async(state: dict) -> dict:
    profile_raw = state.get("profile") or {}
    slots = state.get("slots") or {}
    knowledge = state.get("knowledge_results") or []

    merged = {**profile_raw}
    for k, v in slots.items():
        if v is not None and v != []:
            merged[k] = v

    destination = merged.get("destination", "")
    if not destination:
        return {"next_action": "clarify", "warnings": ["Missing destination"]}

    from schemas import ScoredPOI, UserProfile, Location
    from planner.core.or_scheduler import solve_itinerary_or
    from planner.core.enhancements import (
        OptimizationWeights, PersonaRules, feasibility_check, avoid_peak_hours
    )

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

    # 构建 POI 列表
    pois = [
        ScoredPOI(
            name=k.get("name", f"POI-{i}"),
            category=k.get("category", "attraction"),
            score=k.get("score", 0.5),
            ticket_price=k.get("price"),
            tags=k.get("tags") or [],
            location=Location(lat=0, lng=0) if not k.get("lat") else Location(lat=k["lat"], lng=k["lng"]),
        )
        for i, k in enumerate(knowledge[:30]) if k.get("name")
    ]

    if not pois:
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

    if not pois:
        return {"next_action": "respond", "warnings": ["No POIs found"]}

    # OR-Tools 求解
    schedule = solve_itinerary_or(pois, profile_obj)

    # 错峰调度
    schedule = avoid_peak_hours(schedule, pois)

    itinerary_json = [day.model_dump() for day in schedule]

    return {
        "itinerary": itinerary_json,
        "warnings": conflicts,
        "next_action": "fact_check",
        "stage": "planned",
    }


# ---------------------------------------------------------------------------
# ⑤ FactCheckAgent — 事实校验 & 幻觉拦截
# ---------------------------------------------------------------------------


def fact_check_node(state: dict) -> dict:
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_fact_check_async(state))


async def _fact_check_async(state: dict) -> dict:
    itinerary = state.get("itinerary", [])
    if not itinerary:
        return {"stage": "fact_check_done"}

    conflicts = []
    try:
        from core.database import async_session_maker

        async with async_session_maker() as db:
            for day in itinerary:
                for act in day.get("activities", []):
                    poi_name = act.get("poi_name", "")
                    if not poi_name:
                        continue
                    # 查数据库校验
                    row = await db.fetchrow(
                        "SELECT ticket_price, open_time, close_time, status FROM attractions WHERE name = $1",
                        poi_name,
                    )
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


def output_doc_node(state: dict) -> dict:
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_output_async(state))


async def _output_async(state: dict) -> dict:
    next_action = state.get("next_action", "respond")

    if next_action == "clarify":
        questions = state.get("clarification_questions") or ["请问您想去哪个目的地？", "计划玩几天？"]
        return {
            "messages": state.get("messages", []) + [{
                "role": "assistant", "content": questions[0], "type": "clarification",
                "questions": questions, "missing_slots": state.get("missing_slots", []),
            }],
            "stage": "completed",
        }

    itinerary = state.get("itinerary", [])
    if not itinerary:
        return {
            "messages": state.get("messages", []) + [{
                "role": "assistant", "content": "抱歉，暂时无法生成行程。",
            }],
            "stage": "completed",
        }

    profile_raw = state.get("profile") or {}
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


# ---------------------------------------------------------------------------
# ⑦ BookingToolAgent — 预订工具搜索 & 汇总
# ---------------------------------------------------------------------------


def booking_tool_node(state: dict) -> dict:
    import asyncio
    return asyncio.get_event_loop().run_until_complete(_booking_tool_async(state))


async def _booking_tool_async(state: dict) -> dict:
    """
    预订 Agent：自动搜索机票/酒店/门票/餐厅，汇总展示。
    所有数据标注 source=mock，待后续接入真实 API。
    """
    import random

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
# Graph construction
# ---------------------------------------------------------------------------


def build_graph(checkpointer: Optional[PostgresSaver] = None) -> StateGraph:
    builder = StateGraph(dict)

    builder.add_node("demand_parser", demand_parser_node)
    builder.add_node("user_memory", user_memory_node)
    builder.add_node("rag_retrieval", rag_retrieval_node)
    builder.add_node("itinerary_planner", itinerary_planner_node)
    builder.add_node("fact_check", fact_check_node)
    builder.add_node("output_doc", output_doc_node)
    builder.add_node("booking_tool", booking_tool_node)

    builder.set_entry_point("demand_parser")

    def route_after_parser(state: dict) -> str:
        if state.get("next_action") == "clarify":
            return "output_doc"
        return "user_memory"

    builder.add_conditional_edges("demand_parser", route_after_parser, {
        "output_doc": "output_doc",
        "user_memory": "user_memory",
    })

    builder.add_edge("user_memory", "rag_retrieval")
    builder.add_edge("rag_retrieval", "itinerary_planner")

    def route_after_planner(state: dict) -> str:
        if state.get("next_action") == "fact_check":
            return "fact_check"
        return "output_doc"

    builder.add_conditional_edges("itinerary_planner", route_after_planner, {
        "fact_check": "fact_check",
        "output_doc": "output_doc",
    })

    def route_after_fact_check(state: dict) -> str:
        if state.get("next_action") == "planner":
            return "itinerary_planner"  # 冲突 → 重规划
        return "output_doc"

    builder.add_conditional_edges("fact_check", route_after_fact_check, {
        "itinerary_planner": "itinerary_planner",
        "output_doc": "output_doc",
    })

    # output → booking_tool → memory write-back (trip_end)
    builder.add_edge("output_doc", "booking_tool")
    builder.add_edge("booking_tool", "user_memory")
    builder.add_edge("user_memory", END)

    if checkpointer:
        return builder.compile(checkpointer=checkpointer)
    return builder.compile()


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_graph: Optional[StateGraph] = None


async def get_graph() -> StateGraph:
    global _graph
    if _graph is not None:
        return _graph

    from core.settings import settings

    db_url = settings.database_url
    if db_url.startswith("postgresql+asyncpg://"):
        db_url = db_url.replace("postgresql+asyncpg://", "postgresql://")

    try:
        checkpointer = PostgresSaver.from_conn_string(db_url)
        await checkpointer.setup()
        _graph = build_graph(checkpointer=checkpointer)
        logger.info("LangGraph 6-Agent compiled with PostgresSaver")
    except Exception as exc:
        logger.warning("PostgresSaver unavailable (%s), using in-memory", exc)
        _graph = build_graph(checkpointer=None)

    return _graph
