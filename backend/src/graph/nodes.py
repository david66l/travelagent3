"""LangGraph async nodes for the TravelAgent orchestration layer."""

from __future__ import annotations

import logging
import json
import time
from typing import Any

from core.langsmith_trace import traceable_step
from core.metrics import record_fallback, record_retrieval_latency, record_solve_latency
from graph.exceptions import with_error_handling

logger = logging.getLogger(__name__)


@with_error_handling("profile")
async def profile_node(state: dict[str, Any]) -> dict[str, Any]:
    """User profile recall and memory conflict resolution."""
    from graph.node_impl import _user_memory_async

    result = await _user_memory_async(state)
    from agentic.runtime import initialize_agent_ledger
    from core.settings import settings

    projected_state = {**state, **result}
    result.update(initialize_agent_ledger(projected_state, mode=settings.agentic_policy_mode))
    return result


@with_error_handling("agent_loop")
async def agent_loop_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the bounded Agent Loop only when policy_mode explicitly selects it."""
    from agentic.integration import run_agent_branch

    return await run_agent_branch(state)


@with_error_handling("retrieve")
async def retrieve_node(state: dict[str, Any]) -> dict[str, Any]:
    """RAG POI retrieval."""
    from graph.node_impl import _rag_async

    start = time.perf_counter()
    result = await _rag_async(state)
    record_retrieval_latency(time.perf_counter() - start, source="rag")
    return result


@with_error_handling("weather_check")
async def weather_check_node(state: dict[str, Any]) -> dict[str, Any]:
    """Fetch weather for destination + dates BEFORE planning."""
    from graph.node_impl import _weather_check_async

    return await _weather_check_async(state)


@with_error_handling("plan")
async def plan_node(state: dict[str, Any]) -> dict[str, Any]:
    """Solve the itinerary and produce a draft.

    Confirmation is no longer driven by a stage flag here — the dedicated
    ``confirm_gate`` node pauses (LangGraph interrupt) and decides what comes
    next (enrich / modify / re-solve). This node only does the expensive solve,
    so a resume never re-runs it.
    """
    from graph.node_impl import _planner_async

    start = time.perf_counter()
    result = await _planner_async(state)
    strategy = (result.get("solve_status") or "default") if isinstance(result, dict) else "default"
    record_solve_latency(time.perf_counter() - start, strategy=str(strategy))

    if isinstance(result, dict):
        # Solver returns "planned"; frontend listens for draft_ready to show the
        # structured itinerary immediately (before enrich/polish in output).
        result["stage"] = "draft_ready"
    return result


@with_error_handling("confirm_gate")
async def confirm_gate_node(state: dict[str, Any]) -> dict[str, Any]:
    """Pause for the user to confirm / modify / reject the draft itinerary.

    Uses a LangGraph dynamic ``interrupt``. The runner resumes with
    ``Command(resume={"action": "confirm"|"modify"|"reject", "change": {...}?})``.
    """
    from langgraph.types import interrupt

    decision = interrupt(
        {
            "type": "awaiting_confirm",
            "itinerary": state.get("itinerary"),
        }
    )
    if isinstance(decision, str):
        decision = {"action": decision}
    decision = decision or {}
    action = decision.get("action", "confirm")

    if action == "modify":
        return {
            "confirm_decision": "modify",
            "pending_change": decision.get("change"),
            "stage": "modifying",
        }
    if action == "reject":
        # Re-solve from scratch; treat the next plan as a fresh draft.
        return {"confirm_decision": None, "stage": "rejected"}
    # confirm → proceed to deep enrichment
    agent_patch: dict[str, Any] = {}
    if state.get("policy_mode") == "agent" and state.get("agent_ledger"):
        from agentic.runtime import confirm_agent_ledger

        ledger, completion = confirm_agent_ledger(state["agent_ledger"])
        agent_patch = {
            "agent_ledger": ledger.model_dump(mode="json"),
            "agent_status": "finished",
            "termination_reason": "validated_finish",
            "completion_decision": completion.model_dump(mode="json"),
        }
    return {
        "confirm_decision": "confirm",
        "stage": "confirmed",
        "next_action": "enrich",
        **agent_patch,
    }


@with_error_handling("factcheck")
async def factcheck_node(state: dict[str, Any]) -> dict[str, Any]:
    """Fact checking against structured data."""
    from graph.node_impl import _fact_check_async

    return await _fact_check_async(state)


@traceable_step("planning/hallucination_check", run_type="chain")
def _trace_hallucination_detect(state: dict[str, Any]) -> Any:
    from agents.hallucination_detector import HallucinationDetectionAgent

    return HallucinationDetectionAgent.detect(state)


@with_error_handling("hallucination")
async def hallucination_check_node(state: dict[str, Any]) -> dict[str, Any]:
    """Hallucination detection for generated itinerary."""
    try:
        result = _trace_hallucination_detect(state)
    except Exception as exc:
        logger.warning("Hallucination check failed: %s", exc)
        return {"hallucination_result": {"passed": True}}

    # warnings is a reducer field → return only this node's new suggestions.
    new_warnings = (
        list(result.improvement_suggestions) if result and result.improvement_suggestions else []
    )

    return {
        "hallucination_result": result.model_dump(),
        "warnings": new_warnings,
    }


@traceable_step("planning/tool_call", run_type="chain")
async def _trace_execute_tools(
    tool_calls: list[dict[str, Any]],
    guard_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    from tools.tool_executor import tool_executor

    return await tool_executor.execute(tool_calls, guard_context=guard_context)


@with_error_handling("tool_call")
async def tool_call_node(state: dict[str, Any]) -> dict[str, Any]:
    """Execute pending tool calls and attach results to state."""
    tool_calls = state.get("pending_tool_calls") or _build_default_tool_calls(state)
    if not tool_calls:
        return {"tool_results": [], "stage": "tools_executed"}

    from core.conversation_state import flatten_profile

    profile = flatten_profile(state.get("profile") or {})
    grounded_values: dict[str, set[str]] = {}
    if profile.get("destination"):
        grounded_values["city"] = {str(profile["destination"])}
    results = await _trace_execute_tools(
        tool_calls,
        guard_context={
            "allowed_tools": state.get("allowed_tools"),
            "grounded_values": grounded_values,
        },
    )
    for tr in results:
        result_obj = tr.get("result") or {}
        if result_obj.get("is_fallback"):
            record_fallback(
                source=tr.get("name", "unknown"),
                reason=result_obj.get("fallback_reason", "fallback"),
            )
    return {
        "tool_results": results,
        "pending_tool_calls": [],
        "stage": "tools_executed",
    }


def _build_default_tool_calls(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a bounded, useful enrichment batch from the confirmed draft.

    The former implementation emitted four calls for almost every POI (often
    60+ sequential calls) and read only a flat profile, while the real profile
    is nested. This version keeps the latency predictable and passes dates /
    budgets into the tools that can use them.
    """
    from core.conversation_state import flatten_profile

    profile = flatten_profile(state.get("profile") or {})
    city = profile.get("destination") or ""
    itinerary = state.get("itinerary") or []
    tool_calls: list[dict[str, Any]] = []
    call_id = 0

    def add(name: str, arguments: dict[str, Any]) -> None:
        nonlocal call_id
        call_id += 1
        tool_calls.append(
            {
                "id": f"tool_{call_id}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )

    if city:
        dates = str(profile.get("travel_dates") or "")
        start_date = dates[:10] if dates else None
        add("get_weather", {"city": city, **({"date": start_date} if start_date else {})})
        budget = float(profile.get("budget_range") or 0)
        nights = max(len(itinerary) - 1, 1)
        hotel_budget = budget * 0.35 / nights if budget else None
        add("find_hotels", {"city": city, "budget_per_night": hotel_budget})
        add("find_restaurants", {"city": city})
        add("get_emergency_services", {"city": city})

    seen_pois: set[str] = set()
    for day in itinerary:
        activities = day.get("activities") or []
        attractions = [a for a in activities if a.get("category") == "attraction"][:2]
        for act in attractions:
            poi = act.get("poi_name") or act.get("name") or ""
            if not poi or poi in seen_pois:
                continue
            seen_pois.add(poi)
            add("check_reservation", {"poi_name": poi, "city": city})
            add("get_poi_detail", {"poi_name": poi, "city": city})

        if len(activities) >= 2:
            origin = activities[0].get("poi_name") or activities[0].get("name") or "酒店"
            destination = activities[-1].get("poi_name") or activities[-1].get("name") or "酒店"
            add("get_route", {"origin": origin, "destination": destination, "city": city})

    return tool_calls


@traceable_step("planning/content_safety", run_type="chain")
def _trace_content_safety(state: dict[str, Any]) -> Any:
    from agents.content_safety import ContentSafetyEngine

    return ContentSafetyEngine.check(state)


@traceable_step("planning/output_artifacts", run_type="chain")
async def _trace_output_format(
    *,
    proposal_text: str,
    itinerary: list[Any],
    city: str,
    session_id: str,
    on_token: Any = None,
) -> dict[str, Any]:
    from agents.output_format import output_format_agent

    return await output_format_agent.format(
        proposal_text=proposal_text,
        itinerary=itinerary,
        city=city,
        session_id=session_id,
        on_token=on_token,
    )


@with_error_handling("output")
async def output_node(state: dict[str, Any]) -> dict[str, Any]:
    """Output formatting (Markdown / clarification) + multi-modal export.

    For the itinerary path we stream the prose in real-time from a single LLM
    call while the structured enrichment (card recommendation reasons) runs
    concurrently — so the chat starts streaming immediately after planning
    instead of blocking on a sequential enrich → polish pass.
    """
    import asyncio

    from api.chat_runtime import publish_live_stage, publish_token
    from agents.output_format import output_format_agent
    from core.conversation_state import flatten_profile
    from graph.node_impl import _output_async

    job_id = state.get("job_id")
    session_id = state.get("session_id")
    itinerary = state.get("itinerary", []) or []
    # Some callers/tests restore an itinerary-only checkpoint created before
    # ``next_action`` became part of the state schema.  Treat an existing
    # itinerary as the itinerary path instead of accidentally formatting it as
    # a generic chat response.
    next_action = state.get("next_action") or ("fact_check" if itinerary else "respond")

    async def _on_token(chunk: str) -> None:
        if job_id:
            await publish_token(str(job_id), chunk)
        elif session_id:
            from api.chat_runtime import manager

            await manager.send_json(session_id, {"type": "token", "chunk": chunk})

    # Non-itinerary outputs (clarify / respond / infeasible / empty): single pass.
    if next_action in ("clarify", "respond", "infeasible") or not itinerary:
        return await _output_async(state)

    # ---- Itinerary path -------------------------------------------------- #
    # Content safety first (cheap, sync) so we never stream a blocked plan.
    safety_dict: dict[str, Any] | None = None
    try:
        safety = _trace_content_safety(state)
        safety_dict = safety.model_dump()
    except Exception as exc:
        logger.warning("Content safety check failed: %s", exc)
        safety = None
    if safety and not safety.passed:
        reasons = "；".join(safety.improvement_suggestions) or "内容安全校验未通过"
        return {
            "messages": (state.get("messages") or [])
            + [
                {
                    "role": "assistant",
                    "content": f"内容安全拦截：{reasons}。请调整需求后重试。",
                    "type": "safety_blocked",
                }
            ],
            "stage": "safety_blocked",
            "safety_result": safety_dict,
        }

    # Progress line only — the blinking caret appears on the first real token.
    if session_id or job_id:
        try:
            await publish_live_stage(session_id=session_id, job_id=job_id, stage="writing")
        except Exception as exc:
            logger.debug("Live stage push skipped: %s", exc)

    profile_raw = flatten_profile(state.get("profile") or {})
    city = profile_raw.get("destination") or ""
    sid = session_id or state.get("user_id") or "default"
    on_token = _on_token if (job_id or session_id) else None

    # Kick off structured enrichment (card reasons) in the background, then
    # stream the prose immediately so there is no dead wait.
    enrich_task = asyncio.create_task(_output_async(state))
    try:
        polished = await output_format_agent.stream_markdown(
            itinerary, profile_raw, on_token=on_token
        )
    except Exception as exc:
        logger.warning("Streaming prose failed: %s", exc)
        polished = ""

    try:
        base = await enrich_task
    except Exception as exc:
        logger.warning("Itinerary enrichment failed: %s", exc)
        base = {
            "messages": (state.get("messages") or [])
            + [
                {
                    "role": "assistant",
                    "content": polished,
                    "type": "itinerary",
                    "itinerary": itinerary,
                    "warnings": state.get("warnings", []),
                }
            ],
            "itinerary": itinerary,
            "stage": "awaiting_booking",
        }

    if safety_dict:
        base["safety_result"] = safety_dict

    enriched_itin = base.get("itinerary") or itinerary
    final_md = polished or (base["messages"][-1].get("content", "") if base.get("messages") else "")

    try:
        artifacts = await output_format_agent.build_artifacts(final_md, enriched_itin, city, sid)
    except Exception as exc:
        logger.warning("Artifact build failed: %s", exc)
        artifacts = {"pdf": None, "excel": None, "map": None}

    base["output_markdown"] = final_md
    base["output_pdf_url"] = artifacts.get("pdf")
    base["output_excel_url"] = artifacts.get("excel")
    base["output_map_url"] = artifacts.get("map")
    if base.get("messages"):
        base["messages"][-1]["content"] = final_md
        base["messages"][-1]["output_pdf_url"] = artifacts.get("pdf")
        base["messages"][-1]["output_excel_url"] = artifacts.get("excel")
        base["messages"][-1]["output_map_url"] = artifacts.get("map")

    return base


@with_error_handling("booking")
async def booking_node(state: dict[str, Any]) -> dict[str, Any]:
    """Booking tool aggregation (mock data in MVP)."""
    from graph.node_impl import _booking_tool_async

    return await _booking_tool_async(state)


@traceable_step("planning/apply_change", run_type="chain")
def _trace_apply_single_change(
    change: dict[str, Any],
    itinerary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    action = change.get("action")
    day_number = change.get("day_number")
    poi_id = change.get("poi_id")

    new_itinerary = [dict(d) for d in itinerary]
    for day in new_itinerary:
        if day_number and day.get("day_number") != day_number:
            continue
        acts = list(day.get("activities", []))
        if action == "remove" and poi_id:
            day["activities"] = [a for a in acts if a.get("poi_name") != poi_id]
        elif action == "replace" and poi_id and change.get("new_poi"):
            day["activities"] = [
                {**a, **change["new_poi"]} if a.get("poi_name") == poi_id else a for a in acts
            ]
        elif action == "add" and change.get("new_poi"):
            day["activities"] = acts + [change["new_poi"]]
        elif action == "reorder" and change.get("order"):
            order = {name: i for i, name in enumerate(change["order"])}
            day["activities"] = sorted(acts, key=lambda a: order.get(a.get("poi_name"), len(order)))
    return new_itinerary


@with_error_handling("apply_single_change")
async def apply_single_change_node(state: dict[str, Any]) -> dict[str, Any]:
    """Apply a single Human-in-the-loop modification and replan locally."""
    change = state.get("pending_change")
    itinerary = (
        state.get("itinerary") or state.get("itinerary_final") or state.get("itinerary_draft") or []
    )
    if not change or not itinerary:
        return {"stage": "change_applied"}

    action = change.get("action")
    if action in {"set_budget", "set_pace", "change_days"}:
        from core.conversation_state import flatten_profile, merge_profile
        from schemas import ProfilePatch

        flat = flatten_profile(state.get("profile") or {})
        slots = dict(state.get("slots") or {})
        updates: dict[str, Any] = {}
        if action == "set_budget":
            value = float(change.get("value") or 0)
            if value > 0:
                updates["budget_range"] = value
                slots["total_budget"] = value
        elif action == "set_pace":
            raw = str(change.get("value") or "").strip()
            pace = {"轻松": "relaxed", "适中": "moderate", "紧凑": "intensive"}.get(raw, raw)
            if pace in {"relaxed", "moderate", "intensive"}:
                updates["pace"] = pace
                slots["pace"] = pace
        else:
            current = int(flat.get("travel_days") or len(itinerary) or 1)
            days = max(1, min(30, current + int(change.get("delta") or 0)))
            updates["travel_days"] = days
            slots["travel_days"] = days

        profile = merge_profile(state.get("profile") or {}, ProfilePatch(set=updates))
        return {
            "profile": profile,
            "slots": slots,
            "pending_change": None,
            "confirm_decision": None,
            "stage": "constraints_changed",
            "next_action": "planner",
        }

    new_itinerary = _trace_apply_single_change(change, itinerary)

    # Clear the consumed change so a later confirm does not re-apply it.
    return {
        "itinerary": new_itinerary,
        "pending_change": None,
        "confirm_decision": "modify",
        "stage": "change_applied",
        "next_action": "fact_check",
    }


@traceable_step("planning/replan_local", run_type="chain")
def _trace_replan_local(
    event: dict[str, Any],
    itinerary: list[dict[str, Any]],
    poi_candidates: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    import copy
    import re

    etype = event.get("type")
    poi = event.get("poi")
    detail = event.get("detail", "")
    new_itinerary = copy.deepcopy(itinerary)

    def _clock(value: str | None) -> int | None:
        if not value or ":" not in value:
            return None
        try:
            hour, minute = value.split(":", 1)
            return int(hour) * 60 + int(minute)
        except (TypeError, ValueError):
            return None

    def _hhmm(minutes: int) -> str:
        minutes = max(0, min(minutes, 23 * 60 + 59))
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    if etype == "closure" and poi:
        used = {
            a.get("poi_name")
            for d in new_itinerary
            for a in d.get("activities", [])
            if a.get("poi_name")
        }
        candidates = [
            c
            for c in (poi_candidates or [])
            if (c.get("spot_name") or c.get("name")) not in used
            and (c.get("spot_name") or c.get("name")) != poi
            and c.get("category", "attraction") == "attraction"
        ]
        replacement_name = None
        for day in new_itinerary:
            acts = list(day.get("activities", []))
            for index, activity in enumerate(acts):
                if activity.get("poi_name") != poi:
                    continue
                if candidates:
                    candidate = candidates.pop(0)
                    replacement_name = candidate.get("spot_name") or candidate.get("name")
                    replacement = dict(activity)
                    replacement.update(
                        {
                            "poi_name": replacement_name,
                            "ticket_price": candidate.get("ticket_price") or 0,
                            "tags": candidate.get("tags") or activity.get("tags") or [],
                            "recommendation_reason": f"作为 {poi} 临时关闭后的同城替代安排。",
                        }
                    )
                    if candidate.get("lat") is not None and candidate.get("lng") is not None:
                        replacement["location"] = {
                            "lat": candidate.get("lat"),
                            "lng": candidate.get("lng"),
                            "address": candidate.get("address"),
                        }
                    acts[index] = replacement
                else:
                    acts.pop(index)
                day["activities"] = acts
                break
        if replacement_name:
            note = f"{poi} 临时关闭，已在原时间段替换为 {replacement_name}，请确认调整后的方案。"
        else:
            note = f"{poi} 临时关闭，暂无可靠替代景点，已移除该活动并保留空档。"
    elif etype == "weather":
        indoor_words = ("博物馆", "美术馆", "科技馆", "室内", "商场", "餐", "酒店")
        changed_days = 0
        for day in new_itinerary:
            activities = list(day.get("activities", []))
            if len(activities) < 2:
                continue
            original_names = [a.get("poi_name") for a in activities]
            slots = [(a.get("start_time"), a.get("end_time")) for a in activities]

            def is_indoor(activity: dict[str, Any]) -> bool:
                text = " ".join(
                    [str(activity.get("poi_name") or ""), *map(str, activity.get("tags") or [])]
                )
                return activity.get("category") in {"restaurant", "hotel"} or any(
                    word in text for word in indoor_words
                )

            # For afternoon rain, put outdoor visits first and sheltered stops
            # later; for generic bad weather, prefer sheltered stops first.
            afternoon_rain = "下午" in detail or "午后" in detail
            reordered = sorted(activities, key=is_indoor, reverse=not afternoon_rain)
            if [a.get("poi_name") for a in reordered] == original_names:
                continue
            for activity, (start, end) in zip(reordered, slots):
                activity["start_time"], activity["end_time"] = start, end
            day["activities"] = reordered
            changed_days += 1
        note = (
            f"天气变化（{detail}），已将户外与室内活动按天气时段重新排序，共调整 {changed_days} 天。"
            if changed_days
            else f"天气变化（{detail}），当前安排没有可安全互换的室内外活动，请人工确认。"
        )
    elif etype == "delay":
        match = re.search(r"(\d+(?:\.\d+)?)\s*(小时|分钟)", str(detail))
        delay_min = 60
        if match:
            amount = float(match.group(1))
            delay_min = int(amount * 60) if match.group(2) == "小时" else int(amount)
        target_day = int(event.get("day_number") or 1)
        shifted = 0
        for day in new_itinerary:
            if int(day.get("day_number") or 0) != target_day:
                continue
            for activity in day.get("activities", []):
                start = _clock(activity.get("start_time"))
                end = _clock(activity.get("end_time"))
                if start is not None:
                    activity["start_time"] = _hhmm(start + delay_min)
                if end is not None:
                    activity["end_time"] = _hhmm(end + delay_min)
                shifted += 1
        note = f"行程延误（{detail}），已将第 {target_day} 天的 {shifted} 个活动顺延 {delay_min} 分钟。"
    else:
        note = "收到行程外部事件，已据此更新提示。"
    return new_itinerary, note


@with_error_handling("replan_local")
async def replan_local_node(state: dict[str, Any]) -> dict[str, Any]:
    """In-trip local replan in response to an external event (weather/closure/delay).

    Adjusts the existing itinerary in place rather than re-solving from scratch,
    then flows to output → confirm_gate so the user can accept the adjustment.
    """
    event = state.get("external_event") or {}
    itinerary = (
        state.get("itinerary") or state.get("itinerary_final") or state.get("itinerary_draft") or []
    )
    if not itinerary:
        # No prior plan to adjust → fall back to a fresh solve.
        return await plan_node(state)

    new_itinerary, note = _trace_replan_local(event, itinerary, state.get("poi_candidates") or [])

    return {
        "itinerary": new_itinerary,
        "external_event": None,
        "confirm_decision": None,
        "warnings": [note],
        "stage": "replanned",
    }


@with_error_handling("error_handler")
async def error_handler_node(state: dict[str, Any]) -> dict[str, Any]:
    """Terminal error handler: produce user-facing message."""
    return {
        "messages": (state.get("messages") or [])
        + [
            {
                "role": "assistant",
                "content": state.get("error_message") or "系统处理异常，请稍后重试。",
                "type": "error",
            }
        ],
        "stage": "error",
    }


@with_error_handling("human_interrupt")
async def human_interrupt_node(state: dict[str, Any]) -> dict[str, Any]:
    """Pause graph for human confirmation / modification."""
    return {
        "stage": "awaiting_human",
        "messages": (state.get("messages") or [])
        + [
            {
                "role": "assistant",
                "content": "请确认或修改当前行程方案。",
                "type": "human_interrupt",
            }
        ],
    }
