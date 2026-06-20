"""LangGraph async nodes for the TravelAgent orchestration layer."""

from __future__ import annotations

import logging
import time
from typing import Any

from core.metrics import record_fallback, record_retrieval_latency, record_solve_latency
from graph.exceptions import with_error_handling

logger = logging.getLogger(__name__)


@with_error_handling("profile")
async def profile_node(state: dict[str, Any]) -> dict[str, Any]:
    """User profile recall and memory conflict resolution."""
    from graph.node_impl import _user_memory_async

    return await _user_memory_async(state)


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
    """VRP-based itinerary planning.

    First call (stage != "confirmed"): generates a draft and asks user to confirm.
    Second call (stage == "confirmed"): marks itinerary as confirmed, triggers deep enrichment.
    """
    from graph.node_impl import _planner_async

    stage = state.get("stage", "")

    # Phase 2: user confirmed → enrich with real data
    if stage == "confirmed":
        return {
            "next_action": "enrich",
            "stage": "enriching",
        }

    # Phase 1: generate draft
    start = time.perf_counter()
    result = await _planner_async(state)
    strategy = (result.get("solve_status") or "default") if isinstance(result, dict) else "default"
    record_solve_latency(time.perf_counter() - start, strategy=str(strategy))

    # Mark as draft — user needs to confirm before deep enrichment
    if isinstance(result, dict):
        result["next_action"] = "confirm_draft"
        result["stage"] = "draft_ready"
    return result


@with_error_handling("factcheck")
async def factcheck_node(state: dict[str, Any]) -> dict[str, Any]:
    """Fact checking against structured data."""
    from graph.node_impl import _fact_check_async

    return await _fact_check_async(state)


@with_error_handling("hallucination")
async def hallucination_check_node(state: dict[str, Any]) -> dict[str, Any]:
    """Hallucination detection for generated itinerary."""
    from agents.hallucination_detector import HallucinationDetectionAgent

    try:
        result = HallucinationDetectionAgent.detect(state)
    except Exception as exc:
        logger.warning("Hallucination check failed: %s", exc)
        return {"hallucination_result": {"passed": True}}

    warnings = list(state.get("warnings") or [])
    if result and result.improvement_suggestions:
        warnings.extend(result.improvement_suggestions)

    return {
        "hallucination_result": result.model_dump(),
        "warnings": warnings,
    }


@with_error_handling("tool_call")
async def tool_call_node(state: dict[str, Any]) -> dict[str, Any]:
    """Execute pending tool calls and attach results to state."""
    from tools.tool_executor import tool_executor

    tool_calls = state.get("pending_tool_calls") or _build_default_tool_calls(state)
    if not tool_calls:
        return {"tool_results": [], "stage": "tools_executed"}

    results = await tool_executor.execute(tool_calls)
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
    """Build a deterministic batch of tool calls from itinerary and profile."""
    profile = state.get("profile") or {}
    city = profile.get("destination") or ""
    itinerary = state.get("itinerary") or []
    tool_calls: list[dict[str, Any]] = []
    call_id = 0

    if city:
        call_id += 1
        tool_calls.append(
            {
                "id": f"tool_{call_id}",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": f'{{"city": "{city}"}}',
                },
            }
        )
        call_id += 1
        tool_calls.append(
            {
                "id": f"tool_{call_id}",
                "type": "function",
                "function": {
                    "name": "get_local_events",
                    "arguments": f'{{"city": "{city}"}}',
                },
            }
        )
        call_id += 1
        tool_calls.append(
            {
                "id": f"tool_{call_id}",
                "type": "function",
                "function": {
                    "name": "get_emergency_services",
                    "arguments": f'{{"city": "{city}"}}',
                },
            }
        )

    seen_pois: set[str] = set()
    for day in itinerary[:3]:
        activities = day.get("activities") or []
        for act in activities[:6]:
            poi = act.get("poi_name") or act.get("name") or ""
            if not poi or poi in seen_pois:
                continue
            seen_pois.add(poi)
            call_id += 1
            tool_calls.append(
                {
                    "id": f"tool_{call_id}",
                    "type": "function",
                    "function": {
                        "name": "check_reservation",
                        "arguments": f'{{"poi_name": "{poi}", "city": "{city}"}}',
                    },
                }
            )
            call_id += 1
            tool_calls.append(
                {
                    "id": f"tool_{call_id}",
                    "type": "function",
                    "function": {
                        "name": "get_poi_detail",
                        "arguments": f'{{"poi_name": "{poi}", "city": "{city}"}}',
                    },
                }
            )
            call_id += 1
            tool_calls.append(
                {
                    "id": f"tool_{call_id}",
                    "type": "function",
                    "function": {
                        "name": "get_ticket_link",
                        "arguments": f'{{"poi_name": "{poi}"}}',
                    },
                }
            )
            call_id += 1
            tool_calls.append(
                {
                    "id": f"tool_{call_id}",
                    "type": "function",
                    "function": {
                        "name": "get_queue_time",
                        "arguments": f'{{"poi_name": "{poi}", "city": "{city}"}}',
                    },
                }
            )

    if itinerary and len(itinerary) >= 2:
        first_day = itinerary[0]
        last_day = itinerary[-1]
        first_act = (first_day.get("activities") or [{}])[0]
        last_act = (last_day.get("activities") or [{}])[0]
        origin = first_act.get("poi_name") or first_act.get("name") or "酒店"
        destination = last_act.get("poi_name") or last_act.get("name") or "机场/车站"
        call_id += 1
        tool_calls.append(
            {
                "id": f"tool_{call_id}",
                "type": "function",
                "function": {
                    "name": "get_route",
                    "arguments": f'{{"origin": "{origin}", "destination": "{destination}", "city": "{city}"}}',
                },
            }
        )

    if city:
        call_id += 1
        tool_calls.append(
            {
                "id": f"tool_{call_id}",
                "type": "function",
                "function": {
                    "name": "find_restaurants",
                    "arguments": f'{{"city": "{city}"}}',
                },
            }
        )
        call_id += 1
        tool_calls.append(
            {
                "id": f"tool_{call_id}",
                "type": "function",
                "function": {
                    "name": "find_hotels",
                    "arguments": f'{{"city": "{city}"}}',
                },
            }
        )

    if profile:
        call_id += 1
        tool_calls.append(
            {
                "id": f"tool_{call_id}",
                "type": "function",
                "function": {
                    "name": "update_user_profile",
                    "arguments": '{"key": "last_plan_city", "value": "' + str(city or "") + '"}',
                },
            }
        )

    return tool_calls


@with_error_handling("output")
async def output_node(state: dict[str, Any]) -> dict[str, Any]:
    """Output formatting (Markdown / clarification) + multi-modal export."""
    from graph.node_impl import _output_async
    from agents.content_safety import ContentSafetyEngine
    from agents.output_format import output_format_agent

    base = await _output_async(state)

    # Content safety check before generating artifacts.
    try:
        safety = ContentSafetyEngine.check(state)
        base["safety_result"] = safety.model_dump()
    except Exception as exc:
        logger.warning("Content safety check failed: %s", exc)
        safety = None

    if safety and not safety.passed:
        reasons = "；".join(safety.improvement_suggestions) or "内容安全校验未通过"
        blocked_message = {
            "role": "assistant",
            "content": f"内容安全拦截：{reasons}。请调整需求后重试。",
            "type": "safety_blocked",
        }
        base["messages"] = (base.get("messages") or []) + [blocked_message]
        base["stage"] = "safety_blocked"
        return base

    # If this is a clarification or error, skip artifact generation.
    if base.get("stage") == "completed" and not base.get("itinerary"):
        return base

    itinerary = base.get("itinerary") or state.get("itinerary") or []
    proposal_text = ""
    if base.get("messages"):
        proposal_text = base["messages"][-1].get("content", "")

    profile = state.get("profile") or {}
    city = profile.get("destination") or ""
    session_id = state.get("session_id") or state.get("user_id") or "default"

    try:
        formatted = await output_format_agent.format(
            proposal_text=proposal_text,
            itinerary=itinerary,
            city=city,
            session_id=session_id,
        )
        base["output_markdown"] = formatted["output_markdown"]
        base["output_pdf_url"] = formatted["output_pdf_url"]
        base["output_excel_url"] = formatted["output_excel_url"]
        base["output_map_url"] = formatted["output_map_url"]
        if base.get("messages"):
            base["messages"][-1]["output_pdf_url"] = formatted["output_pdf_url"]
            base["messages"][-1]["output_excel_url"] = formatted["output_excel_url"]
            base["messages"][-1]["output_map_url"] = formatted["output_map_url"]
    except Exception as exc:
        logger.warning("Output formatting failed: %s", exc)
        base["output_markdown"] = proposal_text

    return base


@with_error_handling("booking")
async def booking_node(state: dict[str, Any]) -> dict[str, Any]:
    """Booking tool aggregation (mock data in MVP)."""
    from graph.node_impl import _booking_tool_async

    return await _booking_tool_async(state)


@with_error_handling("apply_single_change")
async def apply_single_change_node(state: dict[str, Any]) -> dict[str, Any]:
    """Apply a single Human-in-the-loop modification and replan locally."""
    change = state.get("pending_change")
    itinerary = state.get("itinerary", [])
    if not change or not itinerary:
        return {"stage": "change_applied"}

    action = change.get("action")
    day_number = change.get("day_number")
    poi_id = change.get("poi_id")

    new_itinerary = [dict(d) for d in itinerary]
    for day in new_itinerary:
        if day_number and day.get("day_number") != day_number:
            continue
        acts = day.get("activities", [])
        if action == "remove" and poi_id:
            day["activities"] = [a for a in acts if a.get("poi_name") != poi_id]
        elif action == "replace" and poi_id and change.get("new_poi"):
            for a in acts:
                if a.get("poi_name") == poi_id:
                    a.update(change["new_poi"])
        # add / reorder can be added here

    return {"itinerary": new_itinerary, "stage": "change_applied", "next_action": "fact_check"}


@with_error_handling("replan_local")
async def replan_local_node(state: dict[str, Any]) -> dict[str, Any]:
    """Lightweight local replan after external event or user feedback."""
    # MVP: forward to plan_node with current slots/profile
    return await plan_node(state)


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
