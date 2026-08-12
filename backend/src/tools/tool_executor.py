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
import time
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from planner.transport_router import HaversineFallback, MapServiceRouter
from agentic.observations import ObservationEnvelope
from agentic.guard import GuardContext, GuardDecision, ToolGuard
from core.settings import settings
from evaluation.validator import ItineraryValidator
from schemas import ToolResult
from tools.tool_definitions import TOOL_NAME_TO_SCHEMA

logger = logging.getLogger(__name__)

Handler = Callable[[dict[str, Any]], Awaitable[ToolResult]]


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
            "get_local_events": self._handle_get_local_events,
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
                    result = ToolResult(
                        data=None,
                        data_source="unavailable",
                        is_fallback=True,
                        fallback_reason=f"{type(exc).__name__}: {exc}",
                    )
                    observation = ObservationEnvelope.failure(
                        tool=name,
                        code="TOOL_EXECUTION_ERROR",
                        message=str(exc),
                        retryable=True,
                        tool_call_id=tool_call_id,
                        latency_ms=int((time.monotonic() - started) * 1000),
                        details={"error_type": type(exc).__name__},
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

    async def _handle_get_local_events(self, args: dict[str, Any]) -> ToolResult:
        city = args.get("city", "")
        return ToolResult(
            data=[
                {
                    "name": f"{city}周末市集",
                    "date": "本周末",
                    "location": "市中心广场",
                    "category": "市集",
                },
                {
                    "name": f"{city}艺术展",
                    "date": "近期",
                    "location": "美术馆",
                    "category": "展览",
                },
            ],
            data_source="fallback",
            confidence=0.5,
            is_fallback=True,
            fallback_reason="local events API not configured",
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
        candidates = await self._poi.search_pois(
            city=city,
            keywords=[poi],
            category="attraction",
        )
        if candidates:
            best = candidates[0]
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
        return await self._poi.run(
            {
                "city": args["city"],
                "keywords": args.get("keywords") or [],
                "category": args.get("category"),
            }
        )

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
