"""Paired full-chain ablation: flat ReAct versus verified itinerary planning.

The two arms share the same frozen cases, model and built-in POI snapshot.

* ``pure_agent`` lets the model freely call read-only travel tools and requires
  the model itself to submit the final itinerary.  It never exposes CP-SAT.
* ``verified_planner`` gives the model one bounded search decision, then a
  deterministic controller gathers facts, calls CP-SAT and verifies the result.

Provider-reported prompt/completion tokens are accumulated across every model
request.  The benchmark is sequential and resumable so an interrupted cloud
run does not lose completed pairs.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

from openai import APIError, APITimeoutError, RateLimitError


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from core.llm_client import LLMClient  # noqa: E402
from core.json_extract import extract_json_text  # noqa: E402
from core.settings import settings  # noqa: E402
from evaluation.validator import ItineraryValidator  # noqa: E402
from skills.city_data import CITY_DEFAULTS  # noqa: E402
from vrp_solver_service.models import (  # noqa: E402
    ConstraintsInput,
    POIInput,
    SolverRequest,
)
from vrp_solver_service.solver import TravelVRPSolver  # noqa: E402


SCHEMA_VERSION = "pure-agent-vs-verified.v1"
DEFAULT_CASES = ROOT / "ml" / "agentic" / "datasets" / "stage31-authorized-replay-v1" / "cases.jsonl"
DEFAULT_OUTPUT = ROOT / "ml" / "agentic" / "reports" / "stage38-pure-agent-vs-verified-v1"
NON_POI_CATEGORIES = {"meal", "restaurant", "hotel", "transport", "rest"}
INTEREST_ALIASES = {
    "history": "历史",
    "historical": "历史",
    "culture": "文化",
    "cultural": "文化",
    "food": "美食",
    "cuisine": "美食",
    "nature": "自然",
    "photography": "摄影",
    "photo": "摄影",
    "architecture": "建筑",
    "museum": "博物馆",
    "shopping": "购物",
}


@dataclass
class TokenCounter:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0
    model_calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add_response(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        self.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "prompt_tokens_details", None)
        self.cached_prompt_tokens += int(getattr(details, "cached_tokens", 0) or 0)
        self.model_calls += 1


@dataclass
class ArmResult:
    schema_version: str
    case_id: str
    destination: str
    input_hash: str
    mode: str
    model: str
    status: str
    hard_pass: bool
    validator_hard_pass: bool
    grounding_pass: bool
    route_pass: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int
    model_calls: int
    tool_calls: int
    latency_ms: float
    itinerary_days: int
    activity_count: int
    solver_calls: int = 0
    solver_status: str | None = None
    violation_codes: list[str] = field(default_factory=list)
    error: str | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)
    itinerary: list[dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--model", default=settings.llm_model)
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--total-token-budget", type=int, default=32000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [
        case
        for case in cases
        if case.get("family") == "solvable_plan"
        and str(case.get("destination") or "") in CITY_DEFAULTS
    ]


def stratified_sample(cases: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    if size <= 0 or size > len(cases):
        raise ValueError(f"sample-size must be in [1, {len(cases)}]")
    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        groups[str(case["destination"])].append(case)
    for rows in groups.values():
        rng.shuffle(rows)
    selected: list[dict[str, Any]] = []
    cities = sorted(groups)
    while len(selected) < size:
        progressed = False
        for city in cities:
            if groups[city] and len(selected) < size:
                selected.append(groups[city].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _duration_minutes(raw: Any) -> int:
    values = [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(raw or ""))]
    if not values:
        return 90
    hours = sum(values[:2]) / min(len(values), 2)
    return max(45, min(240, int(hours * 60)))


def build_catalog(city: str) -> list[dict[str, Any]]:
    rows = CITY_DEFAULTS.get(city) or []
    catalog: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        item = raw.model_dump(mode="json")
        category = str(item.get("category") or "attraction").lower()
        is_restaurant = category in {"restaurant", "food"}
        location = item.get("location") or {}
        duration = 60 if is_restaurant else _duration_minutes(item.get("recommended_hours"))
        default_close = "21:00" if is_restaurant or duration >= 360 else "18:00"
        catalog.append(
            {
                "id": f"{city}-{index + 1:03d}",
                "name": item["name"],
                "category": "restaurant" if is_restaurant else "attraction",
                "tags": list(item.get("tags") or []),
                "score": float(item.get("score") or 0.5),
                "lat": float(location.get("lat") or 0),
                "lng": float(location.get("lng") or 0),
                "ticket_price": float(item.get("ticket_price") or 0),
                "duration_minutes": duration,
                "open_time": item.get("open_time") or ("10:30" if is_restaurant else "09:00"),
                "close_time": item.get("close_time") or default_close,
                "description": item.get("description") or "",
            }
        )
    if not catalog:
        raise ValueError(f"no frozen POI catalog for city {city}")
    return catalog


def _keywords(values: Iterable[Any]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        token = str(value).strip().lower()
        if not token:
            continue
        normalized.append(INTEREST_ALIASES.get(token, token))
    return normalized


def search_catalog(catalog: list[dict[str, Any]], keywords: Iterable[Any], limit: int = 12) -> list[dict[str, Any]]:
    terms = _keywords(keywords)

    def relevance(item: dict[str, Any]) -> tuple[int, float, str]:
        haystack = " ".join([item["name"], item.get("description", ""), *item.get("tags", [])]).lower()
        matches = sum(1 for term in terms if term.lower() in haystack)
        return matches, float(item["score"]), item["id"]

    ranked = sorted(catalog, key=relevance, reverse=True)
    return ranked[: max(4, min(int(limit or 12), 15))]


def _haversine_km(left: dict[str, Any], right: dict[str, Any]) -> float:
    lat1, lng1 = math.radians(left["lat"]), math.radians(left["lng"])
    lat2, lng2 = math.radians(right["lat"]), math.radians(right["lng"])
    dlat, dlng = lat2 - lat1, lng2 - lng1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(value))


def route_minutes(left: dict[str, Any], right: dict[str, Any]) -> int:
    return max(8, int(round(_haversine_km(left, right) / 22 * 60 + 8)))


def route_matrix(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "poi_ids": [item["id"] for item in items],
        "minutes": [
            [0 if left["id"] == right["id"] else route_minutes(left, right) for right in items]
            for left in items
        ],
    }


def _compact_summary(item: dict[str, Any]) -> dict[str, Any]:
    return {key: item[key] for key in ("id", "name", "category", "tags", "score")}


def _compact_detail(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item[key]
        for key in (
            "id",
            "name",
            "category",
            "tags",
            "lat",
            "lng",
            "ticket_price",
            "duration_minutes",
            "open_time",
            "close_time",
        )
    }


def pure_tools() -> list[dict[str, Any]]:
    activity = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "poi_id": {"type": "string"},
            "poi_name": {"type": "string"},
            "category": {"type": "string"},
            "start_time": {"type": "string", "description": "HH:MM"},
            "end_time": {"type": "string", "description": "HH:MM"},
            "duration_min": {"type": "integer"},
            "ticket_price": {"type": "number"},
            "transport_cost": {"type": "number"},
            "transit_from_prev": {
                "type": "object",
                "properties": {"duration_min": {"type": "integer"}},
                "required": ["duration_min"],
            },
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "poi_id",
            "poi_name",
            "category",
            "start_time",
            "end_time",
            "duration_min",
            "ticket_price",
        ],
    }
    day_plan = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "day_number": {"type": "integer"},
            "date": {"type": "string"},
            "activities": {"type": "array", "items": activity},
            "total_cost": {"type": "number"},
            "total_transit_time_min": {"type": "integer"},
        },
        "required": ["day_number", "date", "activities", "total_cost", "total_transit_time_min"],
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "search_pois",
                "description": "Search the frozen city POI snapshot by interests.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {"type": "array", "items": {"type": "string"}},
                        "limit": {"type": "integer", "minimum": 4, "maximum": 15},
                    },
                    "required": ["keywords"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_poi_details",
                "description": "Get trusted hours, prices, duration and coordinates for POI IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {"poi_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["poi_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_route_matrix",
                "description": "Get pairwise travel minutes for POI IDs.",
                "parameters": {
                    "type": "object",
                    "properties": {"poi_ids": {"type": "array", "items": {"type": "string"}}},
                    "required": ["poi_ids"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the frozen weather snapshot for the trip.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_itinerary",
                "description": "Submit the final itinerary. The model must schedule it directly; no solver is available.",
                "parameters": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"itinerary": {"type": "array", "items": day_plan}},
                    "required": ["itinerary"],
                },
            },
        },
    ]


def search_only_tool() -> list[dict[str, Any]]:
    return [pure_tools()[0]]


def _case_prompt(case: dict[str, Any]) -> str:
    interests = "、".join(case.get("interests") or [])
    return (
        f"为用户规划{case['destination']}{case['travel_days']}日游，日期从{case['start_date']}到"
        f"{case['end_date']}，总预算{case['budget']:.0f}元，兴趣为{interests}。"
        "每日活动应在09:00-21:00之间，遵守景点营业时间，活动之间预留真实通勤时间，"
        "不得重复或虚构景点。"
    )


async def _completion(
    client: LLMClient,
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    max_tokens: int,
    tool_choice: str = "required",
) -> Any:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if model.startswith("deepseek-v4"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    for attempt in range(3):
        try:
            return await client._create_completion(**kwargs)
        except (RateLimitError, APITimeoutError, APIError):
            if attempt == 2:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable completion retry state")


def _assistant_message(message: Any) -> dict[str, Any]:
    calls = []
    for call in list(getattr(message, "tool_calls", None) or []):
        calls.append(
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
            }
        )
    return {"role": "assistant", "content": message.content or "", "tool_calls": calls}


def _parse_arguments(call: Any) -> dict[str, Any]:
    raw = call.function.arguments or "{}"
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError("tool arguments must be an object")
    return value


def parse_final_itinerary(content: str) -> list[dict[str, Any]]:
    """Extract the first complete itinerary payload from a prose/JSON response."""
    decoder = json.JSONDecoder()
    candidates = [extract_json_text(content)]
    candidates.extend(content[index:] for index, char in enumerate(content) if char in "{[")
    for candidate in candidates:
        try:
            payload, _ = decoder.raw_decode(candidate.lstrip())
        except json.JSONDecodeError:
            continue
        itinerary = payload.get("itinerary") if isinstance(payload, dict) else payload
        if isinstance(itinerary, list) and all(isinstance(day, dict) for day in itinerary):
            return itinerary
    raise ValueError("final answer did not contain one complete itinerary JSON payload")


def execute_read_tool(name: str, arguments: dict[str, Any], catalog: list[dict[str, Any]], case: dict[str, Any]) -> dict[str, Any]:
    by_id = {item["id"]: item for item in catalog}
    if name == "search_pois":
        rows = search_catalog(catalog, arguments.get("keywords") or case.get("interests") or [], arguments.get("limit") or 12)
        return {"pois": [_compact_summary(item) for item in rows]}
    if name == "get_poi_details":
        rows = [by_id[item] for item in arguments.get("poi_ids") or [] if item in by_id]
        return {"pois": [_compact_detail(item) for item in rows]}
    if name == "get_route_matrix":
        rows = [by_id[item] for item in arguments.get("poi_ids") or [] if item in by_id]
        return route_matrix(rows)
    if name == "get_weather":
        return {
            "days": [
                {"date": (date.fromisoformat(case["start_date"]) + timedelta(days=offset)).isoformat(), "condition": "晴", "temperature": "18-26℃"}
                for offset in range(int(case["travel_days"]))
            ]
        }
    raise ValueError(f"unknown read tool {name}")


def _minutes(value: Any) -> int | None:
    try:
        hours, minutes = str(value).split(":", 1)
        return int(hours) * 60 + int(minutes)
    except (TypeError, ValueError):
        return None


def verify_grounding_and_routes(itinerary: list[dict[str, Any]], catalog: list[dict[str, Any]], max_daily_transit: int = 120) -> tuple[bool, bool, list[str]]:
    by_id = {item["id"]: item for item in catalog}
    grounding_ok = True
    route_ok = True
    violations: list[str] = []
    for day in itinerary:
        activities = [item for item in day.get("activities") or [] if isinstance(item, dict)]
        actual_transit = 0
        previous: tuple[dict[str, Any], dict[str, Any]] | None = None
        for activity in activities:
            poi_id = str(activity.get("poi_id") or "")
            fact = by_id.get(poi_id)
            if fact is None:
                grounding_ok = False
                violations.append("UNGROUNDED_POI")
                previous = None
                continue
            if str(activity.get("poi_name") or "") != fact["name"]:
                grounding_ok = False
                violations.append("POI_NAME_MISMATCH")
            if previous is not None:
                previous_activity, previous_fact = previous
                commute = route_minutes(previous_fact, fact)
                actual_transit += commute
                previous_end = _minutes(previous_activity.get("end_time"))
                current_start = _minutes(activity.get("start_time"))
                if previous_end is None or current_start is None or current_start - previous_end < commute:
                    route_ok = False
                    violations.append("INSUFFICIENT_TRANSIT_GAP")
            previous = (activity, fact)
        if actual_transit > max_daily_transit:
            route_ok = False
            violations.append("MAX_TRANSIT_EXCEEDED_RECOMPUTED")
    return grounding_ok, route_ok, sorted(set(violations))


def validate_itinerary(case: dict[str, Any], itinerary: list[dict[str, Any]], catalog: list[dict[str, Any]]) -> tuple[bool, bool, bool, list[str]]:
    constraints = {
        "travel_days": int(case["travel_days"]),
        "day_start_min": 9 * 60,
        "day_end_min": 21 * 60,
        "max_transit_minutes": 120,
        "total_budget": float(case["budget"]),
        "interests": list(case.get("interests") or []),
    }
    report = ItineraryValidator().validate(itinerary, constraints=constraints, facts=catalog)
    grounding_ok, route_ok, extra = verify_grounding_and_routes(itinerary, catalog)
    codes = [item.code for item in report.hard_violations] + extra
    return report.hard_pass, grounding_ok, route_ok, sorted(set(codes))


async def run_pure_agent(
    case: dict[str, Any],
    *,
    model: str,
    max_turns: int,
    max_tokens: int,
    total_token_budget: int = 32000,
) -> ArmResult:
    started = time.perf_counter()
    counter = TokenCounter()
    tool_count = 0
    trace: list[dict[str, Any]] = []
    itinerary: list[dict[str, Any]] = []
    error: str | None = None
    catalog = build_catalog(str(case["destination"]))
    client = LLMClient()
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你是扁平 ReAct 旅行规划 Agent。你必须自己决定调用哪些工具、调用顺序以及最终日程。"
                "先查询事实再提交，submit_itinerary 是唯一终止动作。系统没有 CP-SAT 或其他排程器。"
                "严格使用工具返回的 POI ID、名称、营业时间、价格和路线时间。每种查询工具最多调用一次，"
                "信息够用后立即调用 submit_itinerary，或直接返回同参数结构的 JSON；不要重复查询。"
            ),
        },
        {"role": "user", "content": _case_prompt(case)},
    ]
    try:
        for turn in range(max_turns):
            if counter.total_tokens >= total_token_budget:
                raise RuntimeError(f"total token budget exceeded ({total_token_budget})")
            response = await _completion(
                client,
                model=model,
                messages=messages,
                tools=pure_tools(),
                max_tokens=max_tokens,
                tool_choice="auto",
            )
            counter.add_response(response)
            message = response.choices[0].message
            calls = list(message.tool_calls or [])
            if not calls:
                content = message.content or ""
                trace.append(
                    {
                        "turn": turn + 1,
                        "action": "final_json",
                        "finish_reason": str(response.choices[0].finish_reason or ""),
                        "content_chars": len(content),
                    }
                )
                try:
                    itinerary = parse_final_itinerary(content)
                except (ValueError, json.JSONDecodeError) as exc:
                    trace[-1]["parse_error"] = type(exc).__name__
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "最终行程无法解析。请保留既有事实，不要重新查询；"
                                "立即调用 submit_itinerary，并输出完整合法参数。"
                            ),
                        }
                    )
                    continue
                break
            messages.append(_assistant_message(message))
            for call in calls:
                tool_count += 1
                name = str(call.function.name)
                trace.append(
                    {
                        "turn": turn + 1,
                        "action": name,
                        "finish_reason": str(response.choices[0].finish_reason or ""),
                        "argument_chars": len(str(call.function.arguments or "")),
                    }
                )
                try:
                    arguments = _parse_arguments(call)
                except (ValueError, json.JSONDecodeError) as exc:
                    trace[-1]["parse_error"] = type(exc).__name__
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": _canonical(
                                {
                                    "error": "INVALID_TOOL_ARGUMENTS",
                                    "instruction": "Do not search again; retry submit_itinerary with complete valid JSON.",
                                }
                            ),
                        }
                    )
                    continue
                if name == "submit_itinerary":
                    candidate = arguments.get("itinerary")
                    if not isinstance(candidate, list):
                        raise ValueError("submit_itinerary omitted itinerary array")
                    itinerary = candidate
                    break
                result = execute_read_tool(name, arguments, catalog, case)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": _canonical(result),
                    }
                )
            if itinerary:
                break
        if not itinerary:
            raise RuntimeError("max turns reached without submit_itinerary")
    except Exception as exc:  # keep the paired benchmark resumable
        error = f"{type(exc).__name__}: {exc}"[:500]

    validator_pass, grounding_pass, route_pass, codes = validate_itinerary(case, itinerary, catalog)
    hard_pass = bool(itinerary) and validator_pass and grounding_pass and route_pass and error is None
    return ArmResult(
        schema_version=SCHEMA_VERSION,
        case_id=case["case_id"],
        destination=str(case["destination"]),
        input_hash=_hash(case),
        mode="pure_agent",
        model=model,
        status="completed" if error is None else "failed",
        hard_pass=hard_pass,
        validator_hard_pass=validator_pass,
        grounding_pass=grounding_pass,
        route_pass=route_pass,
        prompt_tokens=counter.prompt_tokens,
        completion_tokens=counter.completion_tokens,
        total_tokens=counter.total_tokens,
        cached_prompt_tokens=counter.cached_prompt_tokens,
        model_calls=counter.model_calls,
        tool_calls=tool_count,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        itinerary_days=len(itinerary),
        activity_count=sum(len(day.get("activities") or []) for day in itinerary),
        violation_codes=codes,
        error=error,
        trace=trace,
        itinerary=itinerary,
    )


def _solver_itinerary(response: Any, case: dict[str, Any]) -> list[dict[str, Any]]:
    start = date.fromisoformat(case["start_date"])
    itinerary: list[dict[str, Any]] = []
    for offset, raw_day in enumerate(response.days):
        day = raw_day.model_dump(mode="json")
        day["date"] = (start + timedelta(days=offset)).isoformat()
        itinerary.append(day)
    return itinerary


async def run_verified_planner(case: dict[str, Any], *, model: str, max_tokens: int) -> ArmResult:
    started = time.perf_counter()
    counter = TokenCounter()
    trace: list[dict[str, Any]] = []
    itinerary: list[dict[str, Any]] = []
    error: str | None = None
    solver_status: str | None = None
    catalog = build_catalog(str(case["destination"]))
    client = LLMClient()
    tool_count = 0
    try:
        response = await _completion(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是可验证旅行规划器中的有界策略模型。Controller 当前只允许 search_pois。"
                        "根据用户兴趣生成简短检索词，只调用一次该工具；排程和校验由确定性组件负责。"
                    ),
                },
                {"role": "user", "content": _case_prompt(case)},
            ],
            tools=search_only_tool(),
            max_tokens=min(max_tokens, 128),
        )
        counter.add_response(response)
        calls = list(response.choices[0].message.tool_calls or [])
        if not calls:
            raise ValueError("bounded policy returned no search action")
        call = calls[0]
        arguments = _parse_arguments(call)
        tool_count = 1
        trace.append({"turn": 1, "action": "search_pois"})
        selected = search_catalog(
            catalog,
            arguments.get("keywords") or case.get("interests") or [],
            12,
        )
        attraction_candidates = [item for item in selected if item["category"] == "attraction"]
        if attraction_candidates:
            selected = attraction_candidates
        # Match the production detail-hydration budget: keep a compact, coherent
        # candidate cluster instead of feeding every remote landmark to CP-SAT.
        anchor = selected[0]
        detail_limit = min(8, max(6, int(case["travel_days"]) * 2))
        selected = sorted(selected, key=lambda item: route_minutes(anchor, item))[:detail_limit]
        tool_count += 2  # deterministic detail hydration + route matrix
        trace.extend(
            [
                {"controller": True, "action": "get_poi_details"},
                {"controller": True, "action": "get_route_matrix"},
            ]
        )
        start = date.fromisoformat(case["start_date"])
        weekdays = [(start + timedelta(days=offset)).weekday() for offset in range(int(case["travel_days"]))]
        frozen_routes = route_matrix(selected)["minutes"]
        matrix_size = len(selected) + 1
        dist_matrix = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
        tc_matrix = [[0.0 for _ in range(matrix_size)] for _ in range(matrix_size)]
        for left in range(len(selected)):
            for right in range(len(selected)):
                dist_matrix[left + 1][right + 1] = int(frozen_routes[left][right])
                tc_matrix[left + 1][right + 1] = round(float(frozen_routes[left][right]) * 0.2, 2)
        request = SolverRequest(
            pois=[
                POIInput(
                    id=item["id"],
                    name=item["name"],
                    category=item["category"],
                    tags=item["tags"],
                    lat=item["lat"],
                    lng=item["lng"],
                    score=item["score"],
                    ticket_price=item["ticket_price"],
                    duration_minutes=item["duration_minutes"],
                    open_time=item["open_time"],
                    close_time=item["close_time"],
                )
                for item in selected
            ],
            constraints=ConstraintsInput(
                travel_days=int(case["travel_days"]),
                day_weekdays=weekdays,
                day_start_min=9 * 60,
                day_end_min=21 * 60,
                max_transit_minutes=120,
                total_budget=float(case["budget"]),
                interests=list(case.get("interests") or []),
                include_restaurant=False,
                transition_buffer_min=15,
            ),
            strategy="cpsat",
            dist_matrix=dist_matrix,
            tc_matrix=tc_matrix,
        )
        solver_response = TravelVRPSolver().solve(request)
        solver_status = solver_response.status
        trace.append({"controller": True, "action": "solve_itinerary", "status": solver_status})
        itinerary = _solver_itinerary(solver_response, case)
        tool_count += 2  # solver + independent validator
        trace.append({"controller": True, "action": "validate_itinerary"})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]

    validator_pass, grounding_pass, route_pass, codes = validate_itinerary(case, itinerary, catalog)
    hard_pass = bool(itinerary) and validator_pass and grounding_pass and route_pass and error is None
    return ArmResult(
        schema_version=SCHEMA_VERSION,
        case_id=case["case_id"],
        destination=str(case["destination"]),
        input_hash=_hash(case),
        mode="verified_planner",
        model=model,
        status="completed" if error is None else "failed",
        hard_pass=hard_pass,
        validator_hard_pass=validator_pass,
        grounding_pass=grounding_pass,
        route_pass=route_pass,
        prompt_tokens=counter.prompt_tokens,
        completion_tokens=counter.completion_tokens,
        total_tokens=counter.total_tokens,
        cached_prompt_tokens=counter.cached_prompt_tokens,
        model_calls=counter.model_calls,
        tool_calls=tool_count,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        itinerary_days=len(itinerary),
        activity_count=sum(len(day.get("activities") or []) for day in itinerary),
        solver_calls=1,
        solver_status=solver_status,
        violation_codes=codes,
        error=error,
        trace=trace,
        itinerary=itinerary,
    )


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower))


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [float(row["total_tokens"]) for row in rows]
    latencies = [float(row["latency_ms"]) for row in rows]
    passes = sum(bool(row["hard_pass"]) for row in rows)
    return {
        "tasks": len(rows),
        "hard_pass_count": passes,
        "hard_pass_rate": passes / len(rows) if rows else 0.0,
        "validator_hard_pass_rate": (
            sum(bool(row["validator_hard_pass"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "grounding_pass_rate": (
            sum(bool(row["grounding_pass"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "route_pass_rate": (
            sum(bool(row["route_pass"]) for row in rows) / len(rows) if rows else 0.0
        ),
        "mean_prompt_tokens": statistics.fmean(float(row["prompt_tokens"]) for row in rows) if rows else 0.0,
        "mean_completion_tokens": statistics.fmean(float(row["completion_tokens"]) for row in rows) if rows else 0.0,
        "mean_total_tokens": statistics.fmean(tokens) if rows else 0.0,
        "median_total_tokens": statistics.median(tokens) if rows else 0.0,
        "p95_total_tokens": _percentile(tokens, 0.95),
        "tokens_per_hard_pass": sum(tokens) / passes if passes else None,
        "mean_model_calls": statistics.fmean(float(row["model_calls"]) for row in rows) if rows else 0.0,
        "mean_tool_calls": statistics.fmean(float(row["tool_calls"]) for row in rows) if rows else 0.0,
        "mean_latency_ms": statistics.fmean(latencies) if rows else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "failures": sum(row["status"] != "completed" for row in rows),
        "violation_counts": dict(Counter(code for row in rows for code in row.get("violation_codes") or [])),
    }


def _bootstrap_delta(pairs: list[tuple[float, float]], seed: int, rounds: int = 5000) -> list[float]:
    if not pairs:
        return [0.0, 0.0]
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(rounds):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        deltas.append(statistics.fmean(left - right for left, right in sample))
    return [round(_percentile(deltas, 0.025), 3), round(_percentile(deltas, 0.975), 3)]


def _paired_delta(
    paired: list[dict[str, dict[str, Any]]],
    pure_summary: dict[str, Any],
    verified_summary: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    pure_mean = float(pure_summary["mean_total_tokens"])
    verified_mean = float(verified_summary["mean_total_tokens"])
    token_pairs = [
        (float(pair["pure_agent"]["total_tokens"]), float(pair["verified_planner"]["total_tokens"]))
        for pair in paired
    ]
    pure_latency = float(pure_summary["mean_latency_ms"])
    verified_latency = float(verified_summary["mean_latency_ms"])
    return {
        "mean_token_delta_pure_minus_verified": pure_mean - verified_mean,
        "mean_token_delta_95pct_bootstrap_ci": _bootstrap_delta(token_pairs, seed),
        "pure_to_verified_token_ratio": pure_mean / verified_mean if verified_mean else None,
        "pure_token_excess_vs_verified_percent": (
            (pure_mean - verified_mean) / verified_mean * 100 if verified_mean else None
        ),
        "verified_token_reduction_vs_pure_percent": (
            (pure_mean - verified_mean) / pure_mean * 100 if pure_mean else None
        ),
        "hard_pass_rate_delta_pure_minus_verified": (
            pure_summary["hard_pass_rate"] - verified_summary["hard_pass_rate"]
        ),
        "mean_model_call_delta": pure_summary["mean_model_calls"] - verified_summary["mean_model_calls"],
        "mean_latency_delta_ms": pure_latency - verified_latency,
        "pure_to_verified_latency_ratio": pure_latency / verified_latency if verified_latency else None,
        "verified_latency_reduction_vs_pure_percent": (
            (pure_latency - verified_latency) / pure_latency * 100 if pure_latency else None
        ),
    }


def _has_preprocessing_duration_window_conflict(row: dict[str, Any]) -> bool:
    """Detect a solver activity expanded beyond the frozen opening window."""
    destination = str(row.get("destination") or "")
    if destination not in CITY_DEFAULTS:
        return False
    facts = {item["id"]: item for item in build_catalog(destination)}
    for day in row.get("itinerary") or []:
        for activity in day.get("activities") or []:
            fact = facts.get(str(activity.get("poi_id") or ""))
            if not fact:
                continue
            opening = _minutes(fact.get("open_time"))
            closing = _minutes(fact.get("close_time"))
            duration = int(activity.get("duration_min") or 0)
            if opening is not None and closing is not None and duration > closing - opening:
                return True
    return False


def build_report(
    rows: list[dict[str, Any]],
    *,
    model: str,
    seed: int,
    requested_size: int,
    total_token_budget: int = 32000,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[row["case_id"]][row["mode"]] = row
    paired = [value for value in grouped.values() if {"pure_agent", "verified_planner"} <= set(value)]
    pure = [pair["pure_agent"] for pair in paired]
    verified = [pair["verified_planner"] for pair in paired]
    pure_summary = _arm_summary(pure)
    verified_summary = _arm_summary(verified)
    for summary, arm_rows in ((pure_summary, pure), (verified_summary, verified)):
        overruns = [int(row["total_tokens"]) for row in arm_rows if int(row["total_tokens"]) > total_token_budget]
        summary["soft_token_budget_overrun_count"] = len(overruns)
        summary["maximum_token_budget_overrun"] = (
            max(value - total_token_budget for value in overruns) if overruns else 0
        )
    excluded_case_ids = sorted(
        pair["verified_planner"]["case_id"]
        for pair in paired
        if _has_preprocessing_duration_window_conflict(pair["verified_planner"])
    )
    sensitivity_pairs = [
        pair for pair in paired if pair["verified_planner"]["case_id"] not in excluded_case_ids
    ]
    sensitivity_pure = _arm_summary([pair["pure_agent"] for pair in sensitivity_pairs])
    sensitivity_verified = _arm_summary([pair["verified_planner"] for pair in sensitivity_pairs])
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "flat ReAct direct itinerary vs bounded policy plus CP-SAT plus verifier",
        "model": model,
        "seed": seed,
        "requested_sample_size": requested_size,
        "total_token_budget_per_task": total_token_budget,
        "paired_tasks": len(paired),
        "covered_cities": sorted({str(row.get("destination")) for row in rows if row.get("destination")}),
        "token_accounting": "provider-reported prompt_tokens + completion_tokens across all model calls",
        "pure_agent": pure_summary,
        "verified_planner": verified_summary,
        "paired_delta": _paired_delta(paired, pure_summary, verified_summary, seed),
        "integration_bug_sensitivity": {
            "excluded_case_ids": excluded_case_ids,
            "exclusion_reason": (
                "solver preprocessing expanded an activity beyond its frozen opening window; "
                "primary result still counts the case as failed"
            ),
            "paired_tasks": len(sensitivity_pairs),
            "pure_agent": sensitivity_pure,
            "verified_planner": sensitivity_verified,
            "paired_delta": _paired_delta(
                sensitivity_pairs, sensitivity_pure, sensitivity_verified, seed
            ),
        },
        "limitations": [
            "使用冻结的合成授权回放任务与内置 POI 快照，不是真实用户流量。",
            "云端 API 顺序回放；延迟不是并发生产服务基准。",
            "不使用 LLM Judge；成功必须同时通过确定性硬约束、事实接地和路线间隔复算。",
            "纯 Agent 组不可访问 CP-SAT，必须由模型直接编排行程。",
            "32k Token 预算在模型请求之间执行；最后一次请求可能越界，报告保留服务端实际计费 Token。",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    pure = report["pure_agent"]
    verified = report["verified_planner"]
    delta = report["paired_delta"]

    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.2f}%"

    return "\n".join(
        [
            "# 纯 Agent Loop vs 可验证行程规划：全链路 A/B",
            "",
            f"- 协议：`{report['schema_version']}`",
            f"- 模型：`{report['model']}`",
            f"- 配对任务：{report['paired_tasks']}",
            f"- Token 口径：{report['token_accounting']}",
            f"- Token 软预算：每任务 {report['total_token_budget_per_task']}，按请求边界执行",
            "",
            "## 结果",
            "",
            "| 指标 | 纯 Agent Loop | 可验证规划 |",
            "|---|---:|---:|",
            f"| 硬约束+事实+路线通过率 | {pct(pure['hard_pass_rate'] * 100)} | {pct(verified['hard_pass_rate'] * 100)} |",
            f"| 平均输入 Token/任务 | {pure['mean_prompt_tokens']:.2f} | {verified['mean_prompt_tokens']:.2f} |",
            f"| 平均输出 Token/任务 | {pure['mean_completion_tokens']:.2f} | {verified['mean_completion_tokens']:.2f} |",
            f"| 平均总 Token/任务 | {pure['mean_total_tokens']:.2f} | {verified['mean_total_tokens']:.2f} |",
            f"| P95 总 Token/任务 | {pure['p95_total_tokens']:.2f} | {verified['p95_total_tokens']:.2f} |",
            f"| 全批 Token 按通过任务摊销 | {pure['tokens_per_hard_pass'] or 0:.2f} | {verified['tokens_per_hard_pass'] or 0:.2f} |",
            f"| 平均模型调用次数 | {pure['mean_model_calls']:.2f} | {verified['mean_model_calls']:.2f} |",
            f"| 平均工具调用次数 | {pure['mean_tool_calls']:.2f} | {verified['mean_tool_calls']:.2f} |",
            f"| 平均端到端延迟 | {pure['mean_latency_ms']:.2f} ms | {verified['mean_latency_ms']:.2f} ms |",
            f"| P95 端到端延迟 | {pure['p95_latency_ms']:.2f} ms | {verified['p95_latency_ms']:.2f} ms |",
            "",
            "## 核心结论",
            "",
            f"- 纯 Agent Loop 相对可验证规划多消耗 Token：**{pct(delta['pure_token_excess_vs_verified_percent'])}**。",
            f"- 可验证规划相对纯 Agent Loop 节省 Token：**{pct(delta['verified_token_reduction_vs_pure_percent'])}**。",
            f"- 纯 Agent / 可验证规划 Token 倍数：**{delta['pure_to_verified_token_ratio']:.2f}x**。",
            f"- 平均每任务绝对减少：**{delta['mean_token_delta_pure_minus_verified']:.2f} Token**；配对 Bootstrap 95% CI 为 {delta['mean_token_delta_95pct_bootstrap_ci']}。",
            f"- 硬通过率差（纯 Agent - 可验证）：**{delta['hard_pass_rate_delta_pure_minus_verified'] * 100:.2f} 个百分点**。",
            f"- 可验证规划相对纯 Agent 平均延迟降低：**{pct(delta['verified_latency_reduction_vs_pure_percent'])}**。",
            "",
            "## 已定位集成缺陷敏感性",
            "",
            f"- 主结果不删样本；仅在敏感性分析中排除“预处理时长超过冻结营业窗”的案例："
            f"{report['integration_bug_sensitivity']['excluded_case_ids']}。",
            f"- 剩余 {report['integration_bug_sensitivity']['paired_tasks']} 对中，纯 Agent 联合通过率 "
            f"{pct(report['integration_bug_sensitivity']['pure_agent']['hard_pass_rate'] * 100)}，"
            f"可验证规划为 {pct(report['integration_bug_sensitivity']['verified_planner']['hard_pass_rate'] * 100)}。",
            "",
            "## 口径边界",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
        ]
    )


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    selected = stratified_sample(load_cases(args.cases), args.sample_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_path = args.output_dir / "runs.jsonl"
    existing = _load_existing(runs_path) if args.resume else []
    if runs_path.exists() and not args.resume:
        raise FileExistsError(f"{runs_path} already exists; choose a new output dir or pass --resume")
    done = {(row["case_id"], row["mode"]) for row in existing}
    rows = list(existing)
    with runs_path.open("a", encoding="utf-8") as handle:
        for index, case in enumerate(selected):
            arms = ["pure_agent", "verified_planner"]
            if index % 2:
                arms.reverse()
            for arm in arms:
                key = (case["case_id"], arm)
                if key in done:
                    continue
                if arm == "pure_agent":
                    result = await run_pure_agent(
                        case,
                        model=args.model,
                        max_turns=args.max_turns,
                        max_tokens=args.max_tokens,
                        total_token_budget=args.total_token_budget,
                    )
                else:
                    result = await run_verified_planner(case, model=args.model, max_tokens=args.max_tokens)
                payload = asdict(result)
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                handle.flush()
                rows.append(payload)
                done.add(key)
                print(
                    json.dumps(
                        {
                            "progress": f"{len(done)}/{len(selected) * 2}",
                            "case_id": result.case_id,
                            "mode": result.mode,
                            "tokens": result.total_tokens,
                            "hard_pass": result.hard_pass,
                            "error": result.error,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    report = build_report(
        rows,
        model=args.model,
        seed=args.seed,
        requested_size=args.sample_size,
        total_token_budget=args.total_token_budget,
    )
    (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "REPORT.md").write_text(render_markdown(report), encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(asyncio.run(run(parse_args())), ensure_ascii=False, indent=2))
