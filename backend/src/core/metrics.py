"""Prometheus metrics (PRD §12.7)."""

from __future__ import annotations

import threading
import time
from typing import Dict

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

_start_time = time.time()

UP = Gauge("travel_agent_up", "Service is running")
UP.set(1)
UPTIME = Gauge("travel_agent_uptime_seconds", "Process uptime seconds")

PLANNING_JOBS_CREATED = Counter(
    "planning_jobs_created_total",
    "Planning jobs created",
)
PLANNING_JOBS_COMPLETED = Counter(
    "planning_jobs_completed_total",
    "Planning jobs completed",
)
PLANNING_JOBS_FAILED = Counter(
    "planning_jobs_failed_total",
    "Planning jobs failed",
)
PLANNING_JOBS_FORCE_CANCELLED = Counter(
    "planning_jobs_force_cancelled_total",
    "Planning jobs force-cancelled",
)
DEAD_LETTER_ENTRIES = Counter(
    "dead_letter_entries_total",
    "Dead-letter queue entries",
)

LLM_TOKEN_USAGE = Counter(
    "llm_token_usage_total",
    "LLM token usage",
    ["model", "task_type", "token_type", "user_tier"],
)

LLM_REQUEST_DURATION = Histogram(
    "llm_request_duration_seconds",
    "LLM request latency",
    ["model", "task_type"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 60.0),
)

EXTERNAL_API_CALLS = Counter(
    "external_api_calls_total",
    "External API calls",
    ["api_name", "status", "data_source"],
)

EXTERNAL_API_COST = Counter(
    "external_api_cost_cny_total",
    "Estimated external API cost in CNY",
    ["api_name"],
)

COST_CIRCUIT_ACTIVE = Gauge(
    "cost_circuit_breaker_active",
    "Global cost circuit breaker is active (1=yes)",
)

DAILY_TOKEN_COST = Gauge(
    "cost_daily_tokens_total",
    "Global daily LLM tokens consumed",
)

DAILY_API_COST_CNY = Gauge(
    "cost_daily_external_api_cny",
    "Global daily external API cost estimate (CNY)",
)

CACHE_HITS = Counter(
    "cache_hits_total",
    "Cache hits",
    ["cache_type"],
)
CACHE_MISSES = Counter(
    "cache_misses_total",
    "Cache misses",
    ["cache_type"],
)

HTTP_REQUESTS = Counter(
    "http_requests_total",
    "HTTP requests",
    ["method", "endpoint", "status_code"],
)
HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Alias / business-facing metric names required by the operations layer.
REQUEST_TOTAL = HTTP_REQUESTS
LLM_LATENCY = LLM_REQUEST_DURATION

SOLVE_LATENCY = Histogram(
    "solve_latency_seconds",
    "Itinerary solver latency",
    ["strategy"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

RETRIEVAL_LATENCY = Histogram(
    "retrieval_latency_seconds",
    "RAG POI retrieval latency",
    ["source"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

ACTIVE_SESSIONS = Gauge(
    "active_sessions",
    "Current number of active SSE sessions",
)

FALLBACK_TOTAL = Counter(
    "fallback_total",
    "Fallback responses from external dependencies",
    ["source", "reason"],
)

PROMPT_INJECTION_BLOCKED = Counter(
    "prompt_injection_blocked_total",
    "Blocked prompt-injection attempts",
)

_COUNTER_MAP: Dict[str, Counter] = {
    "planning_jobs_created_total": PLANNING_JOBS_CREATED,
    "planning_jobs_completed_total": PLANNING_JOBS_COMPLETED,
    "planning_jobs_failed_total": PLANNING_JOBS_FAILED,
    "planning_jobs_force_cancelled_total": PLANNING_JOBS_FORCE_CANCELLED,
    "dead_letter_entries_total": DEAD_LETTER_ENTRIES,
}

_lock = threading.Lock()
_legacy_counters: Dict[str, int] = {}


def incr(name: str, delta: int = 1) -> None:
    """Increment a named counter (backward compatible)."""
    counter = _COUNTER_MAP.get(name)
    if counter is not None:
        counter.inc(delta)
        return
    with _lock:
        _legacy_counters[name] = _legacy_counters.get(name, 0) + delta


def record_llm_tokens(
    model: str,
    task_type: str,
    prompt_tokens: int,
    completion_tokens: int,
    user_tier: str = "unknown",
) -> None:
    if prompt_tokens:
        LLM_TOKEN_USAGE.labels(
            model=model,
            task_type=task_type,
            token_type="prompt",
            user_tier=user_tier,
        ).inc(prompt_tokens)
    if completion_tokens:
        LLM_TOKEN_USAGE.labels(
            model=model,
            task_type=task_type,
            token_type="completion",
            user_tier=user_tier,
        ).inc(completion_tokens)


def record_llm_duration(model: str, task_type: str, duration_s: float) -> None:
    LLM_REQUEST_DURATION.labels(model=model, task_type=task_type).observe(duration_s)


def record_external_api_call(
    api_name: str,
    *,
    status: str = "success",
    data_source: str = "api",
) -> None:
    EXTERNAL_API_CALLS.labels(
        api_name=api_name,
        status=status,
        data_source=data_source,
    ).inc()


def record_external_api_cost_cny(api_name: str, cost_cny: float) -> None:
    if cost_cny > 0:
        EXTERNAL_API_COST.labels(api_name=api_name).inc(cost_cny)


def set_cost_circuit_active(active: bool) -> None:
    COST_CIRCUIT_ACTIVE.set(1 if active else 0)


def set_daily_cost_gauges(daily_tokens: int, daily_api_cost_cny: float) -> None:
    DAILY_TOKEN_COST.set(daily_tokens)
    DAILY_API_COST_CNY.set(daily_api_cost_cny)


def record_cache_hit(cache_type: str) -> None:
    CACHE_HITS.labels(cache_type=cache_type).inc()


def record_cache_miss(cache_type: str) -> None:
    CACHE_MISSES.labels(cache_type=cache_type).inc()


def record_prompt_injection_blocked() -> None:
    PROMPT_INJECTION_BLOCKED.inc()


def record_http_request(method: str, endpoint: str, status_code: int, duration_s: float) -> None:
    HTTP_REQUESTS.labels(
        method=method,
        endpoint=endpoint,
        status_code=str(status_code),
    ).inc()
    HTTP_DURATION.labels(method=method, endpoint=endpoint).observe(duration_s)


def record_solve_latency(duration_s: float, strategy: str = "default") -> None:
    SOLVE_LATENCY.labels(strategy=str(strategy)).observe(duration_s)


def record_retrieval_latency(duration_s: float, source: str = "rag") -> None:
    RETRIEVAL_LATENCY.labels(source=str(source)).observe(duration_s)


def record_fallback(source: str, reason: str = "fallback") -> None:
    FALLBACK_TOTAL.labels(source=str(source), reason=str(reason)).inc()


def set_active_sessions(count: int) -> None:
    ACTIVE_SESSIONS.set(int(count))


def render_prometheus() -> str:
    UPTIME.set(time.time() - _start_time)
    body = generate_latest().decode("utf-8")
    with _lock:
        if not _legacy_counters:
            return body
        extra_lines = []
        for name, value in sorted(_legacy_counters.items()):
            extra_lines.append(f"# TYPE {name} counter")
            extra_lines.append(f"{name} {value}")
        return body + "\n".join(extra_lines) + "\n"


def prometheus_content_type() -> str:
    return CONTENT_TYPE_LATEST
