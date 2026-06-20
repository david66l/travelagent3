"""Tests for Step 8 Prometheus metric helpers."""

from prometheus_client import CollectorRegistry

from core.metrics import (
    ACTIVE_SESSIONS,
    FALLBACK_TOTAL,
    REQUEST_TOTAL,
    RETRIEVAL_LATENCY,
    SOLVE_LATENCY,
    record_fallback,
    record_retrieval_latency,
    record_solve_latency,
    set_active_sessions,
)


def test_request_total_alias_exists():
    assert REQUEST_TOTAL is not None


def test_record_solve_latency():
    before = float(SOLVE_LATENCY.labels(strategy="test")._sum.get())
    record_solve_latency(1.23, strategy="test")
    after = float(SOLVE_LATENCY.labels(strategy="test")._sum.get())
    assert after >= before + 1.23


def test_record_retrieval_latency():
    before = float(RETRIEVAL_LATENCY.labels(source="test")._sum.get())
    record_retrieval_latency(0.45, source="test")
    after = float(RETRIEVAL_LATENCY.labels(source="test")._sum.get())
    assert after >= before + 0.45


def test_record_fallback():
    before = FALLBACK_TOTAL.labels(source="weather", reason="timeout")._value.get()
    record_fallback("weather", "timeout")
    after = FALLBACK_TOTAL.labels(source="weather", reason="timeout")._value.get()
    assert after == before + 1


def test_set_active_sessions():
    set_active_sessions(5)
    assert ACTIVE_SESSIONS._value.get() == 5
    set_active_sessions(0)
    assert ACTIVE_SESSIONS._value.get() == 0
