"""Tests for Step 8 Prometheus metric helpers."""

from core.metrics import (
    ACTIVE_SESSIONS,
    AGENT_TERMINAL_OUTCOMES,
    CHAT_IDEMPOTENCY,
    FALLBACK_TOTAL,
    REQUEST_TOTAL,
    RETRIEVAL_LATENCY,
    SESSION_RUN_LOCK_WAIT,
    SOLVE_LATENCY,
    record_agent_terminal_outcome,
    record_chat_idempotency,
    record_fallback,
    record_retrieval_latency,
    record_session_lock_wait,
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


def test_industrial_reliability_metrics_are_exported():
    before = CHAT_IDEMPOTENCY.labels(outcome="replay")._value.get()
    record_chat_idempotency("replay")
    assert CHAT_IDEMPOTENCY.labels(outcome="replay")._value.get() == before + 1

    before_agent = AGENT_TERMINAL_OUTCOMES.labels(outcome="clarify")._value.get()
    record_agent_terminal_outcome("clarify")
    assert AGENT_TERMINAL_OUTCOMES.labels(outcome="clarify")._value.get() == before_agent + 1

    before_wait = SESSION_RUN_LOCK_WAIT.labels(outcome="acquired")._sum.get()
    record_session_lock_wait(0.02, "acquired")
    assert SESSION_RUN_LOCK_WAIT.labels(outcome="acquired")._sum.get() >= before_wait + 0.02
