"""Tests for the in-memory circuit breaker."""

from core.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker("test")
        assert cb.is_open() is False
        assert cb.state() == "CLOSED"

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(
            "test",
            failure_threshold=0.5,
            min_failures=2,
            window_seconds=10,
            recovery_seconds=1,
        )
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        # 2 failures / 3 total = 66% >= 50%, failures >= 2
        assert cb.is_open() is True
        assert cb.state() == "OPEN"

    def test_records_success_while_open(self):
        cb = CircuitBreaker(
            "test",
            failure_threshold=0.5,
            min_failures=1,
            window_seconds=10,
            recovery_seconds=10,
        )
        cb.record_failure()
        assert cb.is_open() is True
        cb.record_success()
        assert cb.is_open() is True  # still within recovery window

    def test_transitions_to_half_open_after_recovery(self):
        import time

        cb = CircuitBreaker(
            "test",
            failure_threshold=0.5,
            min_failures=1,
            window_seconds=10,
            recovery_seconds=0.001,
        )
        cb.record_failure()
        assert cb.is_open() is True
        time.sleep(0.002)
        assert cb.is_open() is False  # recovery window expired -> half-open
        assert cb.state() == "CLOSED"

    def test_does_not_open_below_min_failures(self):
        cb = CircuitBreaker(
            "test",
            failure_threshold=0.5,
            min_failures=5,
            window_seconds=10,
        )
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open() is False
