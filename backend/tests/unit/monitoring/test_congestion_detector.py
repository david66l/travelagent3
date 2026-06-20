"""Tests for CongestionDetector."""

from unittest.mock import AsyncMock

import pytest
from prometheus_client import CollectorRegistry, Counter, Histogram

from monitoring.congestion_detector import CongestionDetector


@pytest.fixture
def detector():
    return CongestionDetector()


@pytest.fixture
def mock_redis_for_monitoring(monkeypatch):
    """Provide a fresh Redis mock with monitoring-specific methods."""
    redis_mock = AsyncMock()
    redis_mock.llen = AsyncMock(return_value=0)
    redis_mock.scan = AsyncMock(return_value=(0, []))
    monkeypatch.setattr("monitoring.congestion_detector.redis_client", redis_mock)
    return redis_mock


class TestQueueLengths:
    async def test_returns_lengths_for_prefixed_keys(self, mock_redis_for_monitoring):
        mock_redis_for_monitoring.llen = AsyncMock(return_value=12)
        detector = CongestionDetector()
        result = await detector.queue_lengths()
        assert result == {
            "default": 12,
            "planning": 12,
            "memory": 12,
            "planning_dead_letter": 12,
        }

    async def test_falls_back_to_bare_queue_name(self, mock_redis_for_monitoring):
        async def _llen(key: str) -> int:
            if key == "celery:queue:planning":
                raise ConnectionError("db miss")
            if key == "planning":
                return 42
            return 0

        mock_redis_for_monitoring.llen = AsyncMock(side_effect=_llen)
        detector = CongestionDetector(queues=("planning",))
        result = await detector.queue_lengths()
        assert result == {"planning": 42}

    async def test_returns_minus_one_on_complete_failure(self, mock_redis_for_monitoring):
        mock_redis_for_monitoring.llen = AsyncMock(side_effect=ConnectionError("down"))
        detector = CongestionDetector(queues=("default",))
        result = await detector.queue_lengths()
        assert result == {"default": -1}


class TestActiveSessions:
    async def test_counts_scanned_session_keys(self, mock_redis_for_monitoring):
        mock_redis_for_monitoring.scan = AsyncMock(
            side_effect=[
                (1, ["sse:session:a", "sse:session:b"]),
                (0, ["sse:session:c"]),
            ]
        )
        detector = CongestionDetector()
        assert await detector.active_sessions() == 3

    async def test_returns_zero_when_no_sessions(self, mock_redis_for_monitoring):
        mock_redis_for_monitoring.scan = AsyncMock(return_value=(0, []))
        detector = CongestionDetector()
        assert await detector.active_sessions() == 0

    async def test_returns_zero_on_redis_failure(self, mock_redis_for_monitoring):
        mock_redis_for_monitoring.scan = AsyncMock(side_effect=ConnectionError("down"))
        detector = CongestionDetector()
        assert await detector.active_sessions() == 0


class TestErrorRate:
    def test_zero_when_no_requests(self):
        registry = CollectorRegistry()
        counter = Counter(
            "test_http_requests_total",
            "test",
            ["method", "endpoint", "status_code"],
            registry=registry,
        )
        detector = CongestionDetector(http_requests_counter=counter)
        assert detector.error_rate() == 0.0

    def test_all_time_error_ratio(self):
        registry = CollectorRegistry()
        counter = Counter(
            "test_http_requests_total",
            "test",
            ["method", "endpoint", "status_code"],
            registry=registry,
        )
        counter.labels("GET", "/health", "200").inc(8)
        counter.labels("GET", "/health", "500").inc(2)
        detector = CongestionDetector(http_requests_counter=counter)
        assert detector.error_rate() == 0.2


class TestP99Latency:
    def test_zero_when_no_observations(self):
        registry = CollectorRegistry()
        histogram = Histogram(
            "test_http_duration_seconds",
            "test",
            ["method", "endpoint"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0),
            registry=registry,
        )
        detector = CongestionDetector(http_duration_histogram=histogram)
        assert detector.p99_latency() == 0.0

    def test_approximate_p99(self):
        registry = CollectorRegistry()
        histogram = Histogram(
            "test_http_duration_seconds_2",
            "test",
            ["method", "endpoint"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0),
            registry=registry,
        )
        for _ in range(90):
            histogram.labels("GET", "/x").observe(0.05)
        for _ in range(10):
            histogram.labels("GET", "/x").observe(1.5)
        detector = CongestionDetector(http_duration_histogram=histogram)
        p99 = detector.p99_latency()
        assert 1.0 <= p99 <= 2.0


class TestDetect:
    async def test_normal_state(self, mock_redis_for_monitoring):
        mock_redis_for_monitoring.llen = AsyncMock(return_value=5)
        mock_redis_for_monitoring.scan = AsyncMock(return_value=(0, []))
        detector = CongestionDetector()
        result = await detector.detect()
        assert result["congested"] is False
        assert result["score"] < 0.5
        assert result["details"]["max_queue_length"] == 5
        assert result["details"]["active_sessions"] == 0
        assert result["details"]["error_rate"] == 0.0
        assert result["details"]["p99_latency_seconds"] == 0.0

    async def test_congested_by_queue_length(self, mock_redis_for_monitoring):
        mock_redis_for_monitoring.llen = AsyncMock(return_value=120)
        detector = CongestionDetector()
        result = await detector.detect()
        assert result["congested"] is True
        assert result["score"] > 0.7

    async def test_congested_by_critical_error_rate(self, mock_redis_for_monitoring):
        registry = CollectorRegistry()
        counter = Counter(
            "test_http_requests_total",
            "test",
            ["method", "endpoint", "status_code"],
            registry=registry,
        )
        counter.labels("GET", "/x", "200").inc(1)
        counter.labels("GET", "/x", "503").inc(9)
        detector = CongestionDetector(http_requests_counter=counter)
        result = await detector.detect()
        assert result["congested"] is True
        assert result["details"]["error_rate"] == 0.9

    async def test_redis_failure_produces_defaults(self, mock_redis_for_monitoring):
        mock_redis_for_monitoring.llen = AsyncMock(side_effect=ConnectionError("down"))
        mock_redis_for_monitoring.scan = AsyncMock(side_effect=ConnectionError("down"))
        detector = CongestionDetector()
        result = await detector.detect()
        assert result["congested"] is False
        assert result["details"]["queue_lengths"] == {
            "default": -1,
            "planning": -1,
            "memory": -1,
            "planning_dead_letter": -1,
        }
        assert result["details"]["active_sessions"] == 0
