"""Congestion detector for queues, sessions, errors, and latency."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from prometheus_client import Counter, Histogram

from core.metrics import HTTP_DURATION, HTTP_REQUESTS
from core.redis_client import redis_client

_DEFAULT_QUEUES = ("default", "planning", "memory", "planning_dead_letter")
_METRIC_HISTOGRAMS = {
    "http_duration": HTTP_DURATION,
    "llm_request_duration": None,  # populated lazily to avoid import cycles
}


def _get_llm_duration_histogram() -> Histogram:
    # Lazy import keeps startup lightweight and avoids circular imports.
    from core.metrics import LLM_REQUEST_DURATION

    return LLM_REQUEST_DURATION


class CongestionDetector:
    """Detect backend congestion from Redis queues and Prometheus metrics."""

    def __init__(
        self,
        queues: Optional[tuple[str, ...]] = None,
        queue_threshold: int = 100,
        session_threshold: int = 1000,
        error_rate_threshold: float = 0.1,
        p99_threshold_seconds: float = 2.0,
        critical_queue_length: int = 500,
        critical_error_rate: float = 0.5,
        critical_p99_seconds: float = 5.0,
        critical_session_count: int = 5000,
        http_requests_counter: Optional[Counter] = None,
        http_duration_histogram: Optional[Histogram] = None,
    ):
        self.queues = queues or _DEFAULT_QUEUES
        self.queue_threshold = queue_threshold
        self.session_threshold = session_threshold
        self.error_rate_threshold = error_rate_threshold
        self.p99_threshold_seconds = p99_threshold_seconds
        self.critical_queue_length = critical_queue_length
        self.critical_error_rate = critical_error_rate
        self.critical_p99_seconds = critical_p99_seconds
        self.critical_session_count = critical_session_count
        self._http_requests = http_requests_counter or HTTP_REQUESTS
        self._http_duration = http_duration_histogram or HTTP_DURATION

    async def queue_lengths(self) -> dict[str, int]:
        """Return Celery queue lengths from Redis.

        Tries ``celery:queue:<name>`` first (Kombu default naming) and falls
        back to the bare queue name. Returns ``-1`` for a queue that cannot be
        read.
        """
        lengths: dict[str, int] = {}
        for name in self.queues:
            length = -1
            for key in (f"celery:queue:{name}", name):
                try:
                    length = int(await redis_client.llen(key))
                    break
                except Exception:
                    continue
            lengths[name] = length
        return lengths

    async def active_sessions(self) -> int:
        """Count active SSE session keys in Redis."""
        try:
            cursor = 0
            total = 0
            while True:
                cursor, keys = await redis_client.scan(
                    cursor=cursor,
                    match="sse:session:*",
                    count=100,
                )
                total += len(keys)
                if cursor == 0:
                    break
            return total
        except Exception:
            return 0

    def error_rate(self, window_s: int = 300) -> float:
        """Return the ratio of HTTP 5xx responses to total HTTP requests.

        The ``window_s`` parameter is reserved for future sliding-window
        implementations; the MVP uses the all-time ratio from process-local
        Prometheus counters.
        """
        del window_s  # MVP uses current counter values.
        try:
            total = 0
            errors = 0
            metrics: dict[tuple[str, ...], Counter] = getattr(
                self._http_requests, "_metrics", {}
            )
            for labels, child in metrics.items():
                value_obj = getattr(child, "_value", None)
                value = float(value_obj.get()) if value_obj is not None else 0.0
                total += value
                status = labels[-1] if labels else ""
                if status.isdigit() and int(status) >= 500:
                    errors += value
            if total == 0:
                return 0.0
            return errors / total
        except Exception:
            return 0.0

    def p99_latency(self, metric_name: str = "http_duration") -> float:
        """Approximate P99 latency from a Prometheus histogram.

        Uses the histogram's cumulative buckets and sum. Returns ``0.0`` when
        no observations have been recorded.
        """
        try:
            if metric_name == "http_duration":
                histogram = self._http_duration
            else:
                histogram = self._resolve_histogram(metric_name)
            if histogram is None:
                return 0.0

            metrics: dict[tuple[str, ...], Histogram] = getattr(
                histogram, "_metrics", {}
            )
            if not metrics:
                return 0.0

            merged_buckets: Optional[list[float]] = None
            buckets = list(getattr(histogram, "_upper_bounds", []))

            for child in metrics.values():
                child_buckets = getattr(child, "_buckets", [])
                if merged_buckets is None:
                    merged_buckets = [float(v.get()) for v in child_buckets]
                else:
                    for i, value in enumerate(child_buckets):
                        if i < len(merged_buckets):
                            merged_buckets[i] += float(value.get())

            if merged_buckets is None or not buckets:
                return 0.0

            # Buckets in this prometheus_client version store per-bucket counts;
            # convert to cumulative counts before computing the percentile.
            for i in range(1, len(merged_buckets)):
                merged_buckets[i] += merged_buckets[i - 1]

            total_count = merged_buckets[-1]
            if total_count == 0:
                return 0.0

            target = total_count * 0.99
            for i, count in enumerate(merged_buckets):
                if count >= target:
                    upper = float(buckets[i])
                    if i == 0 or merged_buckets[i - 1] >= target:
                        return upper
                    lower = float(buckets[i - 1])
                    prev_count = merged_buckets[i - 1]
                    fraction = (target - prev_count) / (count - prev_count)
                    return lower + (upper - lower) * fraction

            return float(buckets[-1])
        except Exception:
            return 0.0

    def _resolve_histogram(self, metric_name: str) -> Optional[Histogram]:
        if metric_name == "llm_request_duration":
            return _get_llm_duration_histogram()
        return _METRIC_HISTOGRAMS.get(metric_name)

    async def detect(self) -> dict:
        """Aggregate signals and return a congestion verdict."""
        queue_lengths = await self.queue_lengths()
        sessions = await self.active_sessions()
        errors = self.error_rate()
        p99 = self.p99_latency()

        max_queue = max((length for length in queue_lengths.values() if length >= 0), default=0)

        queue_score = min(max_queue / self.queue_threshold, 1.0)
        session_score = min(sessions / self.session_threshold, 1.0)
        error_score = min(errors / self.error_rate_threshold, 1.0) if self.error_rate_threshold else 0.0
        latency_score = min(p99 / self.p99_threshold_seconds, 1.0) if self.p99_threshold_seconds else 0.0

        # If any dimension exceeds its threshold it saturates at 1.0, making
        # the detector sensitive to single-axis overload while still keeping
        # the score in the 0-1 range.
        score = max(queue_score, session_score, error_score, latency_score)

        critical = (
            max_queue > self.critical_queue_length
            or errors > self.critical_error_rate
            or p99 > self.critical_p99_seconds
            or sessions > self.critical_session_count
        )
        congested = score > 0.7 or critical

        return {
            "congested": congested,
            "score": round(score, 4),
            "details": {
                "queue_lengths": queue_lengths,
                "max_queue_length": max_queue,
                "active_sessions": sessions,
                "error_rate": errors,
                "p99_latency_seconds": p99,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
