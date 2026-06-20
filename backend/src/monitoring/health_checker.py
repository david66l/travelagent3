"""Third-party and infrastructure health checks."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Optional

import httpx
from sqlalchemy import text

from core.database import engine as async_engine
from core.redis_client import redis_client
from core.settings import settings


class _HttpResponse:
    """Minimal response wrapper for health-check HTTP calls.

    This exists so tests can mock ``ThirdPartyHealthChecker._http_get`` with a
    plain object instead of a full ``httpx.Response``.
    """

    def __init__(
        self,
        status_code: int,
        json_data: Optional[dict] = None,
        response: Optional[httpx.Response] = None,
    ):
        self.status_code = status_code
        self._json_data = json_data
        self._response = response

    async def json(self) -> dict:
        if self._json_data is not None:
            return self._json_data
        if self._response is not None:
            try:
                return self._response.json()
            except Exception:
                return {}
        return {}


class ThirdPartyHealthChecker:
    """Run lightweight health probes against external APIs and backing stores."""

    def __init__(self, http_timeout: float = 5.0) -> None:
        self.http_timeout = http_timeout

    @staticmethod
    async def _http_get(url: str, *, timeout: float) -> _HttpResponse:
        """Execute an HTTP GET and return a minimal response object."""
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
        return _HttpResponse(response.status_code, response=response)

    def _result(
        self,
        name: str,
        status: str,
        latency_ms: int,
        error: Optional[str] = None,
    ) -> dict:
        return {
            "name": name,
            "status": status,
            "latency_ms": latency_ms,
            "error": error,
        }

    async def check_amap(self) -> dict:
        """Probe the AMap (Gaode) geocoding API."""
        start = time.perf_counter()
        if not settings.amap_key:
            return self._result("amap", "unknown", int((time.perf_counter() - start) * 1000))

        url = (
            "https://restapi.amap.com/v3/config/district"
            f"?key={settings.amap_key}&keywords=北京&subdistrict=0"
        )
        try:
            response = await self._http_get(url, timeout=self.http_timeout)
            latency_ms = int((time.perf_counter() - start) * 1000)
            if response.status_code == 200:
                data = await response.json()
                if data.get("status") == "1":
                    return self._result("amap", "healthy", latency_ms)
                return self._result(
                    "amap", "degraded", latency_ms, error="amap returned non-success status"
                )
            return self._result(
                "amap", "unhealthy", latency_ms, error=f"http {response.status_code}"
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - start) * 1000)
            return self._result("amap", "degraded", latency_ms, error=str(exc))

    async def check_weather(self) -> dict:
        """Probe a public weather endpoint.

        This is a placeholder implementation that uses OpenWeatherMap when a
        ``weather_key`` is configured. Replace with the actual provider contract
        once it is decided.
        """
        start = time.perf_counter()
        if not settings.weather_key:
            return self._result("weather", "unknown", int((time.perf_counter() - start) * 1000))

        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?q=Beijing&appid={settings.weather_key}&units=metric"
        )
        try:
            response = await self._http_get(url, timeout=self.http_timeout)
            latency_ms = int((time.perf_counter() - start) * 1000)
            if response.status_code == 200:
                return self._result("weather", "healthy", latency_ms)
            return self._result(
                "weather", "degraded", latency_ms, error=f"http {response.status_code}"
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - start) * 1000)
            return self._result("weather", "degraded", latency_ms, error=str(exc))

    async def check_vllm(self) -> dict:
        """Probe the local vLLM inference server."""
        start = time.perf_counter()
        base_url = settings.vllm_base_url.replace("/v1", "").rstrip("/")
        if not base_url:
            return self._result("vllm", "unknown", int((time.perf_counter() - start) * 1000))

        url = f"{base_url}/health"
        try:
            response = await self._http_get(url, timeout=self.http_timeout)
            latency_ms = int((time.perf_counter() - start) * 1000)
            if response.status_code == 200:
                return self._result("vllm", "healthy", latency_ms)
            return self._result(
                "vllm", "unhealthy", latency_ms, error=f"http {response.status_code}"
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - start) * 1000)
            return self._result("vllm", "unhealthy", latency_ms, error=str(exc))

    async def check_postgres(self) -> dict:
        """Probe PostgreSQL via the async SQLAlchemy engine."""
        start = time.perf_counter()
        try:
            async with async_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            latency_ms = int((time.perf_counter() - start) * 1000)
            return self._result("postgres", "healthy", latency_ms)
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - start) * 1000)
            return self._result("postgres", "unhealthy", latency_ms, error=str(exc))

    async def _ping_redis(self) -> bool:
        """Ping Redis, falling back to a GET if the wrapper lacks ``ping``."""
        try:
            return bool(await redis_client.ping())
        except AttributeError:
            # The project RedisClient does not expose ``ping``; use the raw client.
            if redis_client._client is None:
                await redis_client.connect()
            return bool(await redis_client._client.ping())

    async def check_redis(self) -> dict:
        """Probe Redis connectivity."""
        start = time.perf_counter()
        try:
            ok = await self._ping_redis()
            latency_ms = int((time.perf_counter() - start) * 1000)
            if ok:
                return self._result("redis", "healthy", latency_ms)
            return self._result("redis", "unhealthy", latency_ms, error="ping returned false")
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - start) * 1000)
            return self._result("redis", "unhealthy", latency_ms, error=str(exc))

    async def health_report(self) -> dict:
        """Run all checks concurrently and return an aggregated report."""
        checks = await asyncio.gather(
            self.check_amap(),
            self.check_weather(),
            self.check_vllm(),
            self.check_postgres(),
            self.check_redis(),
            return_exceptions=True,
        )

        results: list[dict] = []
        for item in checks:
            if isinstance(item, Exception):
                results.append(
                    {
                        "name": "unknown",
                        "status": "unhealthy",
                        "latency_ms": 0,
                        "error": f"unhandled exception: {item}",
                    }
                )
            else:
                results.append(item)

        healthy = all(result["status"] in ("healthy", "unknown") for result in results)

        return {
            "healthy": healthy,
            "checks": results,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
