"""Unit tests for ``monitoring.health_checker.ThirdPartyHealthChecker``."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from monitoring.health_checker import ThirdPartyHealthChecker, _HttpResponse
import monitoring.health_checker as health_checker


def _fake_response(status_code: int, json_data: dict | None = None) -> _HttpResponse:
    return _HttpResponse(status_code, json_data=json_data)


@pytest.fixture
def checker() -> ThirdPartyHealthChecker:
    return ThirdPartyHealthChecker()


@pytest.fixture
def mock_http_get(monkeypatch):
    """Replace the class-level HTTP helper so no real network calls are made."""
    mock = AsyncMock()
    monkeypatch.setattr(ThirdPartyHealthChecker, "_http_get", mock)
    return mock


@pytest.fixture
def mock_postgres(monkeypatch):
    """Provide a mocked async SQLAlchemy engine."""
    conn = AsyncMock()
    context_manager = AsyncMock()
    context_manager.__aenter__ = AsyncMock(return_value=conn)
    context_manager.__aexit__ = AsyncMock(return_value=False)

    engine = MagicMock()
    engine.connect = MagicMock(return_value=context_manager)
    monkeypatch.setattr(health_checker, "async_engine", engine)
    return engine, conn


@pytest.fixture
def mock_redis_ping(monkeypatch):
    """Make Redis always respond with a successful ping."""
    ping = AsyncMock(return_value=True)
    monkeypatch.setattr(health_checker.redis_client, "ping", ping)
    return ping


def _configure_http_get(
    mock,
    *,
    amap_status: int = 200,
    amap_json: dict | None = None,
    weather_status: int = 200,
    vllm_status: int = 200,
):
    amap_json = amap_json or {"status": "1"}

    async def _side_effect(url: str, *, timeout: float):
        if "amap" in url:
            return _fake_response(amap_status, amap_json)
        if "openweathermap" in url:
            return _fake_response(weather_status)
        if url.endswith("/health"):
            return _fake_response(vllm_status)
        return _fake_response(404)

    mock.side_effect = _side_effect


class TestCheckAmap:
    async def test_healthy(self, checker, mock_http_get, monkeypatch):
        monkeypatch.setattr(health_checker.settings, "amap_key", "test-amap-key")
        _configure_http_get(mock_http_get)

        result = await checker.check_amap()

        assert result["name"] == "amap"
        assert result["status"] == "healthy"
        assert result["error"] is None
        assert isinstance(result["latency_ms"], int)

    async def test_degraded_when_amap_reports_failure(self, checker, mock_http_get, monkeypatch):
        monkeypatch.setattr(health_checker.settings, "amap_key", "test-amap-key")
        _configure_http_get(mock_http_get, amap_status=200, amap_json={"status": "0"})

        result = await checker.check_amap()

        assert result["status"] == "degraded"
        assert result["error"] is not None

    async def test_unhealthy_on_http_error(self, checker, mock_http_get, monkeypatch):
        monkeypatch.setattr(health_checker.settings, "amap_key", "test-amap-key")
        _configure_http_get(mock_http_get, amap_status=503)

        result = await checker.check_amap()

        assert result["status"] == "unhealthy"
        assert "503" in result["error"]

    async def test_unknown_when_key_missing(self, checker, monkeypatch):
        monkeypatch.setattr(health_checker.settings, "amap_key", "")

        result = await checker.check_amap()

        assert result["status"] == "unknown"
        assert result["error"] is None


class TestCheckWeather:
    async def test_healthy(self, checker, mock_http_get, monkeypatch):
        monkeypatch.setattr(health_checker.settings, "weather_key", "test-weather-key")
        _configure_http_get(mock_http_get)

        result = await checker.check_weather()

        assert result["name"] == "weather"
        assert result["status"] == "healthy"

    async def test_degraded_on_http_error(self, checker, mock_http_get, monkeypatch):
        monkeypatch.setattr(health_checker.settings, "weather_key", "test-weather-key")
        _configure_http_get(mock_http_get, weather_status=500)

        result = await checker.check_weather()

        assert result["status"] == "degraded"

    async def test_unknown_when_key_missing(self, checker, monkeypatch):
        monkeypatch.setattr(health_checker.settings, "weather_key", "")

        result = await checker.check_weather()

        assert result["status"] == "unknown"


class TestCheckVllm:
    async def test_healthy(self, checker, mock_http_get, monkeypatch):
        monkeypatch.setattr(health_checker.settings, "vllm_base_url", "http://vllm:8000/v1")
        _configure_http_get(mock_http_get)

        result = await checker.check_vllm()

        assert result["name"] == "vllm"
        assert result["status"] == "healthy"
        assert result["error"] is None

    async def test_unhealthy_on_bad_status(self, checker, mock_http_get, monkeypatch):
        monkeypatch.setattr(health_checker.settings, "vllm_base_url", "http://vllm:8000/v1")
        _configure_http_get(mock_http_get, vllm_status=503)

        result = await checker.check_vllm()

        assert result["status"] == "unhealthy"
        assert "503" in result["error"]

    async def test_unknown_when_base_url_missing(self, checker, monkeypatch):
        monkeypatch.setattr(health_checker.settings, "vllm_base_url", "")

        result = await checker.check_vllm()

        assert result["status"] == "unknown"


class TestCheckPostgres:
    async def test_healthy(self, checker, mock_postgres):
        result = await checker.check_postgres()

        assert result["name"] == "postgres"
        assert result["status"] == "healthy"
        assert result["error"] is None

    async def test_unhealthy_on_execute_failure(self, checker, mock_postgres):
        _, conn = mock_postgres
        conn.execute = AsyncMock(side_effect=RuntimeError("connection refused"))

        result = await checker.check_postgres()

        assert result["status"] == "unhealthy"
        assert "connection refused" in result["error"]


class TestCheckRedis:
    async def test_healthy(self, checker, mock_redis_ping):
        result = await checker.check_redis()

        assert result["name"] == "redis"
        assert result["status"] == "healthy"
        assert result["error"] is None

    async def test_unhealthy_when_ping_false(self, checker, monkeypatch):
        monkeypatch.setattr(health_checker.redis_client, "ping", AsyncMock(return_value=False))

        result = await checker.check_redis()

        assert result["status"] == "unhealthy"
        assert "false" in result["error"].lower()


class TestHealthReport:
    async def test_overall_healthy(
        self, checker, mock_http_get, mock_postgres, mock_redis_ping, monkeypatch
    ):
        monkeypatch.setattr(health_checker.settings, "amap_key", "test-amap-key")
        monkeypatch.setattr(health_checker.settings, "weather_key", "test-weather-key")
        monkeypatch.setattr(health_checker.settings, "vllm_base_url", "http://vllm:8000/v1")
        _configure_http_get(mock_http_get)

        report = await checker.health_report()

        assert report["healthy"] is True
        assert len(report["checks"]) == 5
        assert all(check["status"] in ("healthy", "unknown") for check in report["checks"])
        assert "timestamp" in report

    async def test_one_degraded_makes_overall_unhealthy(
        self, checker, mock_http_get, mock_postgres, mock_redis_ping, monkeypatch
    ):
        monkeypatch.setattr(health_checker.settings, "amap_key", "test-amap-key")
        monkeypatch.setattr(health_checker.settings, "weather_key", "test-weather-key")
        monkeypatch.setattr(health_checker.settings, "vllm_base_url", "http://vllm:8000/v1")
        _configure_http_get(mock_http_get, amap_json={"status": "0"})

        report = await checker.health_report()

        assert report["healthy"] is False
        amap_check = next(check for check in report["checks"] if check["name"] == "amap")
        assert amap_check["status"] == "degraded"

    async def test_all_unhealthy(
        self,
        checker,
        mock_http_get,
        mock_postgres,
        mock_redis_ping,
        monkeypatch,
    ):
        monkeypatch.setattr(health_checker.settings, "amap_key", "test-amap-key")
        monkeypatch.setattr(health_checker.settings, "weather_key", "test-weather-key")
        monkeypatch.setattr(health_checker.settings, "vllm_base_url", "http://vllm:8000/v1")

        async def _failing_side_effect(url: str, *, timeout: float):
            raise RuntimeError("network unreachable")

        mock_http_get.side_effect = _failing_side_effect

        # Force Postgres and Redis to fail as well.
        engine, _ = mock_postgres
        engine.connect.side_effect = RuntimeError("postgres down")
        mock_redis_ping.return_value = False

        report = await checker.health_report()

        assert report["healthy"] is False
        assert all(
            check["status"] in ("degraded", "unhealthy", "unknown") for check in report["checks"]
        )

    async def test_missing_keys_report_unknown(
        self, checker, mock_http_get, mock_postgres, mock_redis_ping, monkeypatch
    ):
        monkeypatch.setattr(health_checker.settings, "amap_key", "")
        monkeypatch.setattr(health_checker.settings, "weather_key", "")
        monkeypatch.setattr(health_checker.settings, "vllm_base_url", "http://vllm:8000/v1")
        _configure_http_get(mock_http_get)

        report = await checker.health_report()

        assert report["healthy"] is True
        amap_check = next(check for check in report["checks"] if check["name"] == "amap")
        weather_check = next(check for check in report["checks"] if check["name"] == "weather")
        assert amap_check["status"] == "unknown"
        assert weather_check["status"] == "unknown"
