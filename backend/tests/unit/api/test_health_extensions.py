"""Tests for extended health endpoints."""

from unittest.mock import AsyncMock, patch


class TestHealthExtensions:
    """Tests for /api/health/dependencies and /api/health/congestion."""

    def test_dependency_health(self, client):
        with patch(
            "monitoring.health_checker.ThirdPartyHealthChecker.health_report",
            new=AsyncMock(
                return_value={
                    "healthy": True,
                    "checks": [{"name": "redis", "status": "healthy"}],
                }
            ),
        ):
            response = client.get("/api/health/dependencies")
        assert response.status_code == 200
        data = response.json()
        assert data["healthy"] is True

    def test_congestion_status(self, client):
        with patch(
            "monitoring.congestion_detector.CongestionDetector.detect",
            new=AsyncMock(
                return_value={
                    "congested": False,
                    "score": 0.1,
                    "details": {},
                }
            ),
        ):
            response = client.get("/api/health/congestion")
        assert response.status_code == 200
        data = response.json()
        assert data["congested"] is False

    def test_readiness_failure_is_sanitized(self, client):
        with patch(
            "api.health.redis_client._client.ping",
            new=AsyncMock(side_effect=ConnectionError("redis://user:secret@internal:6379")),
        ):
            response = client.get("/api/ready")

        assert response.status_code == 503
        data = response.json()
        assert data["redis"] == "unavailable"
        assert "secret" not in response.text
