"""Tests for admin analytics endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

from uuid import uuid4

from api.deps import require_admin


def _headers():
    return {"X-Device-Fingerprint": str(uuid4())}


class TestAnalyticsEndpoint:
    """Tests for GET /api/v1/admin/analytics."""

    def test_analytics_requires_admin(self, client):
        response = client.get("/api/v1/admin/analytics", headers=_headers())
        assert response.status_code == 403

    def test_analytics_returns_report(self, client):
        client.app.dependency_overrides[require_admin] = lambda: MagicMock()
        with patch(
            "monitoring.log_analytics.LogAnalyticsEngine.analyze",
            new=AsyncMock(
                return_value={
                    "planning_failures": [],
                    "modification_intents": [],
                    "destination_ranking": [],
                    "iteration_suggestions": [],
                }
            ),
        ):
            response = client.get("/api/v1/admin/analytics", headers=_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["planning_failures"] == []
