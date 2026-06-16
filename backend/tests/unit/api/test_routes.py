"""Tests for lightweight REST API routes."""


class TestHealthCheck:
    """Test health endpoint."""

    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "travel-agent"}


class TestCreateSession:
    """Test session creation."""

    def test_create_session(self, client):
        response = client.post("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["message"] == "Session created"
