"""Tests for user privacy endpoint."""

from unittest.mock import AsyncMock, patch

from uuid import uuid4


def _headers():
    return {"X-Device-Fingerprint": str(uuid4())}


class TestPrivacyEndpoint:
    """Tests for DELETE /api/v1/users/me/data."""

    def test_delete_my_data(self, client):
        with patch(
            "privacy.delete_all_user_data",
            new=AsyncMock(return_value={"users": 1, "conversations": 2}),
        ):
            response = client.delete("/api/v1/users/me/data", headers=_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["conversations"] == 2
