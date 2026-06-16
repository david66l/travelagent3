"""Tests for v1 API endpoints."""

from uuid import uuid4


def _headers():
    return {"X-Device-Fingerprint": str(uuid4())}


class TestUserEndpoints:
    """Tests for /api/v1/users endpoints."""

    def test_get_me(self, client):
        response = client.get("/api/v1/users/me", headers=_headers())
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["role"] == "guest"

    def test_update_and_get_profile(self, client):
        headers = _headers()
        put_response = client.put(
            "/api/v1/users/me/profile",
            json={"preferences": {"pace": "relaxed"}},
            headers=headers,
        )
        assert put_response.status_code == 200
        data = put_response.json()
        assert data["success"] is True
        assert data["data"]["preferences"]["pace"] == "relaxed"

        get_response = client.get(
            "/api/v1/users/me/profile",
            headers=headers,
        )
        assert get_response.status_code == 200
        assert get_response.json()["data"]["preferences"]["pace"] == "relaxed"


class TestConversationEndpoints:
    """Tests for /api/v1/conversations endpoints."""

    def test_create_and_list(self, client):
        headers = _headers()
        create_response = client.post(
            "/api/v1/conversations",
            json={"title": "杭州3日游"},
            headers=headers,
        )
        assert create_response.status_code == 201
        data = create_response.json()
        assert data["success"] is True
        assert data["data"]["title"] == "杭州3日游"
        conversation_id = data["data"]["id"]

        list_response = client.get("/api/v1/conversations", headers=headers)
        assert list_response.status_code == 200
        assert len(list_response.json()["data"]) == 1

        get_response = client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=headers,
        )
        assert get_response.status_code == 200
        assert get_response.json()["data"]["id"] == conversation_id

    def test_archive_conversation(self, client):
        headers = _headers()
        create_response = client.post(
            "/api/v1/conversations",
            json={"title": "Archived"},
            headers=headers,
        )
        conversation_id = create_response.json()["data"]["id"]

        archive_response = client.post(
            f"/api/v1/conversations/{conversation_id}/archive",
            headers=headers,
        )
        assert archive_response.status_code == 200
        assert archive_response.json()["message"] == "Conversation archived"

    def test_messages(self, client):
        headers = _headers()
        create_response = client.post(
            "/api/v1/conversations",
            json={"title": "Messages"},
            headers=headers,
        )
        conversation_id = create_response.json()["data"]["id"]

        msg_response = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"role": "user", "content": "你好", "token_count": 2},
            headers=headers,
        )
        assert msg_response.status_code == 201
        assert msg_response.json()["data"]["content"] == "你好"

        list_response = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            headers=headers,
        )
        assert list_response.status_code == 200
        assert len(list_response.json()["data"]) == 1


class TestItineraryEndpoints:
    """Tests for /api/v1/itineraries endpoints."""

    def test_create_and_list(self, client):
        headers = _headers()
        conv_response = client.post(
            "/api/v1/conversations",
            json={"title": "Itinerary"},
            headers=headers,
        )
        conversation_id = conv_response.json()["data"]["id"]

        create_response = client.post(
            "/api/v1/itineraries",
            json={
                "conversation_id": conversation_id,
                "destination": "杭州",
                "days": 3,
            },
            headers=headers,
        )
        assert create_response.status_code == 201
        data = create_response.json()
        assert data["success"] is True
        assert data["data"]["destination"] == "杭州"

        list_response = client.get("/api/v1/itineraries", headers=headers)
        assert list_response.status_code == 200
        assert len(list_response.json()["data"]) == 1


class TestPlanningJobEndpoints:
    """Tests for /api/v1/planning-jobs endpoints."""

    def test_create_and_get(self, client):
        headers = _headers()
        conv_response = client.post(
            "/api/v1/conversations",
            json={"title": "Job"},
            headers=headers,
        )
        conversation_id = conv_response.json()["data"]["id"]

        create_response = client.post(
            "/api/v1/planning-jobs",
            json={
                "conversation_id": conversation_id,
                "input_requirements": {"destination": "杭州"},
            },
            headers=headers,
        )
        assert create_response.status_code == 201
        data = create_response.json()
        assert data["success"] is True
        job_id = data["data"]["id"]

        get_response = client.get(
            f"/api/v1/planning-jobs/{job_id}",
            headers=headers,
        )
        assert get_response.status_code == 200
        assert get_response.json()["data"]["id"] == job_id


class TestAuthorizationBranches:
    """Tests for ownership / cross-user access."""

    def test_conversation_belongs_to_another_user(self, client):
        headers_a = _headers()
        create_a = client.post(
            "/api/v1/conversations",
            json={"title": "Mine"},
            headers=headers_a,
        )
        conversation_id = create_a.json()["data"]["id"]

        headers_b = _headers()
        response = client.get(
            f"/api/v1/conversations/{conversation_id}",
            headers=headers_b,
        )
        assert response.status_code == 404

    def test_itinerary_belongs_to_another_user(self, client):
        headers_a = _headers()
        conv_a = client.post(
            "/api/v1/conversations",
            json={"title": "Mine"},
            headers=headers_a,
        )
        itin_a = client.post(
            "/api/v1/itineraries",
            json={
                "conversation_id": conv_a.json()["data"]["id"],
                "destination": "杭州",
                "days": 3,
            },
            headers=headers_a,
        )
        itinerary_id = itin_a.json()["data"]["id"]

        headers_b = _headers()
        response = client.post(
            f"/api/v1/itineraries/{itinerary_id}/favorite",
            json={"is_favorite": True},
            headers=headers_b,
        )
        assert response.status_code == 404


class TestErrorHandling:
    """Tests for unified error responses."""

    def test_not_found(self, client):
        headers = _headers()
        response = client.get(
            f"/api/v1/conversations/{uuid4()}",
            headers=headers,
        )
        assert response.status_code == 404
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "NOT_FOUND"

    def test_not_found_archive(self, client):
        headers = _headers()
        response = client.post(
            f"/api/v1/conversations/{uuid4()}/archive",
            headers=headers,
        )
        assert response.status_code == 404

    def test_not_found_messages(self, client):
        headers = _headers()
        response = client.get(
            f"/api/v1/conversations/{uuid4()}/messages",
            headers=headers,
        )
        assert response.status_code == 404

    def test_not_found_planning_job(self, client):
        headers = _headers()
        response = client.get(
            "/api/v1/planning-jobs/not-a-real-id",
            headers=headers,
        )
        assert response.status_code == 404

    def test_planning_job_cross_user(self, client):
        headers_a = _headers()
        conv_a = client.post(
            "/api/v1/conversations",
            json={"title": "Job"},
            headers=headers_a,
        )
        job_a = client.post(
            "/api/v1/planning-jobs",
            json={"conversation_id": conv_a.json()["data"]["id"]},
            headers=headers_a,
        )
        job_id = job_a.json()["data"]["id"]

        headers_b = _headers()
        response = client.get(
            f"/api/v1/planning-jobs/{job_id}",
            headers=headers_b,
        )
        assert response.status_code == 404

    def test_user_not_found(self, client):
        response = client.get(f"/api/v1/users/{uuid4()}")
        assert response.status_code == 404

    def test_validation_error(self, client):
        headers = _headers()
        response = client.post(
            "/api/v1/conversations",
            json={"title": 123},  # title should be string
            headers=headers,
        )
        assert response.status_code == 422
        data = response.json()
        assert data["success"] is False
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_invalid_message_role(self, client):
        headers = _headers()
        conv = client.post(
            "/api/v1/conversations",
            json={"title": "Role"},
            headers=headers,
        )
        conversation_id = conv.json()["data"]["id"]
        response = client.post(
            f"/api/v1/conversations/{conversation_id}/messages",
            json={"role": "invalid", "content": "hello"},
            headers=headers,
        )
        assert response.status_code == 422

    def test_invalid_itinerary_days(self, client):
        headers = _headers()
        conv = client.post(
            "/api/v1/conversations",
            json={"title": "Days"},
            headers=headers,
        )
        response = client.post(
            "/api/v1/itineraries",
            json={
                "conversation_id": conv.json()["data"]["id"],
                "destination": "杭州",
                "days": 99,
            },
            headers=headers,
        )
        assert response.status_code == 422
