"""Tests for SSE chat endpoints."""

from uuid import uuid4


def _guest_headers(client, fingerprint=None):
    fp = fingerprint or str(uuid4())
    guest_response = client.post(
        "/api/v1/auth/guest",
        json={"device_fingerprint": fp},
    )
    token = guest_response.json()["data"]["access_token"]
    return {
        "Authorization": f"Bearer {token}",
        "X-Device-Fingerprint": fp,
    }


def _create_conversation(client, headers):
    response = client.post(
        "/api/v1/conversations",
        json={"title": "SSE test"},
        headers=headers,
    )
    assert response.status_code == 201
    return response.json()["data"]["id"]


class TestChatSSE:
    def test_post_message_accepted(self, client):
        headers = _guest_headers(client)
        conversation_id = _create_conversation(client, headers)

        response = client.post(
            "/api/v1/chat/message",
            json={
                "conversation_id": conversation_id,
                "content": "想去杭州玩三天",
                "stream": True,
            },
            headers=headers,
        )
        assert response.status_code == 202
        body = response.json()
        assert body["success"] is True
        assert body["data"]["status"] == "accepted"
        assert body["data"]["conversation_id"] == conversation_id

    def test_stream_opens(self, client):
        headers = _guest_headers(client)
        conversation_id = _create_conversation(client, headers)

        with client.stream(
            "GET",
            f"/api/v1/chat/stream?conversation_id={conversation_id}&timeout=120",
            headers=headers,
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            for chunk in response.iter_bytes():
                if chunk:
                    break

    def test_stream_forbidden_for_other_user(self, client):
        owner_headers = _guest_headers(client)
        conversation_id = _create_conversation(client, owner_headers)
        stranger_headers = _guest_headers(client)

        response = client.get(
            f"/api/v1/chat/stream?conversation_id={conversation_id}",
            headers=stranger_headers,
        )
        assert response.status_code == 404

    def test_message_requires_conversation_owner(self, client):
        owner_headers = _guest_headers(client)
        conversation_id = _create_conversation(client, owner_headers)
        stranger_headers = _guest_headers(client)

        response = client.post(
            "/api/v1/chat/message",
            json={
                "conversation_id": conversation_id,
                "content": "hello",
            },
            headers=stranger_headers,
        )
        assert response.status_code == 404
