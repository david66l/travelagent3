"""Tests for SSE chat endpoints."""

import asyncio
import json
from uuid import uuid4

from api.chat_runtime import manager


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

    def test_post_message_idempotency_replays_job_without_duplicate_message(self, client):
        headers = _guest_headers(client)
        headers["Idempotency-Key"] = f"chat-{uuid4()}"
        conversation_id = _create_conversation(client, headers)
        payload = {
            "conversation_id": conversation_id,
            "content": "想去杭州玩三天",
            "stream": True,
        }

        first = client.post("/api/v1/chat/message", json=payload, headers=headers)
        second = client.post("/api/v1/chat/message", json=payload, headers=headers)

        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["data"]["job_id"] == second.json()["data"]["job_id"]
        assert first.json()["data"]["idempotent_replay"] is False
        assert second.json()["data"]["idempotent_replay"] is True

        messages = client.get(
            f"/api/v1/conversations/{conversation_id}/messages",
            headers=headers,
        ).json()["data"]
        matching = [
            item
            for item in messages
            if item["role"] == "user" and item["content"] == payload["content"]
        ]
        assert len(matching) == 1

    def test_post_message_rejects_idempotency_key_reuse_for_different_payload(self, client):
        headers = _guest_headers(client)
        headers["Idempotency-Key"] = f"chat-{uuid4()}"
        conversation_id = _create_conversation(client, headers)

        first = client.post(
            "/api/v1/chat/message",
            json={"conversation_id": conversation_id, "content": "杭州三天"},
            headers=headers,
        )
        conflict = client.post(
            "/api/v1/chat/message",
            json={"conversation_id": conversation_id, "content": "北京五天"},
            headers=headers,
        )

        assert first.status_code == 202
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "IDEMPOTENCY_KEY_CONFLICT"

    def test_stream_opens(self, client):
        headers = _guest_headers(client)
        conversation_id = _create_conversation(client, headers)
        session_id = conversation_id

        # TestClient waits for a streaming response to finish. Seed a real
        # terminal event through the connection manager so this unit test
        # verifies the production SSE path without waiting for its 30-second
        # keepalive or the endpoint's user-facing multi-minute timeout.
        asyncio.run(manager.send_json(session_id, {"type": "done"}))

        with client.stream(
            "GET",
            f"/api/v1/chat/stream?conversation_id={conversation_id}&timeout=120",
            headers=headers,
        ) as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("content-type", "")
            payload = b"".join(response.iter_bytes()).decode("utf-8")
            data_line = next(line for line in payload.splitlines() if line.startswith("data: "))
            assert json.loads(data_line.removeprefix("data: ")) == {"type": "done"}

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
