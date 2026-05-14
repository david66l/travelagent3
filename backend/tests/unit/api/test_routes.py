"""Tests for REST API routes."""

import pytest
from unittest.mock import AsyncMock, patch


class TestHealthCheck:
    """Test health endpoint."""

    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestCreateSession:
    """Test session creation."""

    def test_create_session(self, client):
        response = client.post("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["message"] == "Session created"


class TestChat:
    """Test chat endpoint."""

    def test_chat(self, client):
        client.app.state.graph.ainvoke = AsyncMock(return_value={
            "assistant_response": "你好！",
            "intent": "chitchat",
            "needs_clarification": False,
            "waiting_for_confirmation": False,
        })

        with patch("api.routes.db_session", return_value=AsyncMock()):
            with patch("skills.memory_store.MemoryStoreSkill.save_conversation", AsyncMock()):
                response = client.post("/api/chat", json={
                    "content": "你好",
                    "session_id": "test-sess",
                    "user_id": "user-1",
                })
                assert response.status_code == 200
                data = response.json()
                assert data["assistant_message"] == "你好！"

    def test_chat_no_session(self, client):
        client.app.state.graph.ainvoke = AsyncMock(return_value={
            "assistant_response": "好的",
            "intent": "generate_itinerary",
        })

        with patch("api.routes.db_session", return_value=AsyncMock()):
            with patch("skills.memory_store.MemoryStoreSkill.save_conversation", AsyncMock()):
                response = client.post("/api/chat", json={"content": "我想去北京"})
                assert response.status_code == 200
                data = response.json()
                assert data["assistant_message"] == "好的"
                assert data["intent"] == "generate_itinerary"

    def test_chat_sanitizes_input(self, client):
        client.app.state.graph.ainvoke = AsyncMock(return_value={"assistant_response": "ok"})

        with patch("api.routes.db_session", return_value=AsyncMock()):
            with patch("skills.memory_store.MemoryStoreSkill.save_conversation", AsyncMock()):
                response = client.post("/api/chat", json={"content": "  你好  ", "session_id": "s1"})
                assert response.status_code == 200
