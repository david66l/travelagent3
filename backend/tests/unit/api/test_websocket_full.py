"""Tests for WebSocket endpoint with TestClient (Phase 1 job model)."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from api.websocket import router as ws_router


def _make_ws_app():
    app = FastAPI()
    app.include_router(ws_router)
    app.state.graph = AsyncMock()
    app.state.checkpointer = AsyncMock()
    return app


class TestWebSocketConnect:
    """Test WebSocket connection lifecycle."""

    def test_connect_disconnect(self):
        app = _make_ws_app()
        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            pass  # connect then disconnect

    def test_invalid_json(self):
        app = _make_ws_app()
        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_text("not json")
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Invalid JSON" in resp["error"]

    def test_empty_message(self):
        app = _make_ws_app()
        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_json({"content": "   "})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Empty" in resp["error"]

    @patch("api.websocket.PlanningJobRepository")
    @patch("api.websocket.redis_client")
    def test_chat_message_creates_job(self, mock_redis, mock_repo_cls):
        app = _make_ws_app()
        mock_repo = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_job.status = "pending"
        mock_repo.create = AsyncMock(return_value=mock_job)
        mock_repo.request_cancel = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        mock_redis._client = MagicMock()
        mock_redis._client.publish = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_json({"content": "你好", "user_id": "user-1"})
            resp = ws.receive_json()
            assert resp["type"] == "job_created"
            assert resp["job_id"] == "job-123"
            assert resp["status"] == "pending"

    @patch("api.websocket.PlanningJobRepository")
    @patch("api.websocket.redis_client")
    def test_cancel_job(self, mock_redis, mock_repo_cls):
        app = _make_ws_app()
        mock_repo = MagicMock()
        mock_repo.request_cancel = AsyncMock()
        mock_repo_cls.return_value = mock_repo

        mock_redis._client = MagicMock()
        mock_redis._client.publish = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_json({"type": "cancel", "job_id": "job-123"})
            # Cancel is async; no immediate response expected

    @patch("api.websocket.PlanningJobRepository")
    @patch("api.websocket.redis_client")
    def test_subscribe_reconnect(self, mock_redis, mock_repo_cls):
        app = _make_ws_app()
        mock_repo = MagicMock()
        mock_repo.get_events_after = AsyncMock(return_value=[])
        mock_repo_cls.return_value = mock_repo

        mock_redis._client = MagicMock()
        mock_redis._client.pubsub = MagicMock()

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_json({"type": "subscribe", "job_id": "job-123", "last_event_id": 0})
