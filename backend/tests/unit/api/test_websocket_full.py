"""Tests for WebSocket endpoint with TestClient (Phase 1 job model)."""

from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from api.websocket import router as ws_router


def _make_ws_app():
    app = FastAPI()
    app.include_router(ws_router)
    return app


class TestWebSocketConnect:
    """Test WebSocket connection lifecycle."""

    def test_connect_disconnect(self):
        app = _make_ws_app()
        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as _ws:
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

    def test_chat_message_starts_graph(self):
        with patch("api.websocket.process_chat_message", new_callable=AsyncMock) as mock_process:
            app = _make_ws_app()
            client = TestClient(app)
            with client.websocket_connect("/ws/chat/sess-1") as ws:
                ws.send_json({"content": "北京3天", "user_id": "user-1"})
            mock_process.assert_awaited_once()

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
        mock_repo.get = AsyncMock(return_value=None)  # P1: subscribe→get job
        mock_repo.get_by_session = AsyncMock(return_value=[])  # P1: load_state
        mock_repo_cls.return_value = mock_repo

        mock_redis._client = MagicMock()
        mock_redis._client.pubsub = MagicMock()
        mock_redis.set_json = AsyncMock()  # subscribe handler writes state back to Redis

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_json({"type": "subscribe", "job_id": "job-123", "last_event_id": 0})

    def test_chat_message_starts_graph_for_incomplete_profile(self):
        with patch("api.websocket.process_chat_message", new_callable=AsyncMock) as mock_process:
            app = _make_ws_app()
            client = TestClient(app)
            with client.websocket_connect("/ws/chat/sess-1") as ws:
                ws.send_json({"content": "我想去旅行", "user_id": "user-1"})
            mock_process.assert_awaited_once()
