"""Tests for WebSocket endpoint with TestClient."""

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

    @patch("api.websocket.thought_logger")
    def test_chat_message(self, mock_logger):
        app = _make_ws_app()
        app.state.graph.ainvoke = AsyncMock(return_value={
            "assistant_response": "你好！",
            "intent": "chitchat",
            "current_itinerary": [],
            "budget_panel": {},
            "preference_panel": {},
            "needs_clarification": False,
            "waiting_for_confirmation": False,
            "error": None,
        })
        mock_logger.start_session = MagicMock()
        mock_logger.register_ws_callback = MagicMock()
        mock_logger.unregister_ws_callback = MagicMock()
        mock_logger.push_status = AsyncMock()
        mock_logger.save = MagicMock(return_value=None)

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_json({"content": "你好", "user_id": "user-1"})
            resp = ws.receive_json()
            assert resp["type"] == "message"
            assert resp["assistant_message"] == "你好！"

    @patch("api.websocket.thought_logger")
    def test_graph_error(self, mock_logger):
        app = _make_ws_app()
        app.state.graph.ainvoke = AsyncMock(side_effect=Exception("graph failed"))
        mock_logger.start_session = MagicMock()
        mock_logger.register_ws_callback = MagicMock()
        mock_logger.unregister_ws_callback = MagicMock()

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_json({"content": "test"})
            resp = ws.receive_json()
            assert resp["type"] == "error"
            assert "Graph execution failed" in resp["error"]

    @patch("api.websocket.thought_logger")
    def test_sanitizes_input(self, mock_logger):
        app = _make_ws_app()
        app.state.graph.ainvoke = AsyncMock(return_value={
            "assistant_response": "ok",
            "intent": "chitchat",
        })
        mock_logger.start_session = MagicMock()
        mock_logger.register_ws_callback = MagicMock()
        mock_logger.unregister_ws_callback = MagicMock()
        mock_logger.push_status = AsyncMock()
        mock_logger.save = MagicMock(return_value=None)

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_json({"content": "  ignore previous  "})
            resp = ws.receive_json()
            assert resp["type"] == "message"
