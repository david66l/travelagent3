"""Tests for WebSocket endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from api.websocket import ConnectionManager, _build_response


class TestConnectionManager:
    """Test connection management."""

    def setup_method(self):
        self.manager = ConnectionManager()

    @pytest.mark.asyncio
    async def test_connect(self):
        ws = AsyncMock()
        await self.manager.connect("sess-1", ws)
        assert "sess-1" in self.manager._connections
        ws.accept.assert_called_once()

    def test_disconnect(self):
        ws = AsyncMock()
        self.manager._connections["sess-1"] = ws
        self.manager.disconnect("sess-1")
        assert "sess-1" not in self.manager._connections

    def test_disconnect_unknown(self):
        self.manager.disconnect("unknown")  # should not raise

    @pytest.mark.asyncio
    async def test_send_json(self):
        ws = AsyncMock()
        self.manager._connections["sess-1"] = ws
        await self.manager.send_json("sess-1", {"msg": "hello"})
        ws.send_json.assert_called_once_with({"msg": "hello"})

    @pytest.mark.asyncio
    async def test_send_json_unknown_session(self):
        await self.manager.send_json("unknown", {"msg": "hello"})  # should not raise


class TestBuildResponse:
    """Test response builder."""

    def test_basic(self):
        state = {"assistant_response": "你好", "intent": "chitchat"}
        result = _build_response(state)
        assert result["assistant_message"] == "你好"
        assert result["intent"] == "chitchat"
        assert result["type"] == "message"

    def test_empty_state(self):
        result = _build_response({})
        assert result["assistant_message"] == ""
        assert result["needs_clarification"] is False

    def test_with_itinerary(self):
        state = {"assistant_response": "", "current_itinerary": [{"day": 1}]}
        result = _build_response(state)
        assert result["itinerary"] == [{"day": 1}]
