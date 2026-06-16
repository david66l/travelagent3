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

    @patch("api.chat_runtime.process_user_turn", new_callable=AsyncMock)
    @patch("api.chat_runtime.PlanningJobRepository")
    @patch("api.chat_runtime.redis_client")
    @patch("api.chat_runtime.enqueue_planning_job")
    def test_chat_message_creates_job(
        self, mock_enqueue, mock_redis, mock_repo_cls, mock_process_turn
    ):
        app = _make_ws_app()
        async def _process(state, content):
            from core.conversation_state import merge_profile
            from schemas import ProfilePatch

            state["profile"] = merge_profile(
                state.get("profile", {}),
                ProfilePatch(set={"destination": "北京", "travel_days": 3}),
            )
            result = MagicMock()
            result.missing_required = []
            result.clarification_questions = []
            return result

        mock_process_turn.side_effect = _process
        mock_repo = MagicMock()
        mock_job = MagicMock()
        mock_job.id = "job-123"
        mock_job.status = "pending"
        mock_repo.create = AsyncMock(return_value=mock_job)
        mock_repo.request_cancel = AsyncMock()
        mock_repo.get_by_session = AsyncMock(return_value=[])  # P1: load_state
        mock_repo_cls.return_value = mock_repo

        mock_redis._client = MagicMock()
        mock_redis._client.publish = AsyncMock()
        mock_redis.get_json = AsyncMock(return_value=None)  # Redis cache miss → PG fallback
        mock_redis.set_json = AsyncMock()  # write-back after load + after create_job

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_json({"content": "北京3天", "user_id": "user-1"})
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
        mock_repo.get = AsyncMock(return_value=None)  # P1: subscribe→get job
        mock_repo.get_by_session = AsyncMock(return_value=[])  # P1: load_state
        mock_repo_cls.return_value = mock_repo

        mock_redis._client = MagicMock()
        mock_redis._client.pubsub = MagicMock()
        mock_redis.set_json = AsyncMock()  # subscribe handler writes state back to Redis

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_json({"type": "subscribe", "job_id": "job-123", "last_event_id": 0})

    @patch("api.chat_runtime.process_user_turn", new_callable=AsyncMock)
    @patch("api.chat_runtime.PlanningJobRepository")
    @patch("api.chat_runtime.redis_client")
    def test_chat_message_requests_clarification_when_profile_incomplete(
        self, mock_redis, mock_repo_cls, mock_process_turn
    ):
        app = _make_ws_app()

        intent_result = MagicMock()
        intent_result.intent = "plan"
        intent_result.user_entities = {}
        intent_result.missing_required = ["destination"]
        intent_result.clarification_questions = ["请问您想去哪个目的地？"]

        async def _process(state, content):
            state["profile"] = state.get("profile", {})
            return intent_result

        mock_process_turn.side_effect = _process

        mock_repo = MagicMock()
        mock_repo.get_by_session = AsyncMock(return_value=[])
        mock_repo_cls.return_value = mock_repo

        mock_redis.get_json = AsyncMock(return_value=None)
        mock_redis.set_json = AsyncMock()

        client = TestClient(app)
        with client.websocket_connect("/ws/chat/sess-1") as ws:
            ws.send_json({"content": "我想去旅行", "user_id": "user-1"})
            resp = ws.receive_json()
            assert resp["type"] == "needs_clarification"
            assert "destination" in resp["missing_required"]
