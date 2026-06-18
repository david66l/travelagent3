"""Tests for WebSocket endpoint."""

from contextlib import asynccontextmanager
from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from api.chat_runtime import ConnectionManager, delayed_cancel
from core.conversation_state import default_conversation_state
from core.redis_client import redis_client
from core.memory import memory_manager


class TestConnectionManager:
    """Test connection management."""

    def setup_method(self):
        self.manager = ConnectionManager()

    @pytest.mark.asyncio
    async def test_connect(self):
        ws = AsyncMock()
        await self.manager.connect_ws("sess-1", ws)
        assert "sess-1" in self.manager._connections
        ws.accept.assert_called_once()

    def test_disconnect(self):
        ws = AsyncMock()
        self.manager._connections["sess-1"] = ws
        self.manager.disconnect_ws("sess-1")
        assert "sess-1" not in self.manager._connections

    def test_disconnect_unknown(self):
        self.manager.disconnect_ws("unknown")  # should not raise

    @pytest.mark.asyncio
    async def test_send_json(self):
        ws = AsyncMock()
        self.manager._connections["sess-1"] = ws
        await self.manager.send_json("sess-1", {"msg": "hello"})
        ws.send_json.assert_called_once_with({"msg": "hello"})

    @pytest.mark.asyncio
    async def test_send_json_unknown_session(self):
        await self.manager.send_json("unknown", {"msg": "hello"})  # should not raise


class TestConnectionManagerState:
    """Cover load_state / save_state / helpers."""

    @pytest.mark.asyncio
    async def test_send_json_swallows_send_exception(self):
        ws = AsyncMock()
        ws.send_json.side_effect = RuntimeError("connection closed")
        manager = ConnectionManager()
        manager._connections["s1"] = ws
        # Should not raise.
        await manager.send_json("s1", {"msg": "hi"})

    @pytest.mark.asyncio
    async def test_load_state_prefers_hot_layer(self):
        manager = ConnectionManager()
        hot = {"phase": "gathering", "turn": 2}
        with patch.object(
            memory_manager, "load_state", new=AsyncMock(return_value=hot)
        ):
            with patch.object(memory_manager, "hot_set", new=AsyncMock()):
                state = await manager.load_state("s1")
        assert state["turn"] == 2

    @pytest.mark.asyncio
    async def test_load_state_falls_back_to_warm_and_writes_hot(self):
        manager = ConnectionManager()
        warm = {"phase": "gathering", "turn": 1}
        with patch.object(
            memory_manager, "load_state", new=AsyncMock(return_value=warm)
        ):
            with patch.object(memory_manager, "hot_set", new=AsyncMock()) as mock_hot_set:
                state = await manager.load_state("s1")
        assert state["turn"] == 1
        mock_hot_set.assert_awaited_once_with("s1", warm)

    @pytest.mark.asyncio
    async def test_load_state_falls_back_to_cold_archive(self):
        manager = ConnectionManager()
        cold = {"phase": "completed", "turn": 4, "destination": "上海"}
        with patch.object(
            memory_manager, "load_state", new=AsyncMock(return_value=cold)
        ):
            with patch.object(memory_manager, "hot_set", new=AsyncMock()) as mock_hot_set:
                state = await manager.load_state("s1")
        assert state["destination"] == "上海"
        mock_hot_set.assert_awaited_once_with("s1", cold)

    @pytest.mark.asyncio
    async def test_load_state_falls_back_to_planning_job_feedback(self):
        manager = ConnectionManager()
        feedback = {"phase": "completed", "turn": 5}
        job = MagicMock()
        job.user_feedback = feedback

        fake_repo = MagicMock()
        fake_repo.get_by_session = AsyncMock(return_value=[job])

        with patch("api.chat_runtime.PlanningJobRepository", return_value=fake_repo):
            with patch("api.chat_runtime.async_session_maker", _fake_session()):
                with patch.object(
                    memory_manager,
                    "load_state",
                    new=AsyncMock(return_value=default_conversation_state()),
                ):
                    with patch.object(
                        memory_manager, "hot_set", new=AsyncMock()
                    ) as mock_hot_set:
                        state = await manager.load_state("s1")

        assert state["turn"] == 5
        mock_hot_set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_state_returns_default_when_no_source(self):
        manager = ConnectionManager()
        fake_repo = MagicMock()
        fake_repo.get_by_session = AsyncMock(return_value=[])

        with patch("api.chat_runtime.PlanningJobRepository", return_value=fake_repo):
            with patch("api.chat_runtime.async_session_maker", _fake_session()):
                with patch.object(
                    memory_manager,
                    "load_state",
                    new=AsyncMock(return_value=default_conversation_state()),
                ):
                    with patch.object(
                        memory_manager, "hot_set", new=AsyncMock()
                    ) as mock_hot_set:
                        state = await manager.load_state("s1")

        assert state["phase"] == "gathering"
        mock_hot_set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_save_state_enqueues_archive_when_requested(self):
        manager = ConnectionManager()
        state = {"phase": "completed", "user_id": "user-1", "updated_at": 0}

        fake_repo = MagicMock()
        fake_repo.update_user_feedback = AsyncMock()

        with patch("api.chat_runtime.PlanningJobRepository", return_value=fake_repo):
            with patch("api.chat_runtime.async_session_maker", _fake_session()):
                with patch("api.chat_runtime.archive_session") as mock_archive:
                    with patch.object(memory_manager, "hot_set", new=AsyncMock()):
                        with patch.object(memory_manager, "acquire_lock", _null_lock):
                            await manager.save_state("job-1", "s1", state, trigger_archive=True)

        fake_repo.update_user_feedback.assert_awaited_once()
        mock_archive.delay.assert_called_once_with("s1", "user-1", None)


class TestHelpers:
    """Cover small helper functions."""

    @pytest.mark.asyncio
    async def test_delayed_cancel_requests_cancel_for_active_job(self):
        job = MagicMock()
        job.status = "pending"

        fake_repo = MagicMock()
        fake_repo.get = AsyncMock(return_value=job)
        fake_repo.request_cancel = AsyncMock()

        with patch("api.chat_runtime.PlanningJobRepository", return_value=fake_repo):
            with patch("api.chat_runtime.asyncio.sleep", new=AsyncMock()):
                with patch("api.chat_runtime.async_session_maker", _fake_session()):
                    with patch.object(redis_client, "_client", MagicMock(publish=AsyncMock())):
                        await delayed_cancel("job-1", delay=30)

        fake_repo.request_cancel.assert_awaited_once_with("job-1")

    @pytest.mark.asyncio
    async def test_delayed_cancel_skips_finished_job(self):
        job = MagicMock()
        job.status = "completed"

        fake_repo = MagicMock()
        fake_repo.get = AsyncMock(return_value=job)
        fake_repo.request_cancel = AsyncMock()

        with patch("api.chat_runtime.PlanningJobRepository", return_value=fake_repo):
            with patch("api.chat_runtime.asyncio.sleep", new=AsyncMock()):
                with patch("api.chat_runtime.async_session_maker", _fake_session()):
                    with patch.object(redis_client, "_client", MagicMock(publish=AsyncMock())):
                        await delayed_cancel("job-1", delay=30)

        fake_repo.request_cancel.assert_not_awaited()

    def test_entities_to_patch_converts_scalars_and_lists(self):
        from core.conversation_turn import entities_to_patch

        entities = {
            "destination": "北京",
            "travel_days": 3,
            "interests": ["历史", "文化"],
            "food_preferences": ["烤鸭"],
        }
        patch = entities_to_patch(entities)
        assert patch.set["destination"] == "北京"
        assert patch.set["travel_days"] == 3
        assert patch.add["interests"] == ["历史", "文化"]
        assert patch.add["food_preferences"] == ["烤鸭"]


# --------------------------------------------------------------------------- #
# Test utilities
# --------------------------------------------------------------------------- #


def _fake_session():
    """Return an async context manager that yields a no-op DB session."""

    class _FakeSession:
        async def __aenter__(self):
            session = MagicMock()
            session.commit = AsyncMock()
            session.add = MagicMock()
            session.refresh = AsyncMock()
            return session

        async def __aexit__(self, *exc):
            return False

    return _FakeSession


@asynccontextmanager
async def _null_lock(*args, **kwargs):
    """No-op lock for save_state tests."""
    yield "token"


# --------------------------------------------------------------------------- #
# push_job_status coverage
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_push_job_status_returns_when_job_already_terminal():
    """If the job is already terminal after the initial drain, exit early."""
    from api.chat_runtime import push_job_status

    job = MagicMock()
    job.status = "completed"

    fake_repo = MagicMock()
    fake_repo.get_events_after = AsyncMock(return_value=[])
    fake_repo.get = AsyncMock(return_value=job)

    with patch("api.chat_runtime.PlanningJobRepository", return_value=fake_repo):
        with patch("api.chat_runtime.async_session_maker", _fake_session()):
            await push_job_status("job-1", "s1", from_event_id=0)

    fake_repo.get.assert_awaited_once()


@pytest.mark.asyncio
async def test_load_state_ignores_memory_load_exception():
    manager = ConnectionManager()
    fake_repo = MagicMock()
    fake_repo.get_by_session = AsyncMock(return_value=[])

    with patch("api.chat_runtime.PlanningJobRepository", return_value=fake_repo):
        with patch("api.chat_runtime.async_session_maker", _fake_session()):
            with patch.object(
                memory_manager,
                "load_state",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ):
                with patch.object(memory_manager, "hot_set", new=AsyncMock()):
                    state = await manager.load_state("s1")

    assert state["phase"] == "gathering"


@pytest.mark.asyncio
async def test_load_state_ignores_hot_set_exception_on_write_back():
    """Write-back hot_set failure must not crash load_state."""
    manager = ConnectionManager()
    warm = {"phase": "gathering", "turn": 1}
    with patch.object(
        memory_manager, "load_state", new=AsyncMock(return_value=warm)
    ):
        with patch.object(
            memory_manager, "hot_set", new=AsyncMock(side_effect=RuntimeError("boom"))
        ):
            state = await manager.load_state("s1")

    assert state["turn"] == 1


@pytest.mark.asyncio
async def test_save_gathering_state_reports_session_conflict():
    manager = ConnectionManager()
    state = {"phase": "gathering", "turn": 1}

    @asynccontextmanager
    async def _failing_lock(*args, **kwargs):
        raise RuntimeError("Timeout acquiring lock")
        yield  # pragma: no cover

    sent: list[dict[str, Any]] = []
    with patch.object(memory_manager, "acquire_lock", _failing_lock):
        with patch.object(manager, "send_json", new=AsyncMock(side_effect=lambda _s, d: sent.append(d))):
            ok = await manager.save_gathering_state("s1", state)

    assert ok is False
    assert any(msg.get("code") == "session_conflict" for msg in sent)


@pytest.mark.asyncio
async def test_push_job_status_sends_needs_clarification_for_completed_event():
    """Completed event with needs_human triggers a clarification message."""
    from api.chat_runtime import push_job_status

    job = MagicMock()
    job.status = "completed"

    event = MagicMock()
    event.id = 42
    event.stage = "completed"
    event.event_type = "done"
    event.payload = {
        "needs_human": True,
        "violations": [{"msg": "closed"}],
        "suggested_fixes": [{"msg": "open later"}],
    }

    fake_repo = MagicMock()
    fake_repo.get_events_after = AsyncMock(side_effect=[[event], []])
    fake_repo.get = AsyncMock(return_value=job)

    sent_messages: list[dict[str, Any]] = []

    async def _capture(_sid: str, data: dict[str, Any]) -> None:
        sent_messages.append(data)

    with patch("api.chat_runtime.manager.send_json", new=AsyncMock(side_effect=_capture)):
        with patch("api.chat_runtime.PlanningJobRepository", return_value=fake_repo):
            with patch("api.chat_runtime.async_session_maker", _fake_session()):
                await push_job_status("job-1", "s1", from_event_id=0)

    assert any(msg.get("type") == "needs_clarification" for msg in sent_messages)


@pytest.mark.asyncio
async def test_save_state_swallows_archive_enqueue_exception():
    """Failure to enqueue the archive task should not break save_state."""
    manager = ConnectionManager()
    state = {"phase": "completed", "user_id": "user-1", "updated_at": 0}

    fake_repo = MagicMock()
    fake_repo.update_user_feedback = AsyncMock()

    with patch("api.chat_runtime.PlanningJobRepository", return_value=fake_repo):
        with patch("api.chat_runtime.async_session_maker", _fake_session()):
            with patch("api.chat_runtime.archive_session") as mock_archive:
                mock_archive.delay.side_effect = RuntimeError("celery down")
                with patch("api.chat_runtime.memory_manager") as mock_mem:
                    mock_mem.hot_set = AsyncMock()
                    mock_mem.acquire_lock = _null_lock
                    await manager.save_state("job-1", "s1", state, trigger_archive=True)

    fake_repo.update_user_feedback.assert_awaited_once()
