"""Tests for the three-tier memory manager."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.memory import MemoryManager, memory_manager


@pytest.fixture
def fresh_manager(mock_redis) -> MemoryManager:
    """Return a MemoryManager backed by the shared Redis mock."""
    return MemoryManager()


@pytest.mark.asyncio
async def test_hot_get_returns_state(fresh_manager, mock_redis):
    mock_redis.get_json = AsyncMock(return_value={"session_id": "s1", "turn": 1})
    state = await fresh_manager.hot_get("s1")
    assert state["session_id"] == "s1"


@pytest.mark.asyncio
async def test_hot_set_writes_hot_and_warm(fresh_manager, mock_redis):
    mock_redis.set_json = AsyncMock()
    state = {"session_id": "s1", "turn": 1}
    await fresh_manager.hot_set("s1", state)

    calls = mock_redis.set_json.await_args_list
    assert len(calls) == 2
    assert calls[0].args[0] == "session:s1:state"
    assert calls[1].args[0] == "session:s1:warm"


@pytest.mark.asyncio
async def test_load_state_prefers_hot_over_warm(fresh_manager, mock_redis):
    hot = {"tier": "hot", "turn": 2}
    warm = {"tier": "warm", "turn": 1}
    mock_redis.get_json = AsyncMock(side_effect=[hot, warm])
    state = await fresh_manager.load_state("s1")
    assert state["tier"] == "hot"


@pytest.mark.asyncio
async def test_load_state_falls_back_to_warm(fresh_manager, mock_redis):
    warm = {"tier": "warm", "turn": 1}
    mock_redis.get_json = AsyncMock(side_effect=[None, warm])
    mock_redis.set_json = AsyncMock()
    state = await fresh_manager.load_state("s1")
    assert state["tier"] == "warm"


@pytest.mark.asyncio
async def test_load_state_returns_default_when_nothing_cached(fresh_manager, mock_redis):
    mock_redis.get_json = AsyncMock(return_value=None)
    state = await fresh_manager.load_state("new-session")
    assert state["phase"] == "gathering"
    assert state["turn"] == 0


@pytest.mark.asyncio
async def test_acquire_lock_releases_on_exit(fresh_manager, mock_redis):
    mock_redis.set_nx = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.delete = AsyncMock()

    async with fresh_manager.acquire_lock("s1") as acquired_token:
        assert acquired_token is not None
        # Make the lock release check see the same token.
        mock_redis.get.return_value = acquired_token

    mock_redis.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_acquire_lock_does_not_delete_if_token_changed(fresh_manager, mock_redis):
    mock_redis.set_nx = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value="someone-elses-token")
    mock_redis.delete = AsyncMock()

    async with fresh_manager.acquire_lock("s1"):
        pass

    mock_redis.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_archive_to_cold_skips_anonymous_user(fresh_manager, mock_redis):
    mock_redis.get_json = AsyncMock(return_value={"turn": 1})
    result = await fresh_manager.archive_to_cold("s1", user_id="anonymous")
    assert result is False


@pytest.mark.asyncio
async def test_archive_to_cold_creates_conversation(fresh_manager, mock_redis):
    user_id = str(uuid.uuid4())
    snapshot = {"turn": 3, "profile": {"trip": {"destination": "北京"}}}
    mock_redis.get_json = AsyncMock(return_value=snapshot)

    fake_conv = MagicMock()
    fake_conv.id = uuid.uuid4()

    fake_result = MagicMock()
    fake_result.scalar_one_or_none = MagicMock(return_value=None)

    fake_db = MagicMock()
    fake_db.execute = AsyncMock(return_value=fake_result)
    fake_db.add = MagicMock()
    fake_db.flush = AsyncMock()
    fake_db.refresh = AsyncMock()
    fake_db.commit = AsyncMock()

    class _FakeSessionCM:
        async def __aenter__(self):
            return fake_db

        async def __aexit__(self, *exc):
            return False

    with patch("core.memory.async_session_maker", _FakeSessionCM):
        result = await fresh_manager.archive_to_cold("s1", user_id=user_id)

    assert result is True
    fake_db.add.assert_called_once()
    fake_db.commit.assert_awaited_once()
    # Warm snapshot updated with the new conversation id
    mock_redis.set_json.assert_awaited()


@pytest.mark.asyncio
async def test_archive_to_cold_updates_existing_conversation(fresh_manager, mock_redis):
    user_id = str(uuid.uuid4())
    conv_id = uuid.uuid4()
    snapshot = {"turn": 3, "conversation_id": str(conv_id)}
    mock_redis.get_json = AsyncMock(return_value=snapshot)

    fake_conv = MagicMock()
    fake_conv.id = conv_id

    fake_result = MagicMock()
    fake_result.scalar_one_or_none = MagicMock(return_value=fake_conv)

    fake_db = MagicMock()
    fake_db.execute = AsyncMock(return_value=fake_result)
    fake_db.commit = AsyncMock()

    class _FakeSessionCM:
        async def __aenter__(self):
            return fake_db

        async def __aexit__(self, *exc):
            return False

    with patch("core.memory.async_session_maker", _FakeSessionCM):
        result = await fresh_manager.archive_to_cold("s1", user_id=user_id)

    assert result is True
    assert fake_conv.state_snapshot == snapshot
    fake_db.commit.assert_awaited_once()


@pytest.mark.asyncio
async def test_parse_uuid_helper():
    valid = memory_manager._parse_uuid(str(uuid.uuid4()))
    assert isinstance(valid, uuid.UUID)
    assert memory_manager._parse_uuid("not-a-uuid") is None
    assert memory_manager._parse_uuid(None) is None


@pytest.mark.asyncio
async def test_hot_delete_calls_redis_delete(fresh_manager, mock_redis):
    await fresh_manager.hot_delete("s1")
    mock_redis.delete.assert_awaited_once_with("session:s1:state")


@pytest.mark.asyncio
async def test_warm_delete_calls_redis_delete(fresh_manager, mock_redis):
    await fresh_manager.warm_delete("s1")
    mock_redis.delete.assert_awaited_once_with("session:s1:warm")


@pytest.mark.asyncio
async def test_archive_to_cold_returns_false_when_no_warm_snapshot(fresh_manager, mock_redis):
    mock_redis.get_json = AsyncMock(return_value=None)
    result = await fresh_manager.archive_to_cold("s1", user_id=str(uuid.uuid4()))
    assert result is False


@pytest.mark.asyncio
async def test_acquire_lock_non_blocking_raises_when_not_acquired(fresh_manager, mock_redis):
    mock_redis.set_nx = AsyncMock(return_value=False)

    with pytest.raises(RuntimeError, match="Could not acquire lock"):
        async with fresh_manager.acquire_lock("s1", blocking=False):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_acquire_lock_timeout_raises_when_not_acquired(fresh_manager, mock_redis):
    mock_redis.set_nx = AsyncMock(return_value=False)

    with pytest.raises(RuntimeError, match="Timeout acquiring lock"):
        async with fresh_manager.acquire_lock("s1", blocking_timeout=0.0):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_acquire_lock_swallows_release_exception(fresh_manager, mock_redis):
    mock_redis.set_nx = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.delete = AsyncMock(side_effect=RuntimeError("redis down"))

    async with fresh_manager.acquire_lock("s1") as acquired_token:
        mock_redis.get.return_value = acquired_token

    # No exception should propagate from the context manager.
    mock_redis.delete.assert_awaited_once()


@pytest.mark.asyncio
async def test_acquire_lock_extension_renews_token(fresh_manager, mock_redis):
    """Background task extends the lock while the caller holds it."""
    mock_redis.set_nx = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock()
    mock_redis.delete = AsyncMock()

    async with fresh_manager.acquire_lock("s1", ttl=0.1) as token:
        mock_redis.get.return_value = token
        # Yield enough time for the extension task to run at least once.
        await asyncio.sleep(0.06)

    # set is called for the initial lock and at least one extension.
    assert mock_redis.set.await_count >= 1


@pytest.mark.asyncio
async def test_acquire_lock_extension_stops_when_token_changes(fresh_manager, mock_redis):
    """Extension loop exits early if another owner holds the lock."""
    mock_redis.set_nx = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(return_value="other-token")
    mock_redis.delete = AsyncMock()

    async with fresh_manager.acquire_lock("s1", ttl=0.1):
        await asyncio.sleep(0.06)

    # No extension attempts should issue a SET.
    mock_redis.set.assert_not_awaited()


@pytest.mark.asyncio
async def test_acquire_lock_waits_then_acquires(fresh_manager, mock_redis):
    """Blocking acquisition retries until the lock becomes available."""
    mock_redis.set_nx = AsyncMock(side_effect=[False, True])
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.delete = AsyncMock()

    sleep_calls = []

    original_sleep = asyncio.sleep

    def _patched_sleep(delay: float) -> asyncio.Future[None]:
        sleep_calls.append(delay)
        return original_sleep(0)

    with patch("core.memory.asyncio.sleep", _patched_sleep):
        async with fresh_manager.acquire_lock("s1") as token:
            mock_redis.get.return_value = token

    assert mock_redis.set_nx.await_count == 2
    assert sleep_calls


@pytest.mark.asyncio
async def test_acquire_lock_extension_logs_debug_on_exception(fresh_manager, mock_redis):
    """Extension loop should not propagate Redis errors."""
    mock_redis.set_nx = AsyncMock(return_value=True)
    mock_redis.get = AsyncMock(side_effect=RuntimeError("redis error"))
    mock_redis.delete = AsyncMock()

    async with fresh_manager.acquire_lock("s1", ttl=0.1):
        await asyncio.sleep(0.06)

    # The lock context exits cleanly even though the extension task crashed.
    assert mock_redis.get.await_count > 0
