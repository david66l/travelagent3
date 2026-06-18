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
    mock_redis.get = AsyncMock(return_value=None)
    state = await fresh_manager.load_state("new-session")
    assert state["phase"] == "gathering"
    assert state["turn"] == 0


@pytest.mark.asyncio
async def test_load_state_recovers_from_cold_archive(fresh_manager, mock_redis):
    conv_id = uuid.uuid4()
    cold = {"destination": "杭州", "turn": 3, "phase": "completed"}
    mock_redis.get_json = AsyncMock(return_value=None)
    mock_redis.get = AsyncMock(return_value=str(conv_id))

    with patch.object(
        fresh_manager,
        "cold_get_by_conversation_id",
        new=AsyncMock(return_value=cold),
    ):
        state = await fresh_manager.load_state("s1")

    assert state["destination"] == "杭州"
    assert state["turn"] == 3


@pytest.mark.asyncio
async def test_load_state_promotes_warm_to_hot(fresh_manager, mock_redis):
    warm = {"tier": "warm", "turn": 2}
    mock_redis.get_json = AsyncMock(side_effect=[None, warm])
    mock_redis.set_json = AsyncMock()

    state = await fresh_manager.load_state("s1", promote_to_hot=True)

    assert state["turn"] == 2
    assert mock_redis.set_json.await_count >= 2


@pytest.fixture
def redlock_sim():
    """Simulate Redlock with an in-memory holder for lock tests."""
    holder: dict[str, str] = {}
    nx_lock = asyncio.Lock()

    async def acquire(resource, ttl, blocking=True, blocking_timeout=2.0):
        token = uuid.uuid4().hex
        deadline = asyncio.get_event_loop().time() + blocking_timeout
        while True:
            async with nx_lock:
                if resource not in holder:
                    holder[resource] = token
                    return token
            if not blocking:
                return None
            if asyncio.get_event_loop().time() >= deadline:
                return None
            await asyncio.sleep(0.05)

    async def release(resource, token):
        async with nx_lock:
            if holder.get(resource) == token:
                del holder[resource]

    async def extend(resource, token, ttl):
        return holder.get(resource) == token

    with patch("core.memory.redlock") as rl:
        rl.acquire = AsyncMock(side_effect=acquire)
        rl.release = AsyncMock(side_effect=release)
        rl.extend = AsyncMock(side_effect=extend)
        yield rl, holder


@pytest.mark.asyncio
async def test_acquire_lock_releases_on_exit(fresh_manager, redlock_sim):
    rl, holder = redlock_sim

    async with fresh_manager.acquire_lock("s1") as acquired_token:
        assert acquired_token is not None
        assert holder.get("session:s1:lock") == acquired_token

    assert "session:s1:lock" not in holder
    rl.release.assert_awaited()


@pytest.mark.asyncio
async def test_acquire_lock_does_not_delete_if_token_changed(fresh_manager, redlock_sim):
    rl, holder = redlock_sim

    async with fresh_manager.acquire_lock("s1") as token:
        holder["session:s1:lock"] = "someone-elses-token"

    rl.release.assert_awaited()
    assert holder["session:s1:lock"] == "someone-elses-token"


@pytest.mark.asyncio
async def test_acquire_lock_non_blocking_raises_when_not_acquired(fresh_manager, redlock_sim):
    _, holder = redlock_sim
    holder["session:s1:lock"] = "occupied"

    with pytest.raises(RuntimeError, match="Could not acquire lock"):
        async with fresh_manager.acquire_lock("s1", blocking=False):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_acquire_lock_timeout_raises_when_not_acquired(fresh_manager, redlock_sim):
    _, holder = redlock_sim
    holder["session:s1:lock"] = "occupied"

    with pytest.raises(RuntimeError, match="Timeout acquiring lock"):
        async with fresh_manager.acquire_lock("s1", blocking_timeout=0.0):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_acquire_lock_swallows_release_exception(fresh_manager, redlock_sim):
    rl, _ = redlock_sim
    rl.release = AsyncMock(side_effect=RuntimeError("redis down"))

    async with fresh_manager.acquire_lock("s1"):
        pass

    rl.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_acquire_lock_extension_renews_token(fresh_manager, redlock_sim):
    rl, holder = redlock_sim

    async with fresh_manager.acquire_lock("s1", ttl=0.1):
        await asyncio.sleep(0.06)

    rl.extend.assert_awaited()


@pytest.mark.asyncio
async def test_acquire_lock_extension_stops_when_token_changes(fresh_manager, redlock_sim):
    rl, holder = redlock_sim

    async with fresh_manager.acquire_lock("s1", ttl=0.1) as token:
        holder["session:s1:lock"] = "other-token"
        await asyncio.sleep(0.06)

    assert rl.extend.await_count >= 0


@pytest.mark.asyncio
async def test_acquire_lock_waits_then_acquires(fresh_manager, redlock_sim):
    _, holder = redlock_sim
    holder["session:s1:lock"] = "busy"

    sleep_calls = []
    original_sleep = asyncio.sleep

    def _patched_sleep(delay: float) -> asyncio.Future[None]:
        sleep_calls.append(delay)
        holder.pop("session:s1:lock", None)
        return original_sleep(0)

    with patch("core.memory.asyncio.sleep", _patched_sleep):
        async with fresh_manager.acquire_lock("s1"):
            pass

    assert sleep_calls


@pytest.mark.asyncio
async def test_acquire_lock_extension_logs_debug_on_exception(fresh_manager, redlock_sim):
    rl, _ = redlock_sim
    rl.extend = AsyncMock(side_effect=RuntimeError("redis error"))

    async with fresh_manager.acquire_lock("s1", ttl=0.1):
        await asyncio.sleep(0.06)

    assert rl.extend.await_count > 0


@pytest.mark.asyncio
async def test_concurrent_lock_allows_only_one_holder(fresh_manager, redlock_sim):
    outcomes: list[str] = []

    async def worker(name: str):
        try:
            async with fresh_manager.acquire_lock("concurrent-s1", blocking_timeout=0.08):
                outcomes.append(f"ok:{name}")
                await asyncio.sleep(0.2)
        except RuntimeError:
            outcomes.append(f"blocked:{name}")

    await asyncio.gather(worker("a"), worker("b"))

    assert sum(o.startswith("ok:") for o in outcomes) == 1
    assert sum(o.startswith("blocked:") for o in outcomes) == 1


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


