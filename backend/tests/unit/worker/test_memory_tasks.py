"""Tests for Celery memory background tasks."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from worker import memory_tasks


@pytest.fixture(autouse=True)
def eager_celery():
    """Run Celery tasks synchronously within the test process."""
    old = memory_tasks.celery_app.conf.task_always_eager
    memory_tasks.celery_app.conf.task_always_eager = True
    yield
    memory_tasks.celery_app.conf.task_always_eager = old


@pytest.fixture(autouse=True)
def redis_connected():
    """Pretend Redis is already connected so tasks skip connect()."""
    with patch.object(memory_tasks.redis_client, "_client", object()):
        with patch.object(memory_tasks.redis_client, "connect", new=AsyncMock()):
            yield


def test_archive_session_task_delays_to_memory_manager():
    with patch.object(
        memory_tasks.memory_manager, "archive_to_cold", new=AsyncMock(return_value=True)
    ) as mock_archive:
        result = memory_tasks.archive_session.delay("session-1", "user-1")

    assert result.get() is True
    mock_archive.assert_awaited_once_with("session-1", "user-1", None)


def test_archive_session_task_retries_on_failure():
    with patch.object(
        memory_tasks.memory_manager,
        "archive_to_cold",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ) as mock_archive:
        result = memory_tasks.archive_session.delay("session-1", "user-1")
        # Eager mode stores the final exception in the result.
        with pytest.raises(Exception):
            result.get()

    # initial attempt + 3 retries
    assert mock_archive.await_count == 4


def test_schedule_delayed_archive_enqueues_with_countdown():
    with patch.object(memory_tasks.archive_session, "apply_async") as mock_apply:
        session_id = memory_tasks.schedule_delayed_archive.delay("session-1", "user-1")
    assert session_id.get() == "session-1"
    mock_apply.assert_called_once_with(
        args=["session-1", "user-1"],
        countdown=memory_tasks.DISCONNECT_ARCHIVE_DELAY_SECONDS,
    )


def test_archive_active_sessions_archives_authenticated_hot_sessions():
    snapshot = {"user_id": "user-1", "phase": "gathering"}

    with patch.object(memory_tasks.memory_manager, "hot_get", new=AsyncMock(return_value=snapshot)):
        with patch.object(
            memory_tasks.memory_manager, "archive_to_cold", new=AsyncMock(return_value=True)
        ) as mock_archive:
            with patch.object(
                memory_tasks.redis_client,
                "scan",
                new=AsyncMock(return_value=(0, ["session:s1:state"])),
            ):
                result = memory_tasks.archive_active_sessions.delay()

    summary = result.get()
    assert summary["archived"] == 1
    mock_archive.assert_awaited_once_with("s1", "user-1", None)


def test_compact_stale_sessions_archives_old_warm_snapshots():
    stale_snapshot = {"updated_at": 0, "user_id": "user-1"}

    with patch.object(
        memory_tasks.memory_manager, "warm_get", new=AsyncMock(return_value=stale_snapshot)
    ):
        with patch.object(
            memory_tasks.memory_manager,
            "archive_to_cold",
            new=AsyncMock(return_value=True),
        ) as mock_archive:
            with patch.object(
                memory_tasks.memory_manager, "warm_delete", new=AsyncMock()
            ) as mock_delete:
                with patch.object(
                    memory_tasks.redis_client,
                    "scan",
                    new=AsyncMock(return_value=(0, ["session:s1:warm"])),
                ):
                    result = memory_tasks.compact_stale_sessions.delay()

    summary = result.get()
    assert summary["archived"] == 1
    assert summary["skipped"] == 0
    mock_archive.assert_awaited_once_with("s1", "user-1", None)
    mock_delete.assert_awaited_once_with("s1")


def test_compact_stale_sessions_skips_recent_snapshots():
    recent_snapshot = {"updated_at": int(time.time()), "user_id": "user-1"}

    with patch.object(
        memory_tasks.memory_manager, "warm_get", new=AsyncMock(return_value=recent_snapshot)
    ):
        with patch.object(
            memory_tasks.memory_manager, "archive_to_cold", new=AsyncMock()
        ) as mock_archive:
            with patch.object(
                memory_tasks.redis_client,
                "scan",
                new=AsyncMock(return_value=(0, ["session:s1:warm"])),
            ):
                result = memory_tasks.compact_stale_sessions.delay()

    summary = result.get()
    assert summary["archived"] == 0
    assert summary["skipped"] == 1
    mock_archive.assert_not_awaited()


# ----- worker_process_init / shutdown -----


def test_init_worker_process_starts_background_loop_thread():
    memory_tasks.shutdown_worker_process()

    memory_tasks.init_worker_process()

    assert memory_tasks._worker_loop_thread is not None
    assert memory_tasks._worker_loop_thread.is_alive()
    assert memory_tasks._loop_ready.is_set()
    assert memory_tasks._worker_loop is not None
    assert memory_tasks._worker_loop.is_running()

    memory_tasks.shutdown_worker_process()


def test_init_worker_process_logs_warning_when_loop_start_times_out():
    memory_tasks.shutdown_worker_process()

    with patch.object(memory_tasks._loop_ready, "wait", return_value=False):
        with patch("logging.Logger.warning") as mock_warning:
            memory_tasks.init_worker_process()

    mock_warning.assert_called_once()
    memory_tasks.shutdown_worker_process()


def test_shutdown_worker_process_stops_background_loop():
    memory_tasks.shutdown_worker_process()
    memory_tasks.init_worker_process()

    memory_tasks.shutdown_worker_process()

    assert memory_tasks._worker_loop is None
    assert memory_tasks._worker_loop_thread is None
    assert not memory_tasks._loop_ready.is_set()


# ----- _ensure_redis -----


@pytest.mark.asyncio
async def test_ensure_redis_connects_when_disconnected():
    with patch.object(
        memory_tasks.redis_client, "_ensure_client", side_effect=RuntimeError("not connected")
    ):
        with patch.object(memory_tasks.redis_client, "connect", new=AsyncMock()) as mock_connect:
            await memory_tasks._ensure_redis()

    mock_connect.assert_awaited_once()


# ----- _run_async -----


def test_run_async_uses_worker_loop_when_available():
    memory_tasks.shutdown_worker_process()
    memory_tasks.init_worker_process()

    async def _coro() -> str:
        return "ok"

    assert memory_tasks._run_async(_coro()) == "ok"
    memory_tasks.shutdown_worker_process()


# ----- compact_stale_sessions edge cases -----


def test_compact_stale_sessions_skips_missing_snapshot():
    with patch.object(memory_tasks.memory_manager, "warm_get", new=AsyncMock(return_value=None)):
        with patch.object(
            memory_tasks.memory_manager, "archive_to_cold", new=AsyncMock()
        ) as mock_archive:
            with patch.object(
                memory_tasks.redis_client,
                "scan",
                new=AsyncMock(return_value=(0, ["session:s1:warm"])),
            ):
                result = memory_tasks.compact_stale_sessions.delay()

    summary = result.get()
    assert summary["archived"] == 0
    assert summary["skipped"] == 0
    mock_archive.assert_not_awaited()


def test_compact_stale_sessions_counts_failed_archive_as_skipped():
    stale_snapshot = {"updated_at": 0, "user_id": "user-1"}

    with patch.object(
        memory_tasks.memory_manager, "warm_get", new=AsyncMock(return_value=stale_snapshot)
    ):
        with patch.object(
            memory_tasks.memory_manager,
            "archive_to_cold",
            new=AsyncMock(return_value=False),
        ):
            with patch.object(
                memory_tasks.memory_manager,
                "warm_delete",
                new=AsyncMock(),
            ) as mock_delete:
                with patch.object(
                    memory_tasks.redis_client,
                    "scan",
                    new=AsyncMock(return_value=(0, ["session:s1:warm"])),
                ):
                    result = memory_tasks.compact_stale_sessions.delay()

    summary = result.get()
    assert summary["archived"] == 0
    assert summary["skipped"] == 1
    mock_delete.assert_not_awaited()


def test_compact_stale_sessions_retries_on_failure():
    with patch.object(
        memory_tasks.memory_manager,
        "warm_get",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ) as mock_warm_get:
        with patch.object(
            memory_tasks.redis_client,
            "scan",
            new=AsyncMock(return_value=(0, ["session:s1:warm"])),
        ):
            result = memory_tasks.compact_stale_sessions.delay()
            with pytest.raises(Exception):
                result.get()

    # initial + 2 retries configured on the task
    assert mock_warm_get.await_count == 3


def test_compact_stale_sessions_skips_keys_without_session_prefix():
    stale_snapshot = {"updated_at": 0, "user_id": "user-1"}

    with patch.object(
        memory_tasks.memory_manager, "warm_get", new=AsyncMock(return_value=stale_snapshot)
    ):
        with patch.object(
            memory_tasks.memory_manager,
            "archive_to_cold",
            new=AsyncMock(return_value=True),
        ) as mock_archive:
            with patch.object(
                memory_tasks.memory_manager,
                "warm_delete",
                new=AsyncMock(),
            ):
                with patch.object(
                    memory_tasks.redis_client,
                    "scan",
                    new=AsyncMock(return_value=(0, ["other:s1:warm"])),
                ):
                    result = memory_tasks.compact_stale_sessions.delay()

    summary = result.get()
    assert summary["archived"] == 0
    assert summary["skipped"] == 0
    mock_archive.assert_not_awaited()
