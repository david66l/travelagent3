"""Tests for Celery planning job execution."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from worker import memory_tasks, planning_tasks


@pytest.fixture(autouse=True)
def eager_celery():
    old = planning_tasks.celery_app.conf.task_always_eager
    planning_tasks.celery_app.conf.task_always_eager = True
    yield
    planning_tasks.celery_app.conf.task_always_eager = old


@pytest.fixture(autouse=True)
def redis_connected():
    with patch.object(memory_tasks.redis_client, "_client", object()):
        with patch.object(memory_tasks.redis_client, "connect", new=AsyncMock()):
            yield


def test_execute_planning_job_runs_worker():
    with patch(
        "worker.planning_worker.PlanningWorker.execute_job_by_id",
        new=AsyncMock(return_value=True),
    ) as mock_exec:
        with patch("worker.planning_tasks.reset_retry", new=AsyncMock()):
            result = planning_tasks.execute_planning_job.delay("job-123")

    assert result.get() is True
    mock_exec.assert_awaited_once_with("job-123")


def test_execute_planning_job_not_claimed_returns_false():
    with patch(
        "worker.planning_worker.PlanningWorker.execute_job_by_id",
        new=AsyncMock(return_value=False),
    ):
        result = planning_tasks.execute_planning_job.delay("job-missing")

    assert result.get() is False


def test_execute_planning_job_non_retryable_goes_to_dlq():
    with patch(
        "worker.planning_worker.PlanningWorker.execute_job_by_id",
        new=AsyncMock(side_effect=ValueError("bad input")),
    ):
        with patch(
            "worker.planning_tasks._finalize_dead_letter",
            new=AsyncMock(),
        ) as mock_dlq:
            result = planning_tasks.execute_planning_job.delay("job-bad")

    assert result.get() is False
    mock_dlq.assert_awaited_once()


def test_execute_planning_job_retryable_marks_retrying():
    with patch(
        "worker.planning_worker.PlanningWorker.execute_job_by_id",
        new=AsyncMock(side_effect=TimeoutError("slow")),
    ):
        with patch(
            "worker.planning_tasks.increment_retry",
            new=AsyncMock(return_value=1),
        ):
            with patch(
                "worker.planning_tasks._mark_retrying",
                new=AsyncMock(),
            ) as mock_retrying:
                with patch.object(
                    planning_tasks.execute_planning_job,
                    "retry",
                    side_effect=RuntimeError("retry scheduled"),
                ):
                    try:
                        planning_tasks.execute_planning_job.delay("job-retry")
                    except RuntimeError as exc:
                        assert "retry scheduled" in str(exc)

    mock_retrying.assert_awaited_once()


@pytest.mark.asyncio
async def test_finalize_dead_letter_pushes_record_and_marks_job_failed():
    repo = AsyncMock()
    db = AsyncMock()
    db.execute = AsyncMock()
    db.commit = AsyncMock()

    class SessionContext:
        async def __aenter__(self):
            return db

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with (
        patch("worker.planning_tasks.push_dead_letter", new=AsyncMock()) as mock_push,
        patch("worker.planning_tasks.async_session_maker", return_value=SessionContext()),
        patch("worker.planning_tasks.PlanningJobRepository", return_value=repo),
        patch("worker.planning_tasks.incr") as mock_incr,
    ):
        await planning_tasks._finalize_dead_letter("job-failed", ValueError("bad input"), "task-1")

    mock_push.assert_awaited_once()
    assert mock_push.await_args.kwargs["job_id"] == "job-failed"
    assert mock_push.await_args.kwargs["task_id"] == "task-1"
    repo.update_status.assert_awaited_once_with("job-failed", "failed")
    db.commit.assert_awaited_once()
    mock_incr.assert_called_once_with("planning_jobs_failed_total")
