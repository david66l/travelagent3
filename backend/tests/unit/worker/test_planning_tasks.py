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
