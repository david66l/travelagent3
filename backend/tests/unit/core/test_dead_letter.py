"""Tests for planning dead-letter queue."""

from unittest.mock import AsyncMock

import pytest

from core.dead_letter import (
    list_dead_letters,
    push_dead_letter,
    remove_dead_letter_by_task_id,
)


@pytest.mark.asyncio
async def test_push_and_list_dead_letter(mock_redis):
    stored: list[str] = []

    async def lpush(key, value):
        stored.append(value)
        return len(stored)

    async def lrange(key, start, end):
        if end == -1:
            return stored
        return stored[start : end + 1 if end >= 0 else None]

    mock_redis.lpush = AsyncMock(side_effect=lpush)
    mock_redis.lrange = AsyncMock(side_effect=lrange)

    task_id = await push_dead_letter(
        task_name="worker.planning_tasks.execute_planning_job",
        job_id="job-1",
        exception=RuntimeError("boom"),
        traceback_text="trace",
        kwargs={"job_id": "job-1"},
        task_id="tid-1",
    )
    assert task_id == "tid-1"

    items = await list_dead_letters(limit=10)
    assert len(items) == 1
    assert items[0]["task_id"] == "tid-1"
    assert items[0]["job_id"] == "job-1"
    assert items[0]["exception"] == "RuntimeError"


@pytest.mark.asyncio
async def test_remove_dead_letter_by_task_id(mock_redis):
    payload = '{"task_id":"tid-2","job_id":"j2"}'
    stored = [payload]

    async def lrange(key, start, end):
        return list(stored)

    async def lrem(key, count, value):
        if value in stored:
            stored.remove(value)
            return 1
        return 0

    mock_redis.lrange = AsyncMock(side_effect=lrange)
    mock_redis.lrem = AsyncMock(side_effect=lrem)

    assert await remove_dead_letter_by_task_id("tid-2") is True
    assert stored == []
    assert await remove_dead_letter_by_task_id("missing") is False
