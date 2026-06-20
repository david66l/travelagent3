"""Tests for PlanningWorker graph path."""

from __future__ import annotations

import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from worker.planning_worker import PlanningWorker


def _make_job(
    job_id: str = "job-123",
    session_id: str = "sess-123",
    user_id: str = "user-1",
    user_input: str = "北京3天",
    user_feedback: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=job_id,
        session_id=session_id,
        user_id=user_id,
        user_input=user_input,
        user_feedback=user_feedback or {},
        created_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_run_graph_for_job_maps_events_to_stages():
    """Worker should map graph streaming events to PlanningJobEvent rows."""
    worker = PlanningWorker("worker-test")

    async def _fake_stream(*, session_id, user_id, user_input, messages, attachments=None, **kwargs):
        yield {"type": "thinking", "stage": "understand", "payload": {}}
        yield {
            "type": "tool_call",
            "stage": "tools",
            "payload": {"tool_results": [{"tool": "weather"}]},
        }
        yield {
            "type": "partial",
            "stage": "output",
            "payload": {"content": "草稿"},
        }
        yield {
            "type": "final",
            "stage": "completed",
            "payload": {
                "content": "最终行程",
                "output_pdf_url": "http://pdf",
                "output_excel_url": "http://excel",
            },
        }

    job = _make_job()
    cancel_event = asyncio.Event()

    with patch("graph.runner.stream_graph_events", side_effect=_fake_stream):
        with patch.object(worker, "record_stage", new=AsyncMock(return_value=True)) as record:
            status = await worker._run_graph_for_job(job, cancel_event)

    assert status == "completed"
    recorded_stages = [call.args[1] for call in record.call_args_list]
    assert recorded_stages == [
        "running",
        "understand",
        "tools_executed",
        "output",
        "completed",
    ]
    final_call = record.call_args_list[-1]
    assert final_call.args[2]["content"] == "最终行程"
    assert final_call.args[2]["output_pdf_url"] == "http://pdf"
    assert final_call.args[2]["output_excel_url"] == "http://excel"


@pytest.mark.asyncio
async def test_run_graph_for_job_handles_cancellation():
    """Worker should release job as cancelled when cancel_event is set."""
    worker = PlanningWorker("worker-test")

    async def _fake_stream(*, session_id, user_id, user_input, messages, attachments=None, **kwargs):
        yield {"type": "thinking", "stage": "understand", "payload": {}}

    job = _make_job()
    cancel_event = asyncio.Event()
    cancel_event.set()

    with patch("graph.runner.stream_graph_events", side_effect=_fake_stream):
        with patch.object(worker, "record_stage", new=AsyncMock(return_value=True)) as record:
            mock_repo_cls = MagicMock()
            mock_repo_cls.return_value.release = AsyncMock(return_value=True)
            with patch("worker.planning_worker.PlanningJobRepository", mock_repo_cls):
                status = await worker._run_graph_for_job(job, cancel_event)

    assert status == "cancelled"
    recorded_stages = [call.args[1] for call in record.call_args_list]
    assert recorded_stages == ["running", "cancelled"]
    mock_repo_cls.return_value.release.assert_awaited_once_with(
        job.id, worker.worker_id, "cancelled"
    )


@pytest.mark.asyncio
async def test_run_graph_for_job_raises_on_error_event():
    """Worker should propagate error graph events as exceptions."""
    worker = PlanningWorker("worker-test")

    async def _fake_stream(*, session_id, user_id, user_input, messages, attachments=None, **kwargs):
        yield {"type": "error", "stage": "error", "payload": {"error": "graph exploded"}}

    job = _make_job()
    cancel_event = asyncio.Event()

    with patch("graph.runner.stream_graph_events", side_effect=_fake_stream):
        with patch.object(worker, "record_stage", new=AsyncMock(return_value=True)):
            with pytest.raises(RuntimeError, match="graph exploded"):
                await worker._run_graph_for_job(job, cancel_event)


@pytest.mark.asyncio
async def test_run_graph_for_job_uses_feedback_messages():
    """Worker should pass recent_messages from user_feedback to graph runner."""
    worker = PlanningWorker("worker-test")
    captured: dict = {}

    async def _fake_stream(*, session_id, user_id, user_input, messages, attachments=None, **kwargs):
        captured["messages"] = messages
        captured["conversation_state"] = kwargs.get("conversation_state")
        yield {"type": "final", "stage": "completed", "payload": {}}

    job = _make_job(
        user_feedback={
            "recent_messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
        }
    )
    cancel_event = asyncio.Event()

    with patch("graph.runner.stream_graph_events", side_effect=_fake_stream):
        with patch.object(worker, "record_stage", new=AsyncMock(return_value=True)):
            await worker._run_graph_for_job(job, cancel_event)

    assert captured["messages"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
