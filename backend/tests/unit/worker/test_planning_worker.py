"""Tests for PlanningWorker graph path."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from worker.planning_worker import (
    PlanningWorker,
    _compact_event_payload,
    _public_job_result,
    _terminal_job_metrics,
)


def test_progress_event_payload_drops_full_agent_checkpoint():
    payload = {
        "stage": "agent_loop_step",
        "agent_status": "running",
        "agent_step": 4,
        "agent_ledger": {"artifacts": {"huge": "x" * 100_000}},
        "agent_episode": {"steps": ["x" * 100_000]},
    }

    compact = _compact_event_payload("stage", payload)

    assert compact == {
        "stage": "agent_loop_step",
        "agent_status": "running",
        "agent_step": 4,
    }


def test_terminal_job_projection_persists_cost_without_private_checkpoint():
    state = {
        "agent_status": "awaiting_confirmation",
        "termination_reason": "awaiting_user",
        "itinerary": [{"day_number": 1}],
        "agent_episode": {
            "steps": [
                {
                    "action": {
                        "inference_metrics": {
                            "prompt_tokens": 5900,
                            "completion_tokens": 420,
                        }
                    }
                }
            ]
        },
        "agent_ledger": {
            "budget": {
                "used_tokens": 6320,
                "used_latency_ms": 4100,
                "used_episode_steps": 10,
                "used_tool_calls": 13,
                "used_solver_calls": 1,
            }
        },
        "agent_policy_routing": {"completion_tokens": 420},
    }

    result = _public_job_result(state)
    usage, latency_ms = _terminal_job_metrics(
        state, datetime.now(UTC).replace(tzinfo=None) - timedelta(seconds=2)
    )

    assert result["itinerary"] == [{"day_number": 1}]
    assert "agent_episode" not in result
    assert "agent_ledger" not in result
    assert usage == {
        "total_tokens": 6320,
        "prompt_tokens": 5900,
        "completion_tokens": 420,
        "llm_latency_ms": 4100,
        "agent_steps": 10,
        "tool_calls": 13,
        "solver_calls": 1,
    }
    assert latency_ms >= 1900


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
        created_at=datetime.now(UTC).replace(tzinfo=None),
    )


@pytest.mark.asyncio
async def test_run_graph_for_job_maps_events_to_stages():
    """Worker should map graph streaming events to PlanningJobEvent rows."""
    worker = PlanningWorker("worker-test")

    async def _fake_stream(
        *, session_id, user_id, user_input, messages, attachments=None, **kwargs
    ):
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

    async def _fake_stream(
        *, session_id, user_id, user_input, messages, attachments=None, **kwargs
    ):
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
async def test_run_graph_for_job_marks_nonretryable_error_failed():
    """Business/policy failures are terminal outcomes, not infrastructure retries."""
    worker = PlanningWorker("worker-test")

    async def _fake_stream(
        *, session_id, user_id, user_input, messages, attachments=None, **kwargs
    ):
        yield {
            "type": "error",
            "stage": "error",
            "payload": {"error": "invalid policy output", "retryable": False},
        }

    job = _make_job()
    cancel_event = asyncio.Event()

    with patch("graph.runner.stream_graph_events", side_effect=_fake_stream):
        with patch.object(worker, "record_stage", new=AsyncMock(return_value=True)):
            status = await worker._run_graph_for_job(job, cancel_event)

    assert status == "failed"


@pytest.mark.asyncio
async def test_run_graph_for_job_propagates_retryable_error_event():
    worker = PlanningWorker("worker-test")

    async def _fake_stream(**kwargs):
        yield {
            "type": "error",
            "stage": "error",
            "payload": {"error": "upstream timed out", "retryable": True},
        }

    with patch("graph.runner.stream_graph_events", side_effect=_fake_stream):
        with patch.object(worker, "record_stage", new=AsyncMock(return_value=True)):
            with pytest.raises(TimeoutError, match="upstream timed out"):
                await worker._run_graph_for_job(_make_job(), asyncio.Event())


@pytest.mark.asyncio
async def test_run_graph_for_job_rejects_empty_confirmation_checkpoint():
    worker = PlanningWorker("worker-test")

    async def _fake_stream(**kwargs):
        yield {"type": "awaiting_confirm", "payload": {"itinerary": []}}

    with patch("graph.runner.stream_graph_events", side_effect=_fake_stream):
        with patch.object(worker, "record_stage", new=AsyncMock(return_value=True)) as record:
            status = await worker._run_graph_for_job(_make_job(), asyncio.Event())

    assert status == "failed"
    assert record.call_args_list[-1].args[1] == "failed"


@pytest.mark.asyncio
async def test_run_graph_for_job_rejects_silent_completion():
    worker = PlanningWorker("worker-test")

    async def _fake_stream(**kwargs):
        if False:
            yield {}

    with patch("graph.runner.stream_graph_events", side_effect=_fake_stream):
        with patch.object(worker, "record_stage", new=AsyncMock(return_value=True)) as record:
            status = await worker._run_graph_for_job(_make_job(), asyncio.Event())

    assert status == "failed"
    assert record.call_args_list[-1].args[2]["error_type"] == "GRAPH_TERMINAL_EVENT_MISSING"


@pytest.mark.asyncio
async def test_run_graph_for_job_uses_feedback_messages():
    """Worker should pass recent_messages from user_feedback to graph runner."""
    worker = PlanningWorker("worker-test")
    captured: dict = {}

    async def _fake_stream(
        *, session_id, user_id, user_input, messages, attachments=None, **kwargs
    ):
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


@pytest.mark.asyncio
async def test_session_run_lock_serializes_jobs_for_the_same_session():
    active = 0
    max_active = 0
    gate = asyncio.Lock()

    @asynccontextmanager
    async def _fake_lock(*args, **kwargs):
        nonlocal active, max_active
        async with gate:
            active += 1
            max_active = max(max_active, active)
            try:
                yield "token"
            finally:
                active -= 1

    worker_a = PlanningWorker("worker-a")
    worker_b = PlanningWorker("worker-b")

    async def _hold(worker: PlanningWorker):
        async with worker._acquire_session_run_lock("shared-session", asyncio.Event()):
            await asyncio.sleep(0.01)

    with patch("worker.planning_worker.memory_manager.acquire_lock", _fake_lock):
        await asyncio.gather(_hold(worker_a), _hold(worker_b))

    assert max_active == 1


@pytest.mark.asyncio
async def test_session_run_lock_stops_waiting_when_job_is_cancelled():
    cancel_event = asyncio.Event()
    attempts = 0

    @asynccontextmanager
    async def _busy_lock(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            cancel_event.set()
        raise RuntimeError("Timeout acquiring lock for session agent-run:sess")
        yield  # pragma: no cover

    worker = PlanningWorker("worker-a")
    with patch("worker.planning_worker.memory_manager.acquire_lock", _busy_lock):
        with pytest.raises(asyncio.CancelledError):
            async with worker._acquire_session_run_lock("sess", cancel_event):
                pass

    assert attempts == 1


@pytest.mark.asyncio
async def test_cancel_watcher_uses_idle_safe_polling_and_closes_pubsub():
    worker = PlanningWorker("worker-a")
    event = asyncio.Event()
    worker._cancel_events["job-1"] = event
    pubsub = AsyncMock()
    pubsub.get_message = AsyncMock(side_effect=[None, {"type": "message", "data": "cancel"}])
    redis = MagicMock()
    redis.pubsub.return_value = pubsub

    with patch.object(worker, "_cancelled_jobs", set()):
        with patch("worker.planning_worker.redis_client._client", redis):
            await worker._cancel_watcher("job-1")

    assert event.is_set()
    assert pubsub.get_message.await_count == 2
    pubsub.unsubscribe.assert_awaited_once_with("job:cancel:job-1")
    pubsub.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_record_stage_keeps_committed_result_when_redis_notification_fails():
    worker = PlanningWorker("worker-a")
    job = _make_job()
    db = AsyncMock()
    repo = AsyncMock()
    repo.update_stage.return_value = True

    class _SessionContext:
        async def __aenter__(self):
            return db

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with (
        patch("worker.planning_worker.async_session_maker", return_value=_SessionContext()),
        patch("worker.planning_worker.PlanningJobRepository", return_value=repo),
        patch(
            "worker.planning_worker.redis_client.set_json",
            new=AsyncMock(side_effect=ConnectionError("redis unavailable")),
        ),
        patch("worker.planning_worker.record_planning_notification_failure") as metric,
    ):
        ok = await worker.record_stage(
            job,
            "completed",
            {"itinerary": [{"day_number": 1}]},
            event_type="final",
        )

    assert ok is True
    db.commit.assert_awaited_once()
    repo.add_event.assert_awaited_once()
    metric.assert_called_once_with()


@pytest.mark.asyncio
async def test_execute_job_by_id_propagates_failure_to_celery_retry_owner():
    worker = PlanningWorker("worker-test")
    job = _make_job()
    db = AsyncMock()
    repo = AsyncMock()
    repo.acquire_job_by_id.return_value = job

    class _SessionContext:
        async def __aenter__(self):
            return db

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with (
        patch("worker.planning_worker.async_session_maker", return_value=_SessionContext()),
        patch("worker.planning_worker.PlanningJobRepository", return_value=repo),
        patch.object(worker, "_execute_job", new=AsyncMock(side_effect=TimeoutError("slow"))),
    ):
        with pytest.raises(TimeoutError, match="slow"):
            await worker.execute_job_by_id(job.id)

    repo.release.assert_awaited_once_with(job.id, worker.worker_id, "failed", "slow")
    assert worker._current_job_id is None
