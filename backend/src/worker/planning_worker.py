"""Background worker that pulls planning jobs from DB and executes them."""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Optional

from core.clock import utc_now_naive
from core.database import async_session_maker
from core.memory import memory_manager
from core.metrics import (
    incr,
    record_agent_terminal_outcome,
    record_planning_notification_failure,
    record_session_lock_wait,
)
from core.redis_client import redis_client
from models.planning_job import PlanningJob
from repositories.planning_job import PlanningJobRepository

logger = logging.getLogger(__name__)

_PROGRESS_PAYLOAD_KEYS = {
    "stage",
    "next_action",
    "agent_status",
    "termination_reason",
    "agent_step",
    "current_task_id",
    "solve_status",
    "solve_time_ms",
    "confidence",
    "missing_slots",
    "clarification_questions",
    "intent_ready_message",
}


def _compact_event_payload(event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Keep replay events bounded while the graph checkpoint stores full state."""
    if event_type in {"final", "awaiting_confirm"}:
        return dict(payload)
    if event_type == "error":
        return {
            key: payload[key]
            for key in ("error", "error_type", "retryable", "stage")
            if key in payload
        }
    if event_type == "clarify":
        compact = {
            key: payload[key]
            for key in _PROGRESS_PAYLOAD_KEYS | {"profile", "slots", "missing_slots"}
            if key in payload
        }
        messages = payload.get("messages")
        if isinstance(messages, list) and messages:
            message = messages[-1]
            if isinstance(message, dict):
                compact["messages"] = [
                    {
                        key: message[key]
                        for key in ("role", "content", "type", "task_id", "question_id")
                        if key in message
                    }
                ]
        return compact
    if event_type == "intent_ready":
        return {
            key: payload[key]
            for key in _PROGRESS_PAYLOAD_KEYS
            | {"profile", "slots", "missing_slots", "feasibility_report"}
            if key in payload
        }
    if event_type == "partial":
        return {
            key: payload[key]
            for key in {
                "content",
                "itinerary",
                "output_pdf_url",
                "output_excel_url",
                "output_map_url",
                "agent_policy_routing",
            }
            if key in payload
        }
    if event_type == "tool_call":
        summaries: list[dict[str, Any]] = []
        for item in payload.get("tool_results") or []:
            if not isinstance(item, dict):
                continue
            observation = item.get("observation") or {}
            error = observation.get("error") or {}
            summaries.append(
                {
                    key: value
                    for key, value in {
                        "name": item.get("name") or item.get("tool"),
                        "ok": observation.get("ok", item.get("ok")),
                        "error_code": error.get("code") if isinstance(error, dict) else None,
                    }.items()
                    if value is not None
                }
            )
        return {"tool_results": summaries}
    return {key: payload[key] for key in _PROGRESS_PAYLOAD_KEYS if key in payload}


_PUBLIC_RESULT_KEYS = {
    "agent_policy_routing",
    "agent_status",
    "clarification_questions",
    "itinerary",
    "itinerary_final",
    "missing_slots",
    "next_action",
    "output_excel_url",
    "output_map_url",
    "output_pdf_url",
    "proposal_text",
    "solve_status",
    "stage",
    "termination_reason",
    "validation_report",
    "warnings",
}


def _public_job_result(state: dict[str, Any]) -> dict[str, Any]:
    """Project a checkpoint into a bounded client-visible terminal result."""
    return {key: state[key] for key in _PUBLIC_RESULT_KEYS if state.get(key) is not None}


def _terminal_job_metrics(
    state: dict[str, Any], created_at: datetime
) -> tuple[dict[str, int] | None, int]:
    """Extract auditable Agent cost counters plus end-to-end job latency."""
    if created_at.tzinfo is not None:
        created_at = created_at.replace(tzinfo=None)
    latency_ms = max(0, int((utc_now_naive() - created_at).total_seconds() * 1000))
    ledger = state.get("agent_ledger") or {}
    budget = ledger.get("budget") if isinstance(ledger, dict) else {}
    if not isinstance(budget, dict) or not budget:
        return None, latency_ms
    prompt_tokens = 0
    completion_tokens = 0
    episode = state.get("agent_episode") or {}
    steps = episode.get("steps") if isinstance(episode, dict) else []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        action = step.get("action") or {}
        metrics = action.get("inference_metrics") if isinstance(action, dict) else {}
        if not isinstance(metrics, dict):
            continue
        prompt_tokens += max(0, int(metrics.get("prompt_tokens") or 0))
        completion_tokens += max(0, int(metrics.get("completion_tokens") or 0))
    token_usage = {
        "total_tokens": max(0, int(budget.get("used_tokens") or 0)),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "llm_latency_ms": max(0, int(budget.get("used_latency_ms") or 0)),
        "agent_steps": max(0, int(budget.get("used_episode_steps") or 0)),
        "tool_calls": max(0, int(budget.get("used_tool_calls") or 0)),
        "solver_calls": max(0, int(budget.get("used_solver_calls") or 0)),
    }
    return token_usage, latency_ms


class PlanningWorker:
    """Lease-based worker that executes planning jobs asynchronously."""

    HEARTBEAT_INTERVAL = 10  # seconds
    LEASE_DURATION = 60  # seconds
    POLL_INTERVAL = 1  # seconds when no jobs
    SESSION_LOCK_TTL = 60  # renewed automatically while a graph is running
    SESSION_LOCK_POLL_SECONDS = 1.0

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self._running = False
        self._current_job_id: Optional[str] = None
        self._cancelled_jobs: set[str] = set()
        self._cancel_events: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    async def run(self):
        """Main loop: acquire jobs, execute pipeline, handle errors."""
        self._running = True
        logger.info(f"Worker {self.worker_id} started")

        while self._running:
            job: Optional[PlanningJob] = None
            async with async_session_maker() as db:
                repo = PlanningJobRepository(db)
                job = await repo.acquire_job(
                    self.worker_id,
                    lease_seconds=self.LEASE_DURATION,
                )
                if job is not None:
                    await db.commit()

            if job is None:
                await asyncio.sleep(self.POLL_INTERVAL)
                continue

            self._current_job_id = job.id
            logger.info(f"Worker {self.worker_id} acquired job {job.id}")

            try:
                await self._execute_job(job)
            except Exception as e:
                logger.exception(f"Job {job.id} failed: {e}")
                async with async_session_maker() as db:
                    repo = PlanningJobRepository(db)
                    await repo.release(job.id, self.worker_id, "failed", str(e))
                    await db.commit()

            self._current_job_id = None

    async def execute_job_by_id(self, job_id: str) -> bool:
        """Claim and run a single job (used by Celery executor)."""
        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)
            job = await repo.acquire_job_by_id(
                job_id,
                self.worker_id,
                lease_seconds=self.LEASE_DURATION,
            )
            if job is None:
                return False
            await db.commit()

        self._current_job_id = job.id
        logger.info("Worker %s executing job %s (direct)", self.worker_id, job.id)
        try:
            await self._execute_job(job)
        except Exception as e:
            logger.exception("Job %s failed: %s", job.id, e)
            async with async_session_maker() as db:
                repo = PlanningJobRepository(db)
                await repo.release(job.id, self.worker_id, "failed", str(e))
                await db.commit()
            # The Celery task owns retry/backoff/DLQ policy. Swallowing here
            # makes a failed graph look successful and permanently bypasses
            # that policy.
            raise
        finally:
            self._current_job_id = None
        return True

    def stop(self):
        """Signal the worker to stop after current job."""
        self._running = False
        logger.info(f"Worker {self.worker_id} stopping...")

    # ------------------------------------------------------------------ #
    # Job execution
    # ------------------------------------------------------------------ #

    async def _execute_job(self, job: PlanningJob):
        """Execute the planning pipeline for a single job."""
        cancel_event = asyncio.Event()
        self._cancel_events[job.id] = cancel_event

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_loop(job.id))

        # Start cancel watcher
        cancel_task = asyncio.create_task(self._cancel_watcher(job.id))

        status = "completed"
        durable_state: dict[str, Any] = {}
        try:
            async with self._acquire_session_run_lock(job.session_id or str(job.id), cancel_event):
                status = await self._run_graph_for_job(job, cancel_event)
                durable_state = await self._persist_session_state(job)
        except asyncio.CancelledError:
            if not cancel_event.is_set():
                raise
            status = "cancelled"
            await self.record_stage(job, "cancelled", {"stage": "cancelled"})
            async with async_session_maker() as db:
                repo = PlanningJobRepository(db)
                await repo.release(job.id, self.worker_id, "cancelled")
                await db.commit()
        finally:
            self._cancel_events.pop(job.id, None)
            heartbeat_task.cancel()
            cancel_task.cancel()
            cleanup_results = await asyncio.gather(
                heartbeat_task,
                cancel_task,
                return_exceptions=True,
            )
            for task_name, result in zip(
                ("heartbeat", "cancel_watcher"), cleanup_results, strict=True
            ):
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    # These tasks are coordination aids. Their cleanup happens
                    # after the durable graph outcome and must not turn a
                    # successful itinerary into a whole-job retry.
                    logger.warning(
                        "Job %s %s cleanup failed after graph completion: %s",
                        job.id,
                        task_name,
                        result,
                    )

        if status in ("completed", "awaiting_confirm", "failed"):
            final_status = "failed" if status == "failed" else "completed"
            token_usage, latency_ms = _terminal_job_metrics(durable_state, job.created_at)
            public_result = _public_job_result(durable_state)
            async with async_session_maker() as db:
                repo = PlanningJobRepository(db)
                await repo.update_result(
                    job.id,
                    result=public_result or None,
                    token_usage=token_usage,
                    latency_ms=latency_ms,
                )
                await repo.release(job.id, self.worker_id, final_status)
                await db.commit()
            incr(
                "planning_jobs_failed_total"
                if final_status == "failed"
                else "planning_jobs_completed_total"
            )
        record_agent_terminal_outcome(status)

    @asynccontextmanager
    async def _acquire_session_run_lock(
        self,
        session_id: str,
        cancel_event: asyncio.Event,
    ) -> AsyncIterator[None]:
        """Serialize graph mutations for one conversation across all workers."""
        lock_context = None
        lock_id = f"agent-run:{session_id}"
        wait_started = time.monotonic()
        try:
            while lock_context is None:
                if cancel_event.is_set():
                    record_session_lock_wait(time.monotonic() - wait_started, "cancelled")
                    raise asyncio.CancelledError("job cancelled while waiting for session lock")
                candidate = memory_manager.acquire_lock(
                    lock_id,
                    ttl=self.SESSION_LOCK_TTL,
                    blocking=True,
                    blocking_timeout=self.SESSION_LOCK_POLL_SECONDS,
                )
                try:
                    await candidate.__aenter__()
                    lock_context = candidate
                except RuntimeError as exc:
                    if not str(exc).startswith("Timeout acquiring lock for session"):
                        raise
            record_session_lock_wait(time.monotonic() - wait_started, "acquired")
        except asyncio.CancelledError:
            raise
        except Exception:
            record_session_lock_wait(time.monotonic() - wait_started, "error")
            raise

        try:
            yield
        finally:
            await lock_context.__aexit__(None, None, None)

    async def _heartbeat_loop(self, job_id: str):
        """Renew lease every HEARTBEAT_INTERVAL seconds."""
        while True:
            await asyncio.sleep(self.HEARTBEAT_INTERVAL)
            async with async_session_maker() as db:
                repo = PlanningJobRepository(db)
                ok = await repo.heartbeat(
                    job_id,
                    self.worker_id,
                    lease_seconds=self.LEASE_DURATION,
                )
                if not ok:
                    logger.warning(f"Worker {self.worker_id} lost lease for job {job_id}")
                    event = self._cancel_events.get(job_id)
                    if event:
                        event.set()
                    break
                await db.commit()

    async def _cancel_watcher(self, job_id: str):
        """Watch Redis for cancel signals."""
        pubsub = redis_client._client.pubsub()
        await pubsub.subscribe(f"job:cancel:{job_id}")
        try:
            # ``listen()`` performs an unbounded blocking read. With a finite
            # Redis socket timeout it raises periodically even when nothing is
            # wrong. Short polling keeps cancellation responsive without
            # treating an idle subscription as an infrastructure failure.
            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if message and message.get("type") == "message":
                    self._cancelled_jobs.add(job_id)
                    event = self._cancel_events.get(job_id)
                    if event:
                        event.set()
                    break
        finally:
            try:
                await pubsub.unsubscribe(f"job:cancel:{job_id}")
            finally:
                await pubsub.aclose()

    # ------------------------------------------------------------------ #
    # Graph execution
    # ------------------------------------------------------------------ #

    async def _run_graph_for_job(
        self,
        job: PlanningJob,
        cancel_event: asyncio.Event,
    ) -> str:
        """Execute the new LangGraph runner for a PlanningJob.

        Maps every public graph event to a durable PlanningJobEvent. Redis
        pub/sub only wakes connected clients; PostgreSQL remains the replay
        source after reconnects or API restarts.
        """
        from graph.runner import stream_graph_events

        session_id = job.session_id or str(job.id)
        user_id = job.user_id or "anonymous"
        user_input = job.user_input or ""

        feedback = job.user_feedback or {}
        runtime = feedback.get("_job_context") or {}
        recent = feedback.get("recent_messages", [])
        messages: list[dict] = []
        if isinstance(recent, list):
            messages = [
                {"role": m.get("role", "user"), "content": m.get("content", "")}
                for m in recent[-10:]
                if isinstance(m, dict)
            ]

        await self.record_stage(job, "running", {"stage": "running"})

        try:
            async for event in stream_graph_events(
                session_id=session_id,
                user_id=user_id,
                user_input=user_input,
                messages=messages,
                attachments=runtime.get("attachments_meta") or feedback.get("attachments_meta"),
                profile=feedback.get("profile") or {},
                user_role=str(runtime.get("user_role") or feedback.get("user_role") or "guest"),
                conversation_state=feedback if isinstance(feedback, dict) else None,
                action=str(runtime.get("action") or "chat"),
                action_payload=runtime.get("action_payload") or {},
                job_id=str(job.id),  # lets the output node stream polish tokens
            ):
                if cancel_event.is_set():
                    await self.record_stage(job, "cancelled", {"stage": "cancelled"})
                    async with async_session_maker() as db:
                        repo = PlanningJobRepository(db)
                        await repo.release(job.id, self.worker_id, "cancelled")
                        await db.commit()
                    return "cancelled"

                event_type = event.get("type")
                stage = event.get("stage", event_type)
                payload = event.get("payload") or {}

                if event_type == "thinking":
                    await self.record_stage(
                        job,
                        stage or "running",
                        _compact_event_payload("thinking", payload),
                        event_type="thinking",
                    )
                elif event_type == "tool_call":
                    await self.record_stage(
                        job,
                        "tools_executed",
                        _compact_event_payload("tool_call", payload),
                        event_type="tool_call",
                    )
                elif event_type == "partial":
                    await self.record_stage(
                        job,
                        stage or "writing",
                        _compact_event_payload("partial", payload),
                        event_type="partial",
                    )
                elif event_type == "clarify":
                    await self.record_stage(
                        job,
                        stage or "gathering",
                        _compact_event_payload("clarify", payload),
                        event_type="clarify",
                    )
                    return "completed"
                elif event_type == "intent_ready":
                    await self.record_stage(
                        job,
                        stage or "planning",
                        _compact_event_payload("intent_ready", payload),
                        event_type="intent_ready",
                    )
                elif event_type == "final":
                    if payload.get("agent_status") == "failed" or payload.get(
                        "termination_reason"
                    ) in {"policy_error_fallback", "agent_deadline_exceeded"}:
                        await self.record_stage(job, "failed", dict(payload), event_type="error")
                        return "failed"
                    await self.record_stage(job, "completed", dict(payload), event_type="final")
                    return "completed"
                elif event_type == "awaiting_confirm":
                    itinerary = payload.get("itinerary")
                    if not isinstance(itinerary, list) or not itinerary:
                        failure = {
                            "error": "confirmation checkpoint did not contain an itinerary",
                            "error_type": "INVALID_CONFIRMATION_CHECKPOINT",
                            "retryable": False,
                        }
                        await self.record_stage(job, "failed", failure, event_type="error")
                        return "failed"
                    await self.record_stage(
                        job,
                        "awaiting_confirm",
                        dict(payload),
                        event_type="awaiting_confirm",
                    )
                    return "awaiting_confirm"
                elif event_type == "error":
                    await self.record_stage(job, "failed", payload, event_type="error")
                    if payload.get("retryable") is True:
                        raise TimeoutError(payload.get("error", "transient graph error"))
                    return "failed"
                else:
                    await self.record_stage(
                        job,
                        stage or "running",
                        _compact_event_payload(str(event_type or "stage"), payload),
                        event_type=str(event_type or "stage"),
                    )
        except Exception:
            logger.exception("Graph execution failed for job %s", job.id)
            raise

        # Silent graph completion is a protocol violation. It must never inflate
        # success metrics or consume a user's itinerary quota.
        failure = {
            "error": "graph ended without final, clarify, awaiting_confirm, or error event",
            "error_type": "GRAPH_TERMINAL_EVENT_MISSING",
            "retryable": False,
        }
        await self.record_stage(job, "failed", failure, event_type="error")
        return "failed"

    async def _persist_session_state(self, job: PlanningJob) -> dict[str, Any]:
        """Copy the recoverable graph state into the durable job snapshot."""
        from graph.session_manager import SessionManager

        state = await SessionManager().load(job.session_id or str(job.id))
        if not state:
            return {}
        durable_state: dict[str, Any] = dict(state)
        durable_state.pop("_job_context", None)
        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)
            await repo.update_user_feedback(job.id, durable_state)
            await db.commit()
        return durable_state

    # ------------------------------------------------------------------ #
    # Cancellation API
    # ------------------------------------------------------------------ #

    async def check_cancelled(self, job_id: str) -> bool:
        """Check if job has been cancelled (DB is source of truth)."""
        event = self._cancel_events.get(job_id)
        if event and event.is_set():
            return True
        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)
            return await repo.is_cancelled(job_id)

    async def mark_cancelled(self, job_id: str) -> bool:
        """Confirm cancellation and release lock."""
        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)
            ok = await repo.confirm_cancel(job_id, self.worker_id)
            if ok:
                await repo.add_event(
                    job_id=job_id,
                    stage="cancelled",
                    event_type="cancelled",
                )
            await db.commit()
            return ok

    # ------------------------------------------------------------------ #
    # Stage recording + Redis notification
    # ------------------------------------------------------------------ #

    async def record_stage(
        self,
        job: PlanningJob,
        stage: str,
        payload: Optional[dict] = None,
        *,
        event_type: str = "stage",
    ):
        """Record stage completion to DB and notify via Redis."""
        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)

            # Update job status
            ok = await repo.update_stage(
                job.id,
                stage,
                payload,
                worker_id=self.worker_id,
            )
            if not ok:
                await db.rollback()
                return False

            # Add event log
            await repo.add_event(
                job_id=job.id,
                stage=stage,
                event_type=event_type,
                payload=payload,
            )

            await db.commit()

        # PostgreSQL is the replay source of truth. Redis only reduces client
        # polling latency; a pub/sub outage after the DB commit must not replay
        # the graph or turn an already committed Agent outcome into a retry.
        elapsed = (utc_now_naive() - job.created_at).total_seconds()
        notification = {
            "type": "stage",
            "stage": stage,
            "elapsed": round(elapsed, 1),
            "job_id": job.id,
        }
        try:
            await redis_client.set_json(
                f"job:status:{job.id}",
                notification,
                ttl=300,
            )
            await redis_client._client.publish(
                f"job:status:{job.id}",
                json.dumps(notification, ensure_ascii=False),
            )
        except Exception as exc:
            record_planning_notification_failure()
            logger.warning(
                "Durable stage %s for job %s committed, but Redis notification failed: %s",
                stage,
                job.id,
                exc,
            )
        return True


# Global worker instance (started in main.py lifespan)
planning_worker: Optional[PlanningWorker] = None


async def start_worker():
    """Start the planning worker in a background task."""
    global planning_worker
    import uuid as uuid_mod

    worker_id = f"worker-{uuid_mod.uuid4().hex[:8]}"
    planning_worker = PlanningWorker(worker_id)
    asyncio.create_task(planning_worker.run())
    logger.info(f"Planning worker {worker_id} started")


async def stop_worker():
    """Signal the worker to stop gracefully."""
    global planning_worker
    if planning_worker:
        planning_worker.stop()
        # Give a short grace period
        await asyncio.sleep(0.5)
        planning_worker = None
