"""Shared chat runtime: state, job push, and message handling for WS + SSE."""

import asyncio
import json
import logging
import time as _time
from collections import defaultdict
from copy import deepcopy
from typing import Any, cast
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from core.conversation_state import (
    append_message,
    default_conversation_state,
    is_profile_ready,
)
from core.conversation_turn import process_user_turn
from core.input_guard import sanitize_user_input
from core.output_safety import sanitize_itinerary_payload
from core.guest_policy import ensure_guest_can_plan
from core.database import async_session_maker
from core.memory import memory_manager
from core.metrics import incr
from core.redis_client import redis_client
from core.settings import settings
from repositories.planning_job import PlanningJobRepository

try:
    from worker.memory_tasks import archive_session, schedule_delayed_archive
except ImportError:
    archive_session = None  # type: ignore[misc, assignment]
    schedule_delayed_archive = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

STATE_TTL = 1800
STATE_KEY = "session:{}:state"


def enqueue_planning_job(job_id: str) -> None:
    """Dispatch planning execution to Celery or rely on embedded worker polling."""
    incr("planning_jobs_created_total")
    if settings.planning_executor == "embedded":
        return
    try:
        from worker.planning_tasks import execute_planning_job

        execute_planning_job.apply_async(
            args=[job_id],
            queue=settings.celery_planning_queue,
        )
    except Exception as exc:
        logger.exception("Failed to enqueue planning job %s: %s", job_id, exc)


class ConnectionManager:
    """WebSocket connections and SSE subscriber queues keyed by session_id."""

    def __init__(self) -> None:
        self._connections: dict[str, Any] = {}
        self._sse_queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

    async def connect_ws(self, session_id: str, websocket: Any) -> None:
        await websocket.accept()
        self._connections[session_id] = websocket

    def disconnect_ws(self, session_id: str) -> None:
        self._connections.pop(session_id, None)

    async def register_sse(self, session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)
        self._sse_queues[session_id].append(queue)
        return queue

    def unregister_sse(self, session_id: str, queue: asyncio.Queue) -> None:
        subs = self._sse_queues.get(session_id, [])
        if queue in subs:
            subs.remove(queue)

    async def send_json(self, session_id: str, data: dict[str, Any]) -> None:
        ws = self._connections.get(session_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception as exc:
                logger.warning("Failed to send WebSocket message to %s: %s", session_id, exc)

        for queue in list(self._sse_queues.get(session_id, [])):
            try:
                queue.put_nowait(data)
            except asyncio.QueueFull:
                logger.warning("SSE queue full for session %s", session_id)

    async def load_state(self, session_id: str) -> dict[str, Any]:
        """Recover session state: memory tiers (hot/warm/cold) then planning job fallback."""
        state: dict[str, Any]
        try:
            state = await memory_manager.load_state(session_id)
            state = self._ensure_schema(state)
        except Exception as exc:
            logger.warning("Failed to load memory state for %s: %s", session_id, exc)
            state = self._ensure_schema(default_conversation_state())

        if not self._state_has_session_payload(state):
            try:
                async with async_session_maker() as db:
                    repo = PlanningJobRepository(db)
                    jobs = await repo.get_by_session(session_id, limit=1)
                    if jobs and jobs[0].user_feedback:
                        state = self._ensure_schema(deepcopy(jobs[0].user_feedback))
            except Exception as exc:
                logger.warning(
                    "Failed to load planning job fallback for %s: %s", session_id, exc
                )

        try:
            await memory_manager.hot_set(session_id, state)
        except Exception as exc:
            logger.warning("Failed to write hot state for %s: %s", session_id, exc)
        return cast(dict[str, Any], state)

    def _state_has_session_payload(self, state: dict[str, Any]) -> bool:
        """True when state carries user/session data beyond an empty default."""
        if state.get("recent_messages"):
            return True
        if state.get("destination"):
            return True
        if state.get("conversation_id"):
            return True
        if state.get("turn", 0) > 0:
            return True
        defaults = default_conversation_state()
        if state.get("phase") != defaults.get("phase"):
            return True
        profile = state.get("profile") or {}
        default_profile = defaults.get("profile") or {}
        for key, value in profile.items():
            if value and value != default_profile.get(key):
                return True
        return False

    async def save_state(
        self,
        job_id: str,
        session_id: str,
        state: dict[str, Any],
        *,
        trigger_archive: bool = False,
    ) -> bool:
        """Persist planning/completed state. Returns False if another client holds the lock."""
        user_id: str | None = None
        try:
            async with memory_manager.acquire_lock(
                session_id, blocking_timeout=0.5
            ):
                state["updated_at"] = int(_time.time())
                user_id = state.get("user_id")
                await memory_manager.hot_set(session_id, state)
                async with async_session_maker() as db:
                    repo = PlanningJobRepository(db)
                    await repo.update_user_feedback(job_id, state)
                    await db.commit()
        except RuntimeError as exc:
            logger.warning("Session write lock conflict for %s: %s", session_id, exc)
            await self.send_json(
                session_id,
                {
                    "type": "error",
                    "code": "session_conflict",
                    "message": "会话正在其他设备上更新，请稍后重试",
                },
            )
            return False

        if trigger_archive and archive_session is not None:
            try:
                user_role = state.get("user_role")
                archive_session.delay(session_id, user_id, user_role)
            except Exception:
                logger.debug("Failed to enqueue archive_session for %s", session_id)
        return True

    async def save_gathering_state(self, session_id: str, state: dict[str, Any]) -> bool:
        """Persist gathering-phase state. Returns False if another client holds the lock."""
        try:
            async with memory_manager.acquire_lock(
                session_id, blocking_timeout=0.1
            ):
                state["updated_at"] = int(_time.time())
                try:
                    await memory_manager.hot_set(session_id, state)
                    await memory_manager.warm_set(session_id, state)
                except Exception as exc:
                    logger.warning(
                        "Failed to persist gathering state for %s: %s", session_id, exc
                    )
                try:
                    await redis_client.set_json(
                        STATE_KEY.format(session_id), state, ttl=STATE_TTL
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to sync gathering state to Redis for %s: %s",
                        session_id,
                        exc,
                    )
            return True
        except RuntimeError as exc:
            logger.warning("Session write lock conflict for %s: %s", session_id, exc)
            await self.send_json(
                session_id,
                {
                    "type": "error",
                    "code": "session_conflict",
                    "message": "会话正在其他设备上更新，请稍后重试",
                },
            )
            return False

    def _ensure_schema(self, state: dict[str, Any]) -> dict[str, Any]:
        defaults = default_conversation_state()
        for key in defaults:
            if key not in state:
                state[key] = defaults[key]
        return state


manager = ConnectionManager()


def format_sse_event(data: dict[str, Any]) -> str:
    """Serialize a payload as an SSE frame (event name + JSON data)."""
    event_type = data.get("type", "message")
    if event_type == "stage" or data.get("stage"):
        event_name = "stage"
    elif event_type in ("job_created", "revision_created"):
        event_name = "job"
    elif event_type == "error":
        event_name = "error"
    elif event_type == "token":
        event_name = "token"
    elif event_type == "done":
        event_name = "done"
    else:
        event_name = "message"
    payload = json.dumps(data, ensure_ascii=False)
    event_id = data.get("event_id")
    id_line = f"id: {event_id}\n" if event_id is not None else ""
    return f"event: {event_name}\n{id_line}data: {payload}\n\n"


async def process_chat_message(
    session_id: str,
    user_id: str,
    content: str,
    *,
    conversation_id: UUID | None = None,
    user_role: str = "guest",
) -> str | None:
    """Handle one user message: intent, job creation, status push. Returns job_id if created."""
    safe_content = sanitize_user_input(content.strip())
    if not safe_content:
        await manager.send_json(session_id, {"error": "Empty message", "type": "error"})
        return None

    state = await manager.load_state(session_id)
    state["user_id"] = user_id
    state["user_role"] = user_role

    intent_result = await process_user_turn(state, safe_content)

    if not is_profile_ready(state["profile"]):
        saved = await manager.save_gathering_state(session_id, state)
        if not saved:
            return None
        await manager.send_json(
            session_id,
            {
                "type": "needs_clarification",
                "profile": state["profile"],
                "missing_required": intent_result.missing_required,
                "questions": intent_result.clarification_questions
                or [
                    "请问您想去哪个目的地？",
                    "计划玩几天？",
                ],
            },
        )
        return None

    was_completed = state.get("phase") == "completed"
    state["phase"] = "planning"
    if was_completed:
        state["revision"] = state.get("revision", 1) + 1
        await manager.send_json(
            session_id,
            {
                "type": "revision_created",
                "revision": state["revision"],
                "profile": state["profile"],
            },
        )
    else:
        state.setdefault("revision", 1)

    user_uuid: UUID | None = None
    try:
        user_uuid = UUID(user_id)
    except (TypeError, ValueError):
        user_uuid = None

    async with async_session_maker() as db:
        if user_uuid:
            await ensure_guest_can_plan(db, user_uuid, user_role)
        repo = PlanningJobRepository(db)
        job = await repo.create(
            session_id=session_id,
            user_id=user_id,
            user_input=safe_content,
            user_feedback=deepcopy(state),
            user_uuid=user_uuid,
            conversation_id=conversation_id,
        )
        await db.commit()

    try:
        await redis_client.set_json(STATE_KEY.format(session_id), state, ttl=STATE_TTL)
    except Exception as exc:
        logger.warning("Failed to sync state to Redis for %s: %s", session_id, exc)

    await manager.send_json(
        session_id,
        {
            "type": "job_created",
            "job_id": job.id,
            "status": "pending",
        },
    )

    enqueue_planning_job(job.id)
    asyncio.create_task(push_job_status(job.id, session_id, from_event_id=0))
    return job.id


async def publish_token(job_id: str, chunk: str) -> None:
    """Publish a token chunk to Redis so all SSE subscribers can forward it."""
    try:
        await redis_client._client.publish(
            f"token:{job_id}",
            json.dumps(
                {"type": "token", "chunk": chunk, "job_id": job_id},
                ensure_ascii=False,
            ),
        )
    except Exception as exc:
        logger.warning("Failed to publish token for job %s: %s", job_id, exc)


async def push_job_status(job_id: str, session_id: str, from_event_id: int = 0) -> None:
    """Push job status events via DB polling + Redis pub/sub + token stream."""
    last_event_id = from_event_id

    async def _is_done(db: AsyncSession) -> bool:
        repo = PlanningJobRepository(db)
        job = await repo.get(job_id)
        return bool(job and job.status in ("completed", "failed", "cancelled", "force_cancelled"))

    async def _drain(db: AsyncSession) -> bool:
        nonlocal last_event_id
        repo = PlanningJobRepository(db)
        events = await repo.get_events_after(job_id, last_event_id)
        for event in events:
            last_event_id = event.id
            payload = event.payload
            if settings.output_safety_enabled and payload:
                payload = sanitize_itinerary_payload(payload)
            await manager.send_json(
                session_id,
                {
                    "event_id": event.id,
                    "type": "stage",
                    "stage": event.stage,
                    "event_type": event.event_type,
                    "payload": payload,
                    "job_id": job_id,
                },
            )
            if event.stage == "completed" and (event.payload or {}).get("needs_human"):
                await manager.send_json(
                    session_id,
                    {
                        "type": "needs_clarification",
                        "reason": "行程规划存在需要人工确认的约束冲突",
                        "violations": (event.payload or {}).get("violations", []),
                        "suggested_fixes": (event.payload or {}).get("suggested_fixes", []),
                    },
                )
        return len(events) > 0

    async with async_session_maker() as db:
        while await _drain(db):
            pass
        if await _is_done(db):
            await manager.send_json(session_id, {"type": "done", "job_id": job_id})
            return

    async def _redis_sub() -> None:
        try:
            pubsub = redis_client._client.pubsub()
            await pubsub.subscribe(f"job:status:{job_id}")
            await pubsub.subscribe(f"token:{job_id}")
            try:
                async for message in pubsub.listen():
                    if message["type"] != "message":
                        continue
                    channel = message.get("channel", "")
                    data: dict[str, Any] = {}
                    try:
                        data = json.loads(message.get("data", "{}"))
                    except json.JSONDecodeError:
                        logger.debug("Invalid Redis message on %s", channel)
                        continue

                    if channel == f"token:{job_id}":
                        await manager.send_json(session_id, data)
                        continue

                    async with async_session_maker() as db:
                        await _drain(db)
                    async with async_session_maker() as db:
                        if await _is_done(db):
                            return
            finally:
                await pubsub.unsubscribe(f"job:status:{job_id}")
                await pubsub.unsubscribe(f"token:{job_id}")
        except Exception:
            logger.debug("Redis listener for %s exited", job_id, exc_info=True)

    async def _poll() -> None:
        try:
            async with async_session_maker() as db:
                await _drain(db)
                if await _is_done(db):
                    return

            while True:
                await asyncio.sleep(2.0)
                async with async_session_maker() as db:
                    await _drain(db)
                    if await _is_done(db):
                        return
        except Exception:
            logger.debug("Periodic poll for %s exited", job_id, exc_info=True)

    redis_task = asyncio.create_task(_redis_sub())
    poll_task = asyncio.create_task(_poll())

    done, pending = await asyncio.wait(
        [redis_task, poll_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async with async_session_maker() as db:
        await _drain(db)

    state = await manager.load_state(session_id)
    if state.get("phase") == "planning":
        state["phase"] = "completed"
        append_message(state, "assistant", "行程已生成")
        await manager.save_state(job_id, session_id, state, trigger_archive=True)

    await manager.send_json(session_id, {"type": "done", "job_id": job_id})


async def restore_session_state(session_id: str) -> None:
    """Push state_restored to subscribers (SSE reconnect)."""
    async with async_session_maker() as db:
        repo = PlanningJobRepository(db)
        latest_jobs = await repo.get_by_session(session_id, limit=1)
        if latest_jobs and latest_jobs[0].user_feedback:
            state = dict(latest_jobs[0].user_feedback)
            await redis_client.set_json(STATE_KEY.format(session_id), state, ttl=STATE_TTL)
            await manager.send_json(
                session_id,
                {
                    "type": "state_restored",
                    "profile": state.get("profile", {}),
                    "phase": state.get("phase", "gathering"),
                    "revision": state.get("revision", 1),
                    "recent_messages": state.get("recent_messages", [])[-3:],
                },
            )


async def delayed_cancel(job_id: str, delay: int) -> None:
    """Cancel job after grace period if user disconnected."""
    await asyncio.sleep(delay)
    async with async_session_maker() as db:
        repo = PlanningJobRepository(db)
        job = await repo.get(job_id)
        if job and job.status in ("pending", "running"):
            await repo.request_cancel(job_id)
            await db.commit()
            schedule_cancel_enforcement(job_id)
        await redis_client._client.publish(f"job:cancel:{job_id}", "cancel")


def schedule_cancel_enforcement(job_id: str, delay: int = 30) -> None:
    """After cancel request, force-cancel if worker does not confirm (PRD §4.7)."""
    try:
        from worker.planning_tasks import enforce_cancel_deadline

        enforce_cancel_deadline.apply_async(args=[job_id], countdown=delay)
    except Exception:
        logger.debug("Failed to schedule force cancel for job %s", job_id)


def schedule_disconnect_cleanup(
    session_id: str,
    user_id: str | None,
    job_id: str | None,
    user_role: str | None = None,
) -> None:
    """On SSE/WS disconnect: delayed cancel + delayed archive for registered users."""
    if job_id:
        asyncio.create_task(delayed_cancel(job_id, delay=30))
    if schedule_delayed_archive is not None and user_id and user_role != "guest":
        try:
            schedule_delayed_archive.delay(session_id, user_id)
        except Exception:
            logger.debug("Failed to enqueue delayed archive for %s", session_id)
