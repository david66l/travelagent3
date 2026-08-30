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
    flatten_profile,
)
from core.input_guard import sanitize_user_input
from core.output_safety import sanitize_itinerary_payload
from core.database import async_session_maker
from core.memory import memory_manager
from core.metrics import incr, set_active_sessions
from core.redis_client import redis_client
from core.settings import settings
from graph.runner import stream_graph_events
from repositories.planning_job import PlanningJobRepository

try:
    from worker.memory_tasks import archive_session, schedule_delayed_archive
except ImportError:
    archive_session = None  # type: ignore[misc, assignment]
    schedule_delayed_archive = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

STATE_TTL = 1800
STATE_KEY = "session:{}:state"


async def _persist_assistant_message(
    conversation_id: UUID | None,
    content: str,
    *,
    message_type: str,
    itinerary: list[dict[str, Any]] | None = None,
) -> None:
    """Persist assistant output alongside user messages.

    Redis remains the hot graph state, but conversation history must not lose
    every assistant turn when Redis expires or the service restarts.
    """
    if conversation_id is None or not content.strip():
        return
    try:
        from models import Message

        metadata: dict[str, Any] = {"message_type": message_type}
        if itinerary:
            metadata["itinerary"] = itinerary
        async with async_session_maker() as db:
            db.add(
                Message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=content,
                    token_count=0,
                    metadata_=metadata,
                )
            )
            await db.commit()
    except Exception as exc:
        logger.warning("Failed to persist assistant message for %s: %s", conversation_id, exc)


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


async def create_chat_planning_job(
    repo: PlanningJobRepository,
    *,
    session_id: str,
    user_id: str,
    user_uuid: UUID,
    conversation_id: UUID,
    content: str,
    user_role: str = "guest",
    attachments_meta: list[dict] | None = None,
    action: str = "chat",
    action_payload: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
    idempotency_fingerprint: str | None = None,
) -> str | None:
    """Persist one chat turn as the durable unit of graph execution.

    The API owns validation and transaction commit; a Celery or embedded
    worker owns LangGraph execution. Runtime-only turn metadata travels with
    the job so control actions and attachments work in either executor.
    """
    if action == "chat":
        safe_content = sanitize_user_input(content.strip())
        if not safe_content:
            return None
    else:
        safe_content = sanitize_user_input((content or "").strip())

    state = await manager.load_state(session_id)
    state["user_id"] = user_id
    state["user_role"] = user_role
    if attachments_meta:
        state["attachments_meta"] = attachments_meta
    state["_job_context"] = {
        "action": action,
        "action_payload": action_payload or {},
        "attachments_meta": attachments_meta or [],
        "user_role": user_role,
    }
    if idempotency_fingerprint:
        state["_idempotency_fingerprint"] = idempotency_fingerprint

    job = await repo.create(
        session_id,
        user_id,
        safe_content,
        user_feedback=state,
        user_uuid=user_uuid,
        conversation_id=conversation_id,
        idempotency_key=idempotency_key,
    )
    return str(job.id)


class ConnectionManager:
    """WebSocket connections and SSE subscriber queues keyed by session_id."""

    def __init__(self) -> None:
        self._connections: dict[str, Any] = {}
        self._sse_queues: dict[str, list[asyncio.Queue]] = defaultdict(list)
        # Buffer events when no SSE client is connected yet (e.g. postMessage
        # before openStream). Flushed on register_sse so polish tokens are not lost.
        self._pending_sse: dict[str, list[dict[str, Any]]] = defaultdict(list)
        # Chinese streaming providers often emit one character per token. A
        # five-day plan can exceed 800 events before the final confirmation
        # signal; the old 512-item live queue could drop that signal and leave
        # the UI stuck on the draft. Keep enough headroom for a full turn.
        self._pending_sse_max = 4096

    async def connect_ws(self, session_id: str, websocket: Any) -> None:
        await websocket.accept()
        self._connections[session_id] = websocket

    def disconnect_ws(self, session_id: str) -> None:
        self._connections.pop(session_id, None)

    def has_live_connection(self, session_id: str) -> bool:
        """Return true while any local WebSocket or SSE subscriber is active."""
        return session_id in self._connections or bool(self._sse_queues.get(session_id))

    def _total_sse_queues(self) -> int:
        return sum(len(qs) for qs in self._sse_queues.values())

    async def register_sse(self, session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=4096)
        self._sse_queues[session_id].append(queue)
        for item in self._pending_sse.pop(session_id, []):
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                logger.warning("SSE queue full while replaying pending for %s", session_id)
                break
        set_active_sessions(self._total_sse_queues())
        return queue

    def unregister_sse(self, session_id: str, queue: asyncio.Queue) -> None:
        subs = self._sse_queues.get(session_id, [])
        if queue in subs:
            subs.remove(queue)
        if not subs:
            self._sse_queues.pop(session_id, None)
        set_active_sessions(self._total_sse_queues())

    async def send_json(self, session_id: str, data: dict[str, Any]) -> None:
        ws = self._connections.get(session_id)
        if ws:
            try:
                # Normalize with default=str so non-serializable values (e.g. a
                # LangGraph Interrupt) degrade without changing the send_json
                # contract used by callers and tests.
                normalized = json.loads(json.dumps(data, ensure_ascii=False, default=str))
                await ws.send_json(normalized)
            except Exception as exc:
                logger.warning("Failed to send WebSocket message to %s: %s", session_id, exc)

        subs = list(self._sse_queues.get(session_id, []))
        if subs:
            for queue in subs:
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:
                    logger.warning("SSE queue full for session %s", session_id)
        else:
            pending = self._pending_sse[session_id]
            pending.append(data)
            if len(pending) > self._pending_sse_max:
                del pending[0 : len(pending) - self._pending_sse_max]

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
                logger.warning("Failed to load planning job fallback for %s: %s", session_id, exc)

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
            async with memory_manager.acquire_lock(session_id, blocking_timeout=0.5):
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
            async with memory_manager.acquire_lock(session_id, blocking_timeout=0.1):
                state["updated_at"] = int(_time.time())
                try:
                    await memory_manager.hot_set(session_id, state)
                    await memory_manager.warm_set(session_id, state)
                except Exception as exc:
                    logger.warning("Failed to persist gathering state for %s: %s", session_id, exc)
                try:
                    await redis_client.set_json(STATE_KEY.format(session_id), state, ttl=STATE_TTL)
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


def _apply_conversation_sync(state: dict[str, Any], payload: dict[str, Any] | None) -> None:
    """Merge gathering subgraph fields back into WS conversation state."""
    if not payload:
        return
    sync = payload.get("conversation_sync")
    if isinstance(sync, dict):
        for key, value in sync.items():
            if value is not None:
                state[key] = value
    if payload.get("profile"):
        state["profile"] = payload["profile"]
    embedded = payload.get("_conversation_state")
    if isinstance(embedded, dict):
        for key in ("turn", "recent_messages", "last_intent", "missing_required", "phase"):
            if key in embedded:
                state[key] = embedded[key]


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
    # default=str so non-JSON-serializable values (e.g. a LangGraph Interrupt
    # object that slipped into an event payload) degrade gracefully instead of
    # crashing the stream.
    payload = json.dumps(data, ensure_ascii=False, default=str)
    event_id = data.get("event_id")
    id_line = f"id: {event_id}\n" if event_id is not None else ""
    return f"event: {event_name}\n{id_line}data: {payload}\n\n"


async def _stream_graph_to_manager(
    session_id: str,
    user_id: str,
    user_input: str,
    state: dict[str, Any],
    action: str = "chat",
    action_payload: dict[str, Any] | None = None,
    conversation_id: UUID | None = None,
) -> None:
    """Run the LangGraph turn and forward events to the connection manager."""
    from core.conversation_state import append_message

    messages: list[dict[str, Any]] = []
    recent = state.get("recent_messages", [])
    if isinstance(recent, list):
        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in recent[-10:]
            if isinstance(m, dict)
        ]

    # Seed the graph with the profile already collected during the gathering phase.
    profile = state.get("profile") or {}
    slots = flatten_profile(profile)

    latest_assistant_content = ""
    async for event in stream_graph_events(
        session_id=session_id,
        user_id=user_id,
        user_input=user_input,
        messages=messages,
        attachments=state.get("attachments_meta"),
        profile=profile,
        user_role=str(state.get("user_role") or "guest"),
        slots=slots,
        conversation_state=state,
        action=action,
        action_payload=action_payload,
    ):
        event_type = event.get("type", "message")
        payload = event.get("payload", {})

        if event_type == "clarify":
            clarify_messages = payload.get("messages", []) if isinstance(payload, dict) else []
            clarify_content = (
                clarify_messages[-1].get("content", "")
                if clarify_messages and isinstance(clarify_messages[-1], dict)
                else ""
            )
            _apply_conversation_sync(state, payload if isinstance(payload, dict) else None)
            await manager.send_json(
                session_id,
                {
                    "type": "needs_clarification",
                    "profile": payload.get("profile", state.get("profile", {}))
                    if isinstance(payload, dict)
                    else state.get("profile", {}),
                    "questions": (
                        payload.get("clarification_questions", ["请补充更多信息"])
                        if isinstance(payload, dict)
                        else ["请补充更多信息"]
                    ),
                    "missing_required": (
                        payload.get("missing_slots", []) if isinstance(payload, dict) else []
                    ),
                },
            )
            await manager.save_gathering_state(session_id, state)
            await _persist_assistant_message(
                conversation_id, clarify_content, message_type="clarification"
            )
        elif event_type == "intent_ready":
            content = (
                payload.get("intent_ready_message", "") if isinstance(payload, dict) else ""
            ) or "意图识别已完成，接下来将进行大致的规划。"
            _apply_conversation_sync(state, payload if isinstance(payload, dict) else None)
            append_message(state, "assistant", content)
            await manager.send_json(
                session_id,
                {
                    "type": "intent_ready",
                    "content": content,
                    "profile": payload.get("profile", state.get("profile", {}))
                    if isinstance(payload, dict)
                    else state.get("profile", {}),
                },
            )
            await manager.send_json(
                session_id,
                {"type": "stage", "stage": "planning"},
            )
            await manager.save_gathering_state(session_id, state)
        elif event_type == "error":
            await manager.send_json(
                session_id,
                {"type": "error", "error": payload.get("error", "graph error")},
            )
        elif event_type == "final":
            # Emit the assistant message with itinerary and artifact URLs first,
            # then emit done so the frontend can render the result and close.
            await manager.send_json(
                session_id,
                {
                    "type": "message",
                    "role": "assistant",
                    "stage": payload.get("stage", "completed"),
                    "content": payload.get("content", ""),
                    "itinerary": payload.get("itinerary"),
                    "profile": payload.get("profile"),
                    "tool_results": payload.get("tool_results"),
                    "booking_results": payload.get("booking_results"),
                    "budget_breakdown": payload.get("budget_breakdown"),
                    "output_pdf_url": payload.get("output_pdf_url"),
                    "output_excel_url": payload.get("output_excel_url"),
                    "output_map_url": payload.get("output_map_url"),
                    "agent_policy_routing": payload.get("agent_policy_routing"),
                },
            )
            await manager.send_json(
                session_id,
                {
                    "type": "done",
                    "stage": payload.get("stage", "completed"),
                    "job_id": None,
                    "output_pdf_url": payload.get("output_pdf_url"),
                    "output_excel_url": payload.get("output_excel_url"),
                    "output_map_url": payload.get("output_map_url"),
                },
            )
            # Sync the graph result back to the legacy conversation state so the
            # next turn can resume with the enriched profile/itinerary.
            if payload.get("itinerary"):
                state["itinerary"] = payload["itinerary"]
            if payload.get("profile"):
                state["profile"] = payload["profile"]
            if payload.get("agent_policy_routing"):
                state["agent_policy_routing"] = payload["agent_policy_routing"]
            state.setdefault("recent_messages", []).append(
                {
                    "role": "assistant",
                    "content": payload.get("content", ""),
                    "itinerary": payload.get("itinerary"),
                    "output_pdf_url": payload.get("output_pdf_url"),
                    "output_excel_url": payload.get("output_excel_url"),
                    "output_map_url": payload.get("output_map_url"),
                    "ts": int(_time.time()),
                }
            )
            state["recent_messages"] = state["recent_messages"][-10:]
            state["phase"] = "completed"
            await manager.save_gathering_state(session_id, state)
            await _persist_assistant_message(
                conversation_id,
                payload.get("content", ""),
                message_type=payload.get("message_type", "itinerary"),
                itinerary=payload.get("itinerary"),
            )
        elif event_type == "awaiting_confirm":
            # Draft is ready; graph is paused at the confirm interrupt.
            if isinstance(payload, dict) and payload.get("itinerary"):
                state["itinerary"] = payload["itinerary"]
            state["phase"] = "awaiting_confirm"
            await manager.send_json(
                session_id,
                {
                    "type": "awaiting_confirm",
                    "itinerary": payload.get("itinerary") if isinstance(payload, dict) else None,
                    "warnings": payload.get("warnings", []) if isinstance(payload, dict) else [],
                    "agent_policy_routing": payload.get("agent_policy_routing")
                    if isinstance(payload, dict)
                    else None,
                },
            )
            if isinstance(payload, dict) and payload.get("agent_policy_routing"):
                state["agent_policy_routing"] = payload["agent_policy_routing"]
            await manager.save_gathering_state(session_id, state)
            await _persist_assistant_message(
                conversation_id,
                latest_assistant_content,
                message_type="itinerary_draft",
                itinerary=payload.get("itinerary") if isinstance(payload, dict) else None,
            )
        elif event_type in ("thinking", "tool_call"):
            await manager.send_json(
                session_id,
                {
                    "type": "stage",
                    "stage": event.get("stage", event_type),
                },
            )
        elif event_type == "stage":
            await manager.send_json(
                session_id,
                {
                    "type": "stage",
                    "stage": event.get("stage"),
                    "payload": payload,
                },
            )
        elif event_type == "partial":
            # Output node finished enrich + polish; surface prose + artifacts.
            if isinstance(payload, dict) and payload.get("content"):
                latest_assistant_content = str(payload["content"])
            if isinstance(payload, dict) and payload.get("agent_policy_routing"):
                state["agent_policy_routing"] = payload["agent_policy_routing"]
            await manager.send_json(
                session_id,
                {
                    "type": "partial",
                    "stage": event.get("stage", "writing"),
                    "payload": payload,
                },
            )
        else:
            await manager.send_json(
                session_id,
                {
                    "type": event_type,
                    "stage": event.get("stage"),
                    "payload": payload,
                },
            )


async def process_chat_message_graph(
    session_id: str,
    user_id: str,
    content: str,
    *,
    conversation_id: UUID | None = None,
    user_role: str = "guest",
    attachments_meta: list[dict] | None = None,
    action: str = "chat",
    action_payload: dict[str, Any] | None = None,
) -> str | None:
    """Graph-based message handler. All turns enter via the gathering subgraph."""
    # Control actions (confirm/modify/reject/trip_event) are button-driven and
    # may carry no free text; only "chat" turns require non-empty content.
    if action == "chat":
        safe_content = sanitize_user_input(content.strip())
        if not safe_content:
            await manager.send_json(session_id, {"error": "Empty message", "type": "error"})
            return None
    else:
        safe_content = sanitize_user_input((content or "").strip())

    state = await manager.load_state(session_id)
    state["user_id"] = user_id
    state["user_role"] = user_role
    if attachments_meta:
        state["attachments_meta"] = attachments_meta

    asyncio.create_task(
        _stream_graph_to_manager(
            session_id,
            user_id,
            safe_content,
            state,
            action=action,
            action_payload=action_payload,
            conversation_id=conversation_id,
        )
    )
    return None


async def process_chat_message(
    session_id: str,
    user_id: str,
    content: str,
    *,
    conversation_id: UUID | None = None,
    user_role: str = "guest",
    attachments_meta: list[dict] | None = None,
    action: str = "chat",
    action_payload: dict[str, Any] | None = None,
) -> str | None:
    """Handle one user message via LangGraph (gathering subgraph + planning)."""
    return await process_chat_message_graph(
        session_id,
        user_id,
        content,
        conversation_id=conversation_id,
        user_role=user_role,
        attachments_meta=attachments_meta,
        action=action,
        action_payload=action_payload,
    )


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


async def publish_live_stage(
    *,
    session_id: str | None,
    job_id: str | None,
    stage: str,
    payload: dict | None = None,
) -> None:
    """Push an ephemeral stage hint (writing, etc.) without a DB job event.

    WS clients receive it directly; SSE clients get it via the ``live:{job_id}``
    Redis channel that ``push_job_status`` forwards alongside token chunks.
    """
    message: dict[str, Any] = {
        "type": "stage",
        "stage": stage,
        "payload": payload or {},
    }
    if job_id:
        message["job_id"] = job_id
    if session_id:
        try:
            await manager.send_json(session_id, message)
        except Exception as exc:
            logger.debug("Live stage push to session %s failed: %s", session_id, exc)
    if job_id:
        try:
            await redis_client._client.publish(
                f"live:{job_id}",
                json.dumps(message, ensure_ascii=False),
            )
        except Exception as exc:
            logger.warning("Failed to publish live stage for job %s: %s", job_id, exc)


def _planning_event_message(
    event: Any,
    payload: dict[str, Any] | None,
    job_id: str,
) -> dict[str, Any]:
    """Project a durable worker event onto the public SSE contract."""
    body = payload or {}
    common = {"event_id": event.id, "job_id": job_id}
    event_type = str(event.event_type or "stage")

    if event_type == "clarify":
        messages = body.get("messages") or []
        questions = body.get("clarification_questions") or []
        if not questions and messages and isinstance(messages[-1], dict):
            content = str(messages[-1].get("content") or "").strip()
            questions = [content] if content else []
        return {
            **common,
            "type": "needs_clarification",
            "profile": body.get("profile", {}),
            "questions": questions or ["请补充更多信息"],
            "missing_required": body.get("missing_slots", []),
        }
    if event_type == "intent_ready":
        return {
            **common,
            "type": "intent_ready",
            "content": body.get("intent_ready_message")
            or "意图识别已完成，接下来将进行大致的规划。",
            "profile": body.get("profile", {}),
        }
    if event_type == "awaiting_confirm":
        return {
            **common,
            "type": "awaiting_confirm",
            "itinerary": body.get("itinerary"),
            "warnings": body.get("warnings", []),
            "agent_policy_routing": body.get("agent_policy_routing"),
        }
    if event_type in {"partial", "final"}:
        return {
            **common,
            "type": event_type,
            "stage": event.stage,
            "payload": body,
        }
    if event_type == "error":
        return {
            **common,
            "type": "error",
            "stage": event.stage,
            "error": body.get("error", "graph error"),
        }
    return {
        **common,
        "type": "stage",
        "stage": event.stage,
        "event_type": event_type,
        "payload": body,
    }


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
                _planning_event_message(event, payload, job_id),
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
            await pubsub.subscribe(f"live:{job_id}")
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

                    if channel in (f"token:{job_id}", f"live:{job_id}"):
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
                await pubsub.unsubscribe(f"live:{job_id}")
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
    """Project the unified session state back to an SSE client after reconnect."""
    state = await manager.load_state(session_id)
    if not manager._state_has_session_payload(state):
        return

    recent_messages = state.get("recent_messages", [])[-10:]
    itinerary = state.get("itinerary")
    if not itinerary:
        for message in reversed(recent_messages):
            if isinstance(message, dict) and message.get("itinerary"):
                itinerary = message["itinerary"]
                break

    from graph.approval import public_approval

    await manager.send_json(
        session_id,
        {
            "type": "state_restored",
            "profile": state.get("profile", {}),
            "phase": state.get("phase", "gathering"),
            "revision": state.get("revision", 1),
            "recent_messages": recent_messages,
            "itinerary": itinerary,
            "budget_breakdown": state.get("budget_breakdown"),
            "output_pdf_url": state.get("output_pdf_url"),
            "output_excel_url": state.get("output_excel_url"),
            "output_map_url": state.get("output_map_url"),
            "agent_policy_routing": state.get("agent_policy_routing"),
            "pending_approval": public_approval(state.get("pending_approval")),
        },
    )


async def delayed_cancel(
    job_id: str,
    delay: int,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
) -> None:
    """Cancel only when the session is still disconnected after the grace period."""
    await asyncio.sleep(delay)
    if session_id and manager.has_live_connection(session_id):
        return
    if session_id and user_id:
        try:
            if await redis_client.get(f"sse:session:{user_id}:{session_id}") is not None:
                return
        except Exception:
            logger.debug("Could not verify reconnect state for %s", session_id)
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
        asyncio.create_task(
            delayed_cancel(
                job_id,
                delay=30,
                session_id=session_id,
                user_id=user_id,
            )
        )
    if schedule_delayed_archive is not None and user_id and user_role != "guest":
        try:
            schedule_delayed_archive.delay(session_id, user_id)
        except Exception:
            logger.debug("Failed to enqueue delayed archive for %s", session_id)
