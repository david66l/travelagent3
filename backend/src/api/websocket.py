"""WebSocket endpoint for real-time chat with the TravelAgent."""

import asyncio
import json
import logging
import time as _time
from copy import deepcopy

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from core.conversation_state import (
    default_conversation_state,
    append_message,
    merge_profile,
    is_profile_ready,
)
from core.input_guard import sanitize_user_input
from core.database import async_session_maker
from core.redis_client import redis_client
from repositories.planning_job import PlanningJobRepository

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage active WebSocket connections by session.

    ``_states`` is a hot-cache — the DB (``user_feedback``) is the
    authoritative source.  State is loaded on first message and
    persisted after every update.
    """

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}
        self._states: dict[str, dict] = {}

    # -- connection lifecycle --

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self._connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self._connections.pop(session_id, None)
        self._states.pop(session_id, None)

    async def send_json(self, session_id: str, data: dict):
        ws = self._connections.get(session_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                pass

    # -- state management --

    async def load_state(self, session_id: str) -> dict:
        """Load conversation state from the most recent job for *session_id*."""
        if session_id in self._states:
            return self._states[session_id]

        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)
            jobs = await repo.get_by_session(session_id, limit=1)
            if jobs and jobs[0].user_feedback:
                state = deepcopy(jobs[0].user_feedback)
                # Ensure all expected keys exist after a schema change
                defaults = default_conversation_state()
                for key in defaults:
                    if key not in state:
                        state[key] = defaults[key]
                self._states[session_id] = state
                return state

        state = default_conversation_state()
        self._states[session_id] = state
        return state

    async def save_state(self, job_id: str, session_id: str, state: dict):
        """Persist state to DB and update hot-cache."""
        state["updated_at"] = int(_time.time())
        async with async_session_maker() as db:
            repo = PlanningJobRepository(db)
            await repo.update_user_feedback(job_id, state)
            await db.commit()
        self._states[session_id] = state


manager = ConnectionManager()


@router.websocket("/ws/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint — creates job, pushes status, supports cancellation."""
    await manager.connect(session_id, websocket)
    active_job_id: str | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json(session_id, {"error": "Invalid JSON", "type": "error"})
                continue

            msg_type = msg.get("type", "chat")

            if msg_type == "chat":
                content = msg.get("content", "").strip()
                if not content:
                    await manager.send_json(session_id, {"error": "Empty message", "type": "error"})
                    continue

                safe_content = sanitize_user_input(content)
                user_id = msg.get("user_id", "anonymous")

                # 1. Load or create conversation state
                state = await manager.load_state(session_id)

                # 2. Append user message to recent history
                append_message(state, "user", safe_content)
                state["turn"] += 1

                # 3. Lightweight intent detection + profile merge
                from agents.intent_recognition import IntentRecognitionAgent
                from schemas import ProfilePatch
                intent_agent = IntentRecognitionAgent()
                intent_result = intent_agent._fallback_result(safe_content)

                # Convert flat user_entities into a ProfilePatch for merge
                patch = _entities_to_patch(intent_result.user_entities)
                state["profile"] = merge_profile(state["profile"], patch)

                state["last_intent"] = intent_result.intent
                state["missing_required"] = intent_result.missing_required

                # 4. Check if we have enough to plan
                if not is_profile_ready(state["profile"]):
                    await manager.send_json(session_id, {
                        "type": "needs_clarification",
                        "profile": state["profile"],
                        "missing_required": intent_result.missing_required,
                        "questions": intent_result.clarification_questions or [
                            "请问您想去哪个目的地？",
                            "计划玩几天？",
                        ],
                    })
                    continue

                # 5. Create job with accumulated state
                state["phase"] = "planning"
                state["revision"] = state.get("revision", 1) + 1
                async with async_session_maker() as db:
                    repo = PlanningJobRepository(db)
                    job = await repo.create(
                        session_id=session_id,
                        user_id=user_id,
                        user_input=safe_content,
                        user_feedback=deepcopy(state),
                    )
                    await db.commit()

                active_job_id = job.id
                await manager.send_json(session_id, {
                    "type": "job_created",
                    "job_id": job.id,
                    "status": "pending",
                })

                # 6. Wake up worker + start status pusher
                await redis_client._client.publish("jobs:available", job.id)
                asyncio.create_task(
                    push_job_status(job.id, session_id, from_event_id=0)
                )

            elif msg_type == "subscribe":
                # Reconnect: restore state + subscribe to existing job
                job_id = msg.get("job_id")
                last_event_id = msg.get("last_event_id", 0)
                if job_id:
                    # Restore conversation state from DB
                    async with async_session_maker() as db:
                        repo = PlanningJobRepository(db)
                        job = await repo.get(job_id)
                        if job and job.user_feedback:
                            state = dict(job.user_feedback)
                            manager._states[session_id] = state
                            await manager.send_json(session_id, {
                                "type": "state_restored",
                                "profile": state.get("profile", {}),
                                "phase": state.get("phase", "gathering"),
                                "recent_messages": state.get("recent_messages", [])[-3:],
                            })
                    # Resume status push
                    asyncio.create_task(
                        push_job_status(job_id, session_id, from_event_id=last_event_id)
                    )

            elif msg_type == "cancel":
                job_id = msg.get("job_id") or active_job_id
                if job_id:
                    async with async_session_maker() as db:
                        repo = PlanningJobRepository(db)
                        await repo.request_cancel(job_id)
                        await db.commit()
                    # Fast signal to worker
                    await redis_client._client.publish(f"job:cancel:{job_id}", "cancel")
                    active_job_id = None

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(session_id)
        if active_job_id:
            # Grace period: if user disconnected, cancel after 30s
            asyncio.create_task(_delayed_cancel(active_job_id, delay=30))


async def push_job_status(
    job_id: str, session_id: str, from_event_id: int = 0
):
    """Push job status events to frontend via DB polling + Redis pub/sub.

    Strategy:
    1. Drain any historical events already in the DB.
    2. Subscribe to Redis ``job:status:{job_id}`` for fast wake-up.
    3. Poll every *poll_interval* seconds as a safety net.
    4. When either mechanism detects a terminal state, do one final
       drain and return.
    """
    last_event_id = from_event_id

    async def _is_done(db: AsyncSession) -> bool:
        repo = PlanningJobRepository(db)
        job = await repo.get(job_id)
        return bool(job and job.status in ("completed", "failed", "cancelled"))

    async def _drain(db: AsyncSession) -> bool:
        """Fetch and push new events; return True if anything was sent."""
        nonlocal last_event_id
        repo = PlanningJobRepository(db)
        events = await repo.get_events_after(job_id, last_event_id)
        for event in events:
            last_event_id = event.id
            await manager.send_json(session_id, {
                "event_id": event.id,
                "type": "stage",
                "stage": event.stage,
                "event_type": event.event_type,
                "payload": event.payload,
            })
        return len(events) > 0

    # 1. Drain historical events
    async with async_session_maker() as db:
        while await _drain(db):
            pass
        if await _is_done(db):
            return

    # 2. Redis listener
    async def _redis_sub():
        try:
            pubsub = redis_client._client.pubsub()
            await pubsub.subscribe(f"job:status:{job_id}")
            try:
                async for message in pubsub.listen():
                    if message["type"] == "message":
                        async with async_session_maker() as db:
                            await _drain(db)
                    async with async_session_maker() as db:
                        if await _is_done(db):
                            return
            finally:
                await pubsub.unsubscribe(f"job:status:{job_id}")
        except Exception:
            logger.debug("Redis listener for %s exited", job_id, exc_info=True)

    # 3. Periodic poll — start with an immediate probe then sleep-loop
    async def _poll():
        try:
            # Immediate first poll to catch events that landed during setup
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

    # Let the other task finish naturally if it is already on its way out
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # 4. Final drain — push any events that landed between the last poll
    #    and the moment the job was marked terminal
    async with async_session_maker() as db:
        await _drain(db)


def _build_response(state: dict) -> dict:
    """Build response from graph state (kept for backward compat)."""
    return {
        "type": "message",
        "assistant_message": state.get("assistant_response") or "",
        "intent": state.get("intent"),
        "itinerary": state.get("current_itinerary"),
        "itinerary_status": state.get("itinerary_status"),
        "budget_panel": state.get("budget_panel"),
        "preference_panel": state.get("preference_panel"),
        "validation_result": state.get("validation_result"),
        "optimized_routes": state.get("optimized_routes"),
        "needs_clarification": state.get("needs_clarification", False),
        "waiting_for_confirmation": state.get("waiting_for_confirmation", False),
        "needs_replan": state.get("needs_replan", False),
    }


async def _delayed_cancel(job_id: str, delay: int):
    """Cancel job after grace period if user disconnected."""
    await asyncio.sleep(delay)
    async with async_session_maker() as db:
        repo = PlanningJobRepository(db)
        job = await repo.get(job_id)
        if job and job.status in ("pending", "running"):
            await repo.request_cancel(job_id)
            await db.commit()
        await redis_client._client.publish(f"job:cancel:{job_id}", "cancel")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _entities_to_patch(entities: dict):
    """Convert flat ``user_entities`` dict into a ``ProfilePatch``.

    Scalars go into ``set``; list values go into ``add`` so they
    accumulate across turns rather than overwrite.
    """
    from schemas import ProfilePatch

    scalar_keys = {
        "destination", "travel_days", "travel_dates",
        "travelers_count", "travelers_type", "pace", "budget_range",
    }
    list_keys = {"interests", "food_preferences", "avoid", "special_requests"}

    set_patch = {}
    add_patch = {}
    for key, value in entities.items():
        if value is None:
            continue
        if key in scalar_keys:
            set_patch[key] = value
        elif key in list_keys and isinstance(value, list):
            add_patch[key] = value

    return ProfilePatch(set=set_patch, add=add_patch)
