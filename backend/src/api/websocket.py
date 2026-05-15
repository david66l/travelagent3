"""WebSocket endpoint for real-time chat with the TravelAgent."""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession

from core.input_guard import sanitize_user_input
from core.database import async_session_maker
from core.redis_client import redis_client
from repositories.planning_job import PlanningJobRepository

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage active WebSocket connections by session."""

    def __init__(self):
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self._connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self._connections.pop(session_id, None)

    async def send_json(self, session_id: str, data: dict):
        ws = self._connections.get(session_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                pass


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

                # 1. Create job
                async with async_session_maker() as db:
                    repo = PlanningJobRepository(db)
                    job = await repo.create(
                        session_id=session_id,
                        user_id=user_id,
                        user_input=safe_content,
                    )
                    await db.commit()

                active_job_id = job.id
                await manager.send_json(session_id, {
                    "type": "job_created",
                    "job_id": job.id,
                    "status": "pending",
                })

                # 2. Wake up worker via Redis
                await redis_client._client.publish("jobs:available", job.id)

                # 3. Start status pusher
                asyncio.create_task(
                    push_job_status(job.id, session_id, from_event_id=0)
                )

            elif msg_type == "subscribe":
                # Reconnect: subscribe to existing job
                job_id = msg.get("job_id")
                last_event_id = msg.get("last_event_id", 0)
                if job_id:
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
    """Push job status events to frontend via DB polling + Redis fallback."""
    last_event_id = from_event_id

    async def is_terminal(db: AsyncSession) -> bool:
        repo = PlanningJobRepository(db)
        job = await repo.get(job_id)
        return bool(job and job.status in ("completed", "failed", "cancelled"))

    async def poll_and_push(db: AsyncSession) -> bool:
        """Fetch new events from DB and push to frontend."""
        nonlocal last_event_id
        repo = PlanningJobRepository(db)
        events = await repo.get_events_after(job_id, last_event_id)
        for event in events:
            last_event_id = event.id
            await manager.send_json(session_id, {
                "event_id": event.id,
                "type": event.event_type,
                "stage": event.stage,
                "payload": event.payload,
            })
        return len(events) > 0

    # Push historical events first
    async with async_session_maker() as db:
        while await poll_and_push(db):
            pass
        if await is_terminal(db):
            return

    # Redis pub/sub + periodic polling
    poll_interval = 2.0

    async def redis_listener():
        pubsub = redis_client._client.pubsub()
        await pubsub.subscribe(f"job:status:{job_id}")
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    async with async_session_maker() as db:
                        await poll_and_push(db)
                # Check terminal state
                async with async_session_maker() as db:
                    if await is_terminal(db):
                        break
        finally:
            await pubsub.unsubscribe(f"job:status:{job_id}")

    async def periodic_poll():
        while True:
            await asyncio.sleep(poll_interval)
            async with async_session_maker() as db:
                await poll_and_push(db)
                if await is_terminal(db):
                    break

    redis_task = asyncio.create_task(redis_listener())
    poll_task = asyncio.create_task(periodic_poll())

    done, pending = await asyncio.wait(
        [redis_task, poll_task],
        return_when=asyncio.FIRST_COMPLETED,
    )

    if redis_task in done and poll_task not in done:
        await poll_task

    for task in (redis_task, poll_task):
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


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
