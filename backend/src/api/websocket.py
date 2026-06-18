"""WebSocket endpoint for real-time chat with the TravelAgent."""

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.chat_runtime import (
    manager,
    process_chat_message,
    push_job_status,
    restore_session_state,
    schedule_cancel_enforcement,
    schedule_disconnect_cleanup,
)
from core.database import async_session_maker
from core.redis_client import redis_client
from repositories.planning_job import PlanningJobRepository

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str) -> None:
    """WebSocket endpoint — creates job, pushes status, supports cancellation."""
    await manager.connect_ws(session_id, websocket)
    active_job_id: str | None = None
    last_user_id: str | None = None
    last_user_role: str = "guest"

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

                user_id = msg.get("user_id", "anonymous")
                user_role = msg.get("user_role", "guest")
                last_user_id = user_id
                last_user_role = user_role
                job_id = await process_chat_message(
                    session_id, user_id, content, user_role=user_role
                )
                if job_id:
                    active_job_id = job_id

            elif msg_type == "subscribe":
                job_id = msg.get("job_id")
                last_event_id = msg.get("last_event_id", 0)
                if job_id:
                    await restore_session_state(session_id)
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
                    await redis_client._client.publish(f"job:cancel:{job_id}", "cancel")
                    schedule_cancel_enforcement(job_id)
                    active_job_id = None

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect_ws(session_id)
        schedule_disconnect_cleanup(
            session_id,
            last_user_id if last_user_id and last_user_id != "anonymous" else None,
            active_job_id,
            last_user_role,
        )
