"""SSE chat stream and message submission (PRD §7.3)."""

import asyncio
import logging
import time
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from api.chat_runtime import (
    format_sse_event,
    manager,
    process_chat_message,
    push_job_status,
    restore_session_state,
)
from api.deps import get_conversation_service, get_current_user
from api.v1.schemas import ChatMessageRequest
from core.database import async_session_maker
from core.exceptions import NotFoundException, RateLimitException
from core.redis_client import redis_client
from core.responses import success_response
from core.settings import settings
from models import User
from repositories.planning_job import PlanningJobRepository
from services import ConversationService

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


def _session_id_for(conversation_id: UUID) -> str:
    return str(conversation_id)


async def _ensure_conversation(
    conversation_id: UUID,
    user: User,
    service: ConversationService,
) -> None:
    conversation = await service.get(conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise NotFoundException("Conversation", conversation_id)


async def _track_sse_connection(user_id: UUID, timeout: int) -> str:
    key = f"sse:connections:{user_id}"
    count = await redis_client.incr(key)
    if count == 1:
        await redis_client.expire(key, timeout)
    if count > settings.rate_limit_max_concurrent_sse:
        await redis_client.decr(key)
        raise RateLimitException(
            "Too many concurrent SSE connections",
            retry_after=60,
        )
    return key


async def _release_sse_connection(key: str) -> None:
    try:
        remaining = await redis_client.decr(key)
        if remaining <= 0:
            await redis_client.delete(key)
    except Exception:
        logger.debug("Failed to release SSE connection counter for %s", key)


@router.get("/stream")
async def chat_stream(
    request: Request,
    conversation_id: UUID = Query(..., description="Conversation UUID"),
    timeout: int = Query(1800, ge=60, le=3600),
    last_event_id: int = Query(0, ge=0),
    job_id: Optional[str] = Query(None, description="Resume a specific job stream"),
    user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """SSE stream for chat events (stage / message / job / error / done)."""
    await _ensure_conversation(conversation_id, user, service)
    session_id = _session_id_for(conversation_id)
    sse_key = await _track_sse_connection(user.id, timeout)

    request_id = request.headers.get("X-Request-ID") or str(conversation_id)

    async def event_generator():
        queue = await manager.register_sse(session_id)
        push_task: asyncio.Task | None = None
        try:
            await restore_session_state(session_id)

            async with async_session_maker() as db:
                repo = PlanningJobRepository(db)
                if job_id:
                    push_task = asyncio.create_task(
                        push_job_status(job_id, session_id, from_event_id=last_event_id)
                    )
                else:
                    jobs = await repo.get_by_session(session_id, limit=1)
                    if jobs and jobs[0].status in ("pending", "running"):
                        push_task = asyncio.create_task(
                            push_job_status(
                                jobs[0].id,
                                session_id,
                                from_event_id=last_event_id,
                            )
                        )

            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                wait = min(30.0, deadline - time.monotonic())
                if wait <= 0:
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=wait)
                    yield format_sse_event(data)
                    if data.get("type") == "done":
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("SSE stream error for %s: %s", conversation_id, exc)
            yield format_sse_event({"type": "error", "error": str(exc)})
            yield format_sse_event({"type": "done"})
        finally:
            manager.unregister_sse(session_id, queue)
            if push_task and not push_task.done():
                push_task.cancel()
                try:
                    await push_task
                except asyncio.CancelledError:
                    pass
            await _release_sse_connection(sse_key)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-Request-ID": request_id,
    }
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=headers,
    )


@router.post("/message")
async def chat_message(
    body: ChatMessageRequest,
    user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """Submit a user message; results are pushed on the SSE stream."""
    await _ensure_conversation(body.conversation_id, user, service)
    session_id = _session_id_for(body.conversation_id)

    await service.add_message(body.conversation_id, "user", body.content)

    asyncio.create_task(
        process_chat_message(
            session_id,
            str(user.id),
            body.content,
            conversation_id=body.conversation_id,
        )
    )

    return success_response(
        data={
            "conversation_id": str(body.conversation_id),
            "status": "accepted",
            "stream": body.stream,
        },
        status_code=202,
    )
