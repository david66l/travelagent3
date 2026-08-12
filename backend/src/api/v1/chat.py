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
    schedule_disconnect_cleanup,
)
from api.deps import get_conversation_service, get_current_user
from api.v1.schemas import ChatMessageRequest
from perception import AttachmentParser, build_text_perception
from core.database import async_session_maker
from core.exceptions import NotFoundException, RateLimitException
from core.input_safety import validate_user_input
from core.metrics import record_prompt_injection_blocked
from core.redis_client import redis_client
from core.responses import success_response
from core.settings import settings
from models import User
from monitoring.rate_limit_controller import RateLimitCostController
from repositories.planning_job import PlanningJobRepository
from services import ConversationService

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

# Global singleton: PaddleOCR is expensive to initialize, so share one parser
# across requests. Lazy-loading happens on first attachment parse.
_attachment_parser = AttachmentParser()


def get_attachment_parser() -> AttachmentParser:
    """Dependency provider for the shared attachment parser."""
    return _attachment_parser


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


def _sse_session_key(user_id: UUID, session_id: str) -> str:
    return f"sse:session:{user_id}:{session_id}"


async def _count_active_sse_sessions(user_id: UUID) -> int:
    """Count live SSE session keys (each key has TTL — stale slots self-heal)."""
    pattern = f"sse:session:{user_id}:*"
    cursor = 0
    total = 0
    while True:
        cursor, keys = await redis_client.scan(cursor, match=pattern, count=100)
        total += len(keys)
        if cursor == 0:
            break
    return total


async def _track_sse_connection(user_id: UUID, session_id: str, timeout: int) -> str:
    """Secondary SSE guard - primary control is at Go gateway.

    Tracks one slot per *conversation* (session_id). Reconnecting the same
    conversation refreshes TTL instead of consuming another concurrent slot,
    so frontend backoff/reconnect loops do not hit 429 spuriously.
    """
    key = _sse_session_key(user_id, session_id)
    is_reconnect = await redis_client.get(key) is not None
    if not is_reconnect:
        active = await _count_active_sse_sessions(user_id)
        limit = settings.rate_limit_max_concurrent_sse
        if settings.debug:
            limit = max(limit, 10)
        if active >= limit:
            raise RateLimitException(
                "Too many concurrent SSE connections",
                retry_after=30,
            )
    await redis_client.set(key, "1", ttl=timeout + 120)
    return key


async def _release_sse_connection(key: str) -> None:
    try:
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
    sse_key = await _track_sse_connection(user.id, session_id, timeout)

    request_id = request.headers.get("X-Request-ID") or str(conversation_id)

    async def event_generator():
        queue = await manager.register_sse(session_id)
        push_task: asyncio.Task | None = None
        active_job_id: str | None = job_id
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
                        active_job_id = jobs[0].id
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
            schedule_disconnect_cleanup(
                session_id,
                str(user.id),
                active_job_id,
                user.role,
            )

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
    request: Request,
    body: ChatMessageRequest,
    user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
    parser: AttachmentParser = Depends(get_attachment_parser),
):
    """Submit a user message; results are pushed on the SSE stream."""
    await _ensure_conversation(body.conversation_id, user, service)
    session_id = _session_id_for(body.conversation_id)

    # Unified rate / quota / cost guard.
    controller = RateLimitCostController()
    await controller.check_request_allowed(
        str(user.id),
        user.role,
        ip=request.client.host if request.client else None,
        concurrent_sse=False,
        estimated_tokens=0,
    )

    # --- Control actions (confirm/modify/reject/trip_event) ---
    # Button-driven; carry no new user text and bypass perception/safety.
    if body.action != "chat":
        if body.action == "modify":
            action_payload: dict | None = {"change": body.change}
        elif body.action == "trip_event":
            action_payload = body.external_event or {}
        else:
            action_payload = None
        asyncio.create_task(
            process_chat_message(
                session_id,
                str(user.id),
                body.content or "",
                conversation_id=body.conversation_id,
                user_role=user.role,
                action=body.action,
                action_payload=action_payload,
            )
        )
        return success_response(
            data={
                "conversation_id": str(body.conversation_id),
                "status": "accepted",
                "stream": body.stream,
                "action": body.action,
            },
            status_code=202,
        )

    # --- Perception layer: unify text + attachments ---
    attachment_metas = await parser.parse_many(
        [att.model_dump() for att in (body.attachments or [])]
    )

    extracted_texts: list[str] = []
    for meta in attachment_metas:
        if meta.get("extracted_text"):
            extracted_texts.append(meta["extracted_text"])

    user_input = body.content
    if extracted_texts:
        user_input = f"{body.content}\n\n[附件内容]\n" + "\n---\n".join(extracted_texts)

    perception = build_text_perception(user_input)
    perception["attachments_meta"] = attachment_metas
    # --- end perception layer ---

    if settings.input_safety_enabled:
        try:
            validate_user_input(user_input)
        except Exception:
            record_prompt_injection_blocked()
            raise

    message_metadata = {"attachments_meta": attachment_metas} if attachment_metas else None
    await service.add_message(body.conversation_id, "user", user_input, metadata=message_metadata)

    asyncio.create_task(
        process_chat_message(
            session_id,
            str(user.id),
            user_input,
            conversation_id=body.conversation_id,
            user_role=user.role,
            attachments_meta=attachment_metas,
        )
    )

    return success_response(
        data={
            "conversation_id": str(body.conversation_id),
            "status": "accepted",
            "stream": body.stream,
            "attachments_parsed": len(attachment_metas),
        },
        status_code=202,
    )
