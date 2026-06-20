"""
LangGraph Agent SSE 端点 — 使用 StateGraph 编排全链路 Agent 流程。

每个节点完成后通过 SSE 实时推送状态变化。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from graph.runner import runner as graph_runner
from api.deps import get_conversation_service, get_current_user
from api.v1.schemas import ChatMessageRequest
from core.database import async_session_maker
from core.exceptions import NotFoundException
from core.redis_client import redis_client
from core.responses import success_response
from models import User
from services import ConversationService

router = APIRouter(prefix="/agent", tags=["agent"])
logger = logging.getLogger(__name__)

# SSE 事件格式化
def _sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _ensure_conversation(
    conversation_id: UUID, user: User, service: ConversationService
) -> None:
    conversation = await service.get(conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise NotFoundException("Conversation", conversation_id)


@router.get("/stream")
async def agent_stream(
    request: Request,
    conversation_id: UUID = Query(...),
    timeout: int = Query(1800, ge=60, le=3600),
    user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """LangGraph Agent SSE 流式端点。

    每个 Graph 节点完成时推送:
      event: node
      data: {"node": "understand", "stage": "understanding", ...}

    最终输出:
      event: message
      data: {"role": "assistant", "content": "..."}

      event: done
      data: {}
    """
    await _ensure_conversation(conversation_id, user, service)
    session_id = str(conversation_id)
    request_id = request.headers.get("X-Request-ID") or session_id

    # 加载历史消息和画像
    messages = []
    profile = {}
    try:
        from core.conversation_state import flatten_profile
        from core.redis_client import redis_client as rc

        state = await rc.get_json(f"session:{session_id}:state")
        if state:
            messages = state.get("recent_messages", []) or []
            profile = flatten_profile(state.get("profile", {}))
    except Exception:
        pass

    async def event_generator():
        """LangGraph streaming → SSE"""
        deadline = time.monotonic() + timeout
        queue: asyncio.Queue = asyncio.Queue()

        async def run_agent():
            try:
                async for event in agent_runner.stream(
                    "",  # 空消息，实际由 /message 端点触发
                    session_id=session_id,
                    user_id=str(user.id),
                    user_role=user.role,
                    messages=messages,
                    profile=profile,
                ):
                    await queue.put(event)
                await queue.put({"type": "done"})
            except Exception as exc:
                logger.exception("Agent stream error: %s", exc)
                await queue.put({"type": "error", "error": str(exc)})

        task = asyncio.create_task(run_agent())

        try:
            while time.monotonic() < deadline:
                try:
                    data = await asyncio.wait_for(
                        queue.get(), timeout=30.0
                    )
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if data.get("type") == "done":
                    yield _sse_event("done", {})
                    break
                if data.get("type") == "error":
                    yield _sse_event("error", {"error": data["error"]})
                    yield _sse_event("done", {})
                    break

                yield _sse_event("node", data)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

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
async def agent_message(
    body: ChatMessageRequest,
    user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """向 LangGraph Agent 提交消息，触发全链路规划。

    流程: understand → profile → knowledge → planner → output
    通过 SSE 推送各节点状态 + 最终行程。
    """
    await _ensure_conversation(body.conversation_id, user, service)
    session_id = str(body.conversation_id)

    # 安全检查
    from core.input_safety import validate_user_input
    from core.settings import settings

    if settings.input_safety_enabled:
        try:
            validate_user_input(body.content)
        except Exception:
            from core.metrics import record_prompt_injection_blocked

            record_prompt_injection_blocked()
            raise

    # 保存用户消息
    await service.add_message(body.conversation_id, "user", body.content)

    # 异步启动 Agent
    asyncio.create_task(
        _run_agent_and_notify(
            session_id=session_id,
            user_id=str(user.id),
            user_role=user.role,
            content=body.content,
            conversation_id=body.conversation_id,
        )
    )

    return success_response(
        data={
            "conversation_id": str(body.conversation_id),
            "status": "accepted",
        },
        status_code=202,
    )


async def _run_agent_and_notify(
    *,
    session_id: str,
    user_id: str,
    user_role: str,
    content: str,
    conversation_id: UUID,
):
    """后台运行 Agent 并通过 Redis Pub/Sub 推送结果。"""
    try:
        result = await graph_runner.run(
            user_input=content,
            session_id=session_id,
            user_id=user_id,
        )

        # 推送最终消息
        await redis_client._client.publish(
            f"session:{session_id}:events",
            json.dumps(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": result.get("content", ""),
                    "message_type": result.get("message_type"),
                    "itinerary": result.get("itinerary"),
                    "output_pdf_url": result.get("output_pdf_url"),
                    "output_excel_url": result.get("output_excel_url"),
                    "output_map_url": result.get("output_map_url"),
                    "warnings": result.get("warnings", []),
                },
                ensure_ascii=False,
            ),
        )

        # 推送完成
        await redis_client._client.publish(
            f"session:{session_id}:events",
            json.dumps({"type": "done"}, ensure_ascii=False),
        )

        # 保存会话状态
        try:
            state = {
                "profile": {
                    "destination": (result.get("itinerary") or [{}])[0].get("destination")
                    if result.get("itinerary")
                    else None,
                },
                "recent_messages": [
                    {"role": "assistant", "content": result.get("content", "")}
                ],
                "phase": "completed",
            }
            await redis_client.set_json(
                f"session:{session_id}:state", state, ttl=3600
            )
        except Exception:
            pass

        # 持久化行程到数据库
        _save_itinerary_background(conversation_id, user_id, result)

    except Exception as exc:
        logger.exception("Agent run failed for session %s: %s", session_id, exc)
        await redis_client._client.publish(
            f"session:{session_id}:events",
            json.dumps(
                {"type": "error", "error": "Agent 执行失败，请重试"},
                ensure_ascii=False,
            ),
        )


def _save_itinerary_background(
    conversation_id: UUID, user_id: str, result: dict
):
    """异步保存行程到数据库。"""
    import asyncio

    async def _save():
        try:
            from core.database import async_session_maker
            from repositories.v1 import ItineraryRepository

            itinerary = result.get("itinerary", [])
            if not itinerary:
                return

            async with async_session_maker() as db:
                repo = ItineraryRepository(db)
                await repo.create(
                    user_id=user_id,
                    conversation_id=conversation_id,
                    content=itinerary,
                )
                await db.commit()
        except Exception:
            pass

    asyncio.create_task(_save())
