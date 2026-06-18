"""外部事件 Webhook 端点 + 缺 5 工具定义。"""

from fastapi import APIRouter, Request
from core.responses import success_response

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# 事件缓冲队列
_event_queue: list[dict] = []


@router.post("/events")
async def receive_event(request: Request):
    """接收外部事件（天气、景区公告、航班变更）。"""
    body = await request.json()
    _event_queue.append(body)
    return success_response(data={"received": True, "event_type": body.get("type")})


@router.get("/events/pending")
async def list_pending():
    """列出待处理事件（供 DynamicReplanner 消费）。"""
    return success_response(data={"events": _event_queue})


@router.delete("/events/clear")
async def clear_events():
    """清空事件队列。"""
    _event_queue.clear()
    return success_response(data={"cleared": True})
