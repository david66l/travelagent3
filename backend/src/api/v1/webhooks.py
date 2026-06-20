"""External event webhook endpoints (PRD §7.5)."""

from fastapi import APIRouter, Request

from core.responses import success_response
from perception.webhook_handler import WebhookHandler

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/events")
async def receive_event(request: Request):
    """Receive external events (weather, attraction closure, traffic, flight changes)."""
    body = await request.json()
    ok, error = WebhookHandler.validate(body)
    if not ok:
        return success_response(data={"received": False, "error": error}, status_code=400)

    await WebhookHandler.enqueue(body)
    return success_response(
        data={"received": True, "event_type": body.get("type")},
        status_code=202,
    )


@router.get("/events/pending")
async def list_pending(limit: int = 100):
    """List pending replan events from Redis without removing them."""
    events = await WebhookHandler.list_pending(limit=limit)
    return success_response(data={"events": events})


@router.delete("/events/clear")
async def clear_events():
    """Clear the replan event queue."""
    await WebhookHandler.clear()
    return success_response(data={"cleared": True})
