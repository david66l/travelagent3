"""Webhook event normalizer for external travel events."""

from __future__ import annotations

import json
import logging
from typing import Any

from core.redis_client import redis_client

logger = logging.getLogger(__name__)

REPLAN_QUEUE_KEY = "replan_queue"
VALID_EVENT_TYPES = {
    "weather_alert",
    "attraction_closed",
    "traffic_delay",
    "flight_changed",
    "user_manual_update",
}


class WebhookHandler:
    """Receive, validate, and enqueue external events for async replanning."""

    @staticmethod
    def validate(payload: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate that the payload is a recognized external event."""
        event_type = payload.get("type")
        if not event_type:
            return False, "missing 'type'"
        if event_type not in VALID_EVENT_TYPES:
            return False, f"unsupported event type: {event_type}"
        if "payload" not in payload:
            return False, "missing 'payload'"
        return True, None

    @staticmethod
    async def enqueue(payload: dict[str, Any]) -> None:
        """Push a validated event into the Redis replan queue."""
        await redis_client.lpush(REPLAN_QUEUE_KEY, json.dumps(payload))
        logger.info("Enqueued replan event: %s", payload.get("type"))

    @staticmethod
    async def list_pending(limit: int = 100) -> list[dict[str, Any]]:
        """Return pending events without removing them."""
        raw = await redis_client.lrange(REPLAN_QUEUE_KEY, 0, limit - 1)
        events: list[dict[str, Any]] = []
        for item in raw:
            try:
                events.append(json.loads(item))
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in replan_queue: %s", item)
        return events

    @staticmethod
    async def clear() -> None:
        """Clear the replan queue."""
        await redis_client.delete(REPLAN_QUEUE_KEY)
