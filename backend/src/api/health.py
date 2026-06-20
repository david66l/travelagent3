"""Lightweight service endpoints."""

import uuid

from fastapi import APIRouter
from pydantic import BaseModel

from core.redis_client import redis_client

router = APIRouter(prefix="/api")


class CreateSessionResponse(BaseModel):
    session_id: str
    message: str = "Session created"


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session() -> CreateSessionResponse:
    """Create a new chat session id."""
    return CreateSessionResponse(session_id=str(uuid.uuid4()))


@router.get("/health")
async def health_check() -> dict:
    """Liveness probe used by Docker, K8s, and CI."""
    return {"status": "ok", "service": "travel-agent"}


@router.get("/ready")
async def readiness_check() -> dict:
    """Readiness probe — verifies Redis connectivity."""
    try:
        if redis_client._client is None:
            await redis_client.connect()
        pong = await redis_client._client.ping()
        if not pong:
            return {"status": "degraded", "redis": "no_pong"}
        return {"status": "ready", "redis": "ok"}
    except Exception as exc:
        return {"status": "degraded", "redis": str(exc)}


@router.get("/health/dependencies")
async def dependency_health() -> dict:
    """Health report for third-party dependencies."""
    from monitoring.health_checker import ThirdPartyHealthChecker

    checker = ThirdPartyHealthChecker()
    return await checker.health_report()


@router.get("/health/congestion")
async def congestion_status() -> dict:
    """Current system congestion signals."""
    from monitoring.congestion_detector import CongestionDetector

    return await CongestionDetector().detect()
