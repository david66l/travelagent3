"""Lightweight service endpoints."""

import uuid

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from core.database import async_session_maker
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
async def readiness_check(response: Response) -> dict:
    """Readiness probe — verifies both stateful dependencies."""
    result: dict[str, str] = {"status": "ready"}
    try:
        if redis_client._client is None:
            await redis_client.connect()
        pong = await redis_client._client.ping()
        result["redis"] = "ok" if pong else "no_pong"
    except Exception as exc:
        result["redis"] = str(exc)

    try:
        async with async_session_maker() as db:
            await db.execute(text("SELECT 1"))
        result["database"] = "ok"
    except Exception as exc:
        result["database"] = str(exc)

    if result.get("redis") != "ok" or result.get("database") != "ok":
        result["status"] = "degraded"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return result


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
