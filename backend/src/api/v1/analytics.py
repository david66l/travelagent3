"""Admin analytics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, require_admin
from core.responses import success_response
from models import User
from monitoring.log_analytics import LogAnalyticsEngine

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/analytics")
async def get_admin_analytics(
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Return planning-failure clusters, modification intents, destination ranking."""
    engine = LogAnalyticsEngine()
    report = await engine.analyze(db)
    return success_response(data=report)
