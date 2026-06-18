"""Guest account limits (PRD §4.1)."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import ForbiddenException
from core.settings import settings
from repositories.planning_job import PlanningJobRepository


async def ensure_guest_can_plan(db: AsyncSession, user_id: UUID, role: str) -> None:
    """Guests may complete at most one itinerary."""
    if role != "guest":
        return
    repo = PlanningJobRepository(db)
    completed = await repo.count_completed_for_user(user_id)
    if completed >= settings.guest_max_completed_itineraries:
        raise ForbiddenException(
            "游客仅可体验一次完整行程规划，请注册账号继续使用",
            code="GUEST_LIMIT_EXCEEDED",
        )
