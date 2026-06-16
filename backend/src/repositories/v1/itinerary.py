"""Itinerary repository."""

from typing import Optional, Sequence
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from models import Itinerary
from repositories.v1.base import BaseRepository


class ItineraryRepository(BaseRepository[Itinerary]):
    """Repository for Itinerary model."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, Itinerary)

    async def get_by_user(
        self,
        user_id: UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> Sequence[Itinerary]:
        """List itineraries for a user."""
        result = await self.db.execute(
            select(Itinerary)
            .where(Itinerary.user_id == user_id)
            .order_by(Itinerary.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def get_by_conversation(
        self,
        conversation_id: UUID,
    ) -> Sequence[Itinerary]:
        """List itineraries for a conversation."""
        result = await self.db.execute(
            select(Itinerary)
            .where(Itinerary.conversation_id == conversation_id)
            .order_by(Itinerary.created_at.desc())
        )
        return result.scalars().all()

    async def create_itinerary(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        destination: str,
        days: int,
        content: Optional[dict] = None,
        proposal_text: Optional[str] = None,
        job_id: Optional[str] = None,
    ) -> Itinerary:
        """Create a new itinerary."""
        itinerary = Itinerary(
            user_id=user_id,
            conversation_id=conversation_id,
            destination=destination,
            days=days,
            content=content or {},
            proposal_text=proposal_text,
            job_id=job_id,
        )
        return await self.create(itinerary)

    async def set_favorite(self, itinerary_id: UUID, is_favorite: bool) -> int:
        """Mark an itinerary as favorite or not."""
        result = await self.db.execute(
            update(Itinerary).where(Itinerary.id == itinerary_id).values(is_favorite=is_favorite)
        )
        return result.rowcount

    async def batch_favorite(
        self,
        itinerary_ids: Sequence[UUID],
        is_favorite: bool,
    ) -> int:
        """Favorite or unfavorite multiple itineraries."""
        result = await self.db.execute(
            update(Itinerary).where(Itinerary.id.in_(itinerary_ids)).values(is_favorite=is_favorite)
        )
        return result.rowcount
