"""Itinerary application service."""

from typing import Optional, Sequence
from uuid import UUID

from models import Itinerary
from repositories.v1 import ConversationRepository, ItineraryRepository
from services.base import BaseService


class ItineraryService(BaseService):
    """Service for itinerary creation and queries."""

    def __init__(
        self,
        repo: ItineraryRepository,
        conversation_repo: ConversationRepository,
    ):
        self.repo = repo
        self.conversation_repo = conversation_repo

    async def create(
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
        """Create an itinerary linked to a user and conversation."""
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        return await self.repo.create_itinerary(
            user_id=user_id,
            conversation_id=conversation_id,
            destination=destination,
            days=days,
            content=content,
            proposal_text=proposal_text,
            job_id=job_id,
        )

    async def list_by_user(
        self,
        user_id: UUID,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> Sequence[Itinerary]:
        """List itineraries for a user."""
        return await self.repo.get_by_user(user_id, limit=limit, offset=offset)

    async def list_by_conversation(
        self,
        conversation_id: UUID,
    ) -> Sequence[Itinerary]:
        """List itineraries for a conversation."""
        return await self.repo.get_by_conversation(conversation_id)

    async def set_favorite(self, itinerary_id: UUID, is_favorite: bool) -> bool:
        """Toggle favorite status for an itinerary."""
        rowcount = await self.repo.set_favorite(itinerary_id, is_favorite)
        return rowcount > 0
