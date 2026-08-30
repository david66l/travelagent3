"""Conversation repository."""

from typing import Optional, Sequence
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.clock import utc_now_naive
from models import Conversation
from repositories.v1.base import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    """Repository for Conversation model."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, Conversation)

    async def get_by_user(
        self,
        user_id: UUID,
        *,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Sequence[Conversation]:
        """List conversations for a user."""
        stmt = (
            select(Conversation)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if status is not None:
            stmt = stmt.where(Conversation.status == status)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def create_conversation(
        self,
        user_id: UUID,
        *,
        title: Optional[str] = None,
        state_snapshot: Optional[dict] = None,
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(
            user_id=user_id,
            title=title,
            state_snapshot=state_snapshot or {},
        )
        return await self.create(conversation)

    async def archive(self, conversation_id: UUID) -> int:
        """Archive a conversation."""
        result = await self.db.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(status="archived", archived_at=utc_now_naive())
        )
        return result.rowcount

    async def batch_archive(self, conversation_ids: Sequence[UUID]) -> int:
        """Archive multiple conversations."""
        result = await self.db.execute(
            update(Conversation)
            .where(Conversation.id.in_(conversation_ids))
            .values(status="archived", archived_at=utc_now_naive())
        )
        return result.rowcount
