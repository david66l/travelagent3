"""Message repository."""

from typing import Optional, Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import Message
from repositories.v1.base import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """Repository for Message model."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, Message)

    async def get_by_conversation(
        self,
        conversation_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Message]:
        """Get messages for a conversation."""
        result = await self.db.execute(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def create_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        *,
        token_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> Message:
        """Create a new message."""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            token_count=token_count,
            metadata=metadata or {},
        )
        return await self.create(message)

    async def get_total_token_count(self, conversation_id: UUID) -> int:
        """Get total token count for a conversation."""
        from sqlalchemy import func

        result = await self.db.execute(
            select(func.coalesce(func.sum(Message.token_count), 0)).where(
                Message.conversation_id == conversation_id
            )
        )
        return result.scalar_one()
