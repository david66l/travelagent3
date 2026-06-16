"""Message application service."""

from typing import Optional, Sequence
from uuid import UUID

from models import Message
from repositories.v1 import ConversationRepository, MessageRepository
from services.base import BaseService


class MessageService(BaseService):
    """Service for message-level operations."""

    def __init__(
        self,
        repo: MessageRepository,
        conversation_repo: ConversationRepository,
    ):
        self.repo = repo
        self.conversation_repo = conversation_repo

    async def create(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        *,
        token_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> Message:
        """Create a message and ensure the conversation exists."""
        conversation = await self.conversation_repo.get_by_id(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversation {conversation_id} not found")
        return await self.repo.create_message(
            conversation_id,
            role,
            content,
            token_count=token_count,
            metadata=metadata,
        )

    async def list_by_conversation(
        self,
        conversation_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Message]:
        """List messages for a conversation."""
        return await self.repo.get_by_conversation(conversation_id, limit=limit, offset=offset)
