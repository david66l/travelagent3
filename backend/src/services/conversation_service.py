"""Conversation application service."""

from typing import Optional, Sequence
from uuid import UUID

from models import Conversation, Message
from repositories.v1 import ConversationRepository, MessageRepository
from services.base import BaseService


class ConversationService(BaseService):
    """Service for conversation lifecycle and query logic."""

    def __init__(
        self,
        repo: ConversationRepository,
        message_repo: MessageRepository,
    ):
        self.repo = repo
        self.message_repo = message_repo

    async def create(
        self,
        user_id: UUID,
        *,
        title: Optional[str] = None,
        state_snapshot: Optional[dict] = None,
    ) -> Conversation:
        """Create a conversation owned by user_id."""
        return await self.repo.create_conversation(
            user_id, title=title, state_snapshot=state_snapshot
        )

    async def list_by_user(
        self,
        user_id: UUID,
        *,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Sequence[Conversation]:
        """List conversations for a user."""
        return await self.repo.get_by_user(user_id, status=status, limit=limit, offset=offset)

    async def get(self, conversation_id: UUID) -> Optional[Conversation]:
        """Fetch a conversation by ID."""
        return await self.repo.get_by_id(conversation_id)

    async def archive(self, conversation_id: UUID) -> bool:
        """Archive a conversation."""
        rowcount = await self.repo.archive(conversation_id)
        return rowcount > 0

    async def get_messages(
        self,
        conversation_id: UUID,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Message]:
        """Return messages for a conversation."""
        return await self.message_repo.get_by_conversation(
            conversation_id, limit=limit, offset=offset
        )

    async def add_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        *,
        token_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> Message:
        """Append a message to a conversation."""
        return await self.message_repo.create_message(
            conversation_id,
            role,
            content,
            token_count=token_count,
            metadata=metadata,
        )
