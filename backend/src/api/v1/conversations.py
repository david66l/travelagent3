"""Conversation v1 API endpoints."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Query

from api.deps import get_conversation_service, get_current_user
from api.v1.schemas import (
    ConversationResponse,
    CreateConversationRequest,
    CreateMessageRequest,
    MessageResponse,
)
from core.exceptions import NotFoundException
from core.responses import success_response
from models import User
from services import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.get("")
async def list_conversations(
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """List conversations for the current user."""
    conversations = await service.list_by_user(user.id, status=status, limit=limit, offset=offset)
    return success_response(
        data=[ConversationResponse.model_validate(c).model_dump() for c in conversations],
        meta={"limit": limit, "offset": offset},
    )


@router.post("")
async def create_conversation(
    body: CreateConversationRequest,
    user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """Create a new conversation."""
    conversation = await service.create(
        user.id,
        title=body.title,
        state_snapshot=body.state_snapshot,
    )
    return success_response(
        data=ConversationResponse.model_validate(conversation).model_dump(),
        status_code=201,
    )


@router.get("/{conversation_id}")
async def get_conversation(
    conversation_id: UUID,
    user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """Get a single conversation."""
    conversation = await service.get(conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise NotFoundException("Conversation", conversation_id)
    return success_response(data=ConversationResponse.model_validate(conversation).model_dump())


@router.post("/{conversation_id}/archive")
async def archive_conversation(
    conversation_id: UUID,
    user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """Archive a conversation."""
    conversation = await service.get(conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise NotFoundException("Conversation", conversation_id)
    await service.archive(conversation_id)
    return success_response(message="Conversation archived")


@router.get("/{conversation_id}/messages")
async def list_messages(
    conversation_id: UUID,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """List messages in a conversation."""
    conversation = await service.get(conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise NotFoundException("Conversation", conversation_id)
    messages = await service.get_messages(conversation_id, limit=limit, offset=offset)
    return success_response(
        data=[MessageResponse.model_validate(m).model_dump() for m in messages],
        meta={"limit": limit, "offset": offset},
    )


@router.post("/{conversation_id}/messages")
async def create_message(
    conversation_id: UUID,
    body: CreateMessageRequest,
    user: User = Depends(get_current_user),
    service: ConversationService = Depends(get_conversation_service),
):
    """Add a message to a conversation."""
    conversation = await service.get(conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise NotFoundException("Conversation", conversation_id)
    message = await service.add_message(
        conversation_id,
        body.role,
        body.content,
        token_count=body.token_count,
        metadata=body.metadata,
    )
    return success_response(
        data=MessageResponse.model_validate(message).model_dump(),
        status_code=201,
    )
