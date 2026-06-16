"""Pydantic request/response schemas for v1 API."""

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, AliasChoices


class CreateConversationRequest(BaseModel):
    title: Optional[str] = None
    state_snapshot: Optional[dict] = None


class ConversationResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: Optional[str]
    status: str
    state_snapshot: Optional[dict]
    created_at: Any
    updated_at: Any

    model_config = ConfigDict(from_attributes=True)


class CreateMessageRequest(BaseModel):
    role: str = Field(..., pattern="^(user|assistant|system|tool)$")
    content: str
    token_count: int = 0
    metadata: Optional[dict] = None


class MessageResponse(BaseModel):
    id: UUID
    conversation_id: UUID
    role: str
    content: str
    token_count: int
    metadata: Optional[dict] = Field(
        default=None, validation_alias=AliasChoices("metadata_", "metadata")
    )
    created_at: Any

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class CreateItineraryRequest(BaseModel):
    conversation_id: UUID
    destination: str = Field(..., min_length=1)
    days: int = Field(..., ge=1, le=30)
    content: Optional[dict] = None
    proposal_text: Optional[str] = None
    job_id: Optional[str] = None


class ItineraryResponse(BaseModel):
    id: UUID
    user_id: UUID
    conversation_id: UUID
    destination: str
    days: int
    content: Optional[dict]
    proposal_text: Optional[str]
    is_favorite: bool
    created_at: Any

    model_config = ConfigDict(from_attributes=True)


class UpdateFavoriteRequest(BaseModel):
    is_favorite: bool


class CreatePlanningJobRequest(BaseModel):
    conversation_id: Optional[UUID] = None
    queue_name: str = "default"
    input_requirements: Optional[dict] = None


class ChatMessageRequest(BaseModel):
    conversation_id: UUID
    content: str = Field(..., min_length=1, max_length=8000)
    stream: bool = True


class PlanningJobResponse(BaseModel):
    id: str
    user_uuid: Optional[UUID]
    conversation_id: Optional[UUID]
    queue_name: str
    status: str
    input_requirements: Optional[dict]
    result: Optional[dict]
    token_usage: Optional[dict]
    latency_ms: Optional[int]
    created_at: Any
    updated_at: Any

    model_config = ConfigDict(from_attributes=True)


class UpdatePlanningJobRequest(BaseModel):
    status: str


class UserProfileRequest(BaseModel):
    personal: Optional[dict] = None
    preferences: Optional[dict] = None
    frequent_destinations: Optional[list] = None


class UserProfileResponse(BaseModel):
    user_id: UUID
    personal: Optional[dict]
    preferences: Optional[dict]
    frequent_destinations: Optional[list]
    updated_at: Any

    model_config = ConfigDict(from_attributes=True)


class GuestTokenRequest(BaseModel):
    device_fingerprint: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=1)
    phone: Optional[str] = None
    password: str = Field(..., min_length=6)


class LoginRequest(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    password: str = Field(..., min_length=1)

    model_config = ConfigDict(str_strip_whitespace=True)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)


class UpgradeRequest(BaseModel):
    email: str = Field(..., min_length=1)
    phone: Optional[str] = None
    password: str = Field(..., min_length=6)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: Optional[int] = None
    role: str


class UserResponse(BaseModel):
    id: UUID
    email: Optional[str]
    phone: Optional[str]
    role: str
    created_at: Any

    model_config = ConfigDict(from_attributes=True)
