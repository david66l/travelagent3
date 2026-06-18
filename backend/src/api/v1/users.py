"""User v1 API endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends

from api.deps import get_current_user, get_user_service, require_admin
from api.v1.schemas import (
    UserProfileRequest,
    UserProfileResponse,
    UserResponse,
)
from core.exceptions import ForbiddenException, NotFoundException
from core.responses import success_response
from models import User
from services import UserService

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me")
async def get_me(
    user: User = Depends(get_current_user),
):
    """Return the current user."""
    return success_response(data=UserResponse.model_validate(user).model_dump())


@router.get("/me/profile")
async def get_my_profile(
    user: User = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
):
    """Return the current user's profile."""
    profile = await service.get_profile(user.id)
    if profile is None:
        return success_response(data=None)
    return success_response(data=UserProfileResponse.model_validate(profile).model_dump())


@router.put("/me/profile")
async def update_my_profile(
    body: UserProfileRequest,
    user: User = Depends(get_current_user),
    service: UserService = Depends(get_user_service),
):
    """Create or update the current user's profile."""
    profile = await service.update_profile(
        user.id,
        personal=body.personal,
        preferences=body.preferences,
        frequent_destinations=body.frequent_destinations,
    )
    return success_response(data=UserProfileResponse.model_validate(profile).model_dump())


@router.get("/{user_id}")
async def get_user(
    user_id: UUID,
    service: UserService = Depends(get_user_service),
):
    """Get a user by ID (placeholder, will be restricted in P3)."""
    user = await service.get_user(user_id)
    if user is None:
        raise NotFoundException("User", user_id)
    return success_response(data=UserResponse.model_validate(user).model_dump())


@router.post("/{user_id}/ban")
async def ban_user_account(
    user_id: UUID,
    admin: User = Depends(require_admin),
    service: UserService = Depends(get_user_service),
):
    """Revoke all tokens for a user (admin)."""
    if user_id == admin.id:
        raise ForbiddenException("Cannot ban yourself", code="BAN_SELF")
    user = await service.get_user(user_id)
    if user is None:
        raise NotFoundException("User", user_id)
    await service.ban_user(user_id)
    return success_response(message="User tokens revoked")
