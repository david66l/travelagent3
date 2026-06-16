"""Authentication endpoints for v1 API."""

from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials

from api.deps import get_user_service, security
from api.v1.schemas import (
    GuestTokenRequest,
    LoginRequest,
    RefreshRequest,
    RegisterRequest,
    TokenResponse,
    UpgradeRequest,
)
from core.exceptions import ForbiddenException, UnauthorizedException, ValidationException
from core.responses import success_response
from core.security import blacklist_token, create_access_token, decode_token
from core.settings import settings
from models import User
from services.user_service import UserService

router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response(
    user: User,
    access_token: str,
    refresh_token: str | None = None,
    expires_in: int | None = None,
) -> TokenResponse:
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=expires_in,
        role=user.role,
    )


@router.post("/guest")
async def create_guest_token(
    body: GuestTokenRequest,
    service: UserService = Depends(get_user_service),
):
    """Issue a guest token bound to a device fingerprint."""
    user, token = await service.create_guest_token(body.device_fingerprint)
    data = _token_response(
        user,
        token,
        expires_in=settings.guest_token_expire_hours * 3600,
    )
    return success_response(data=data.model_dump())


@router.post("/register")
async def register(
    body: RegisterRequest,
    service: UserService = Depends(get_user_service),
):
    """Register a new user account."""
    user = await service.register_user(
        email=body.email,
        phone=body.phone,
        password=body.password,
    )
    tokens = service.create_token_pair(user)
    data = _token_response(
        user,
        tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=settings.access_token_expire_minutes * 60,
    )
    return success_response(data=data.model_dump(), status_code=201)


@router.post("/login")
async def login(
    body: LoginRequest,
    service: UserService = Depends(get_user_service),
):
    """Authenticate and receive access/refresh tokens."""
    if not body.email and not body.phone:
        raise ValidationException("Email or phone is required")
    user = await service.authenticate_user(
        email=body.email,
        phone=body.phone,
        password=body.password,
    )
    tokens = service.create_token_pair(user)
    data = _token_response(
        user,
        tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=settings.access_token_expire_minutes * 60,
    )
    return success_response(data=data.model_dump())


@router.post("/refresh")
async def refresh(
    body: RefreshRequest,
    service: UserService = Depends(get_user_service),
):
    """Exchange a valid refresh token for a new access token."""
    payload = decode_token(body.refresh_token)
    if payload.get("type") != "refresh":
        raise UnauthorizedException("Invalid refresh token")

    user_id = payload.get("sub")
    user = await service.get_user(user_id)
    if user is None:
        raise UnauthorizedException("User not found")

    access_token = create_access_token(user.id, user.role)
    data = _token_response(
        user,
        access_token,
        expires_in=settings.access_token_expire_minutes * 60,
    )
    return success_response(data=data.model_dump())


@router.post("/upgrade")
async def upgrade(
    body: UpgradeRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    service: UserService = Depends(get_user_service),
):
    """Upgrade a guest account to a registered user."""
    if not credentials or not credentials.credentials:
        raise UnauthorizedException("Authentication required")

    payload = decode_token(credentials.credentials)
    current_user = await service.get_user(payload.get("sub"))
    if current_user is None:
        raise UnauthorizedException("User not found")
    if current_user.role != "guest":
        raise ForbiddenException("Only guest accounts can be upgraded")

    user = await service.upgrade_guest_to_user(
        current_user.id,
        email=body.email,
        phone=body.phone,
        password=body.password,
    )

    # Invalidate the old guest token and issue a real token pair.
    await blacklist_token(credentials.credentials)
    tokens = service.create_token_pair(user)
    data = _token_response(
        user,
        tokens["access_token"],
        refresh_token=tokens["refresh_token"],
        expires_in=settings.access_token_expire_minutes * 60,
    )
    return success_response(data=data.model_dump())


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    service: UserService = Depends(get_user_service),
):
    """Revoke the current access token."""
    if not credentials or not credentials.credentials:
        raise UnauthorizedException("Authentication required")
    await service.logout(credentials.credentials)
    return success_response(message="Logged out successfully")
