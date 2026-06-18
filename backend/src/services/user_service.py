"""User application service."""

from typing import Optional
from uuid import UUID

from core.exceptions import ConflictException, UnauthorizedException
from core.security import (
    blacklist_token,
    create_access_token,
    create_guest_token,
    create_refresh_token,
    hash_password,
    verify_password,
)
from models import User
from repositories.v1 import UserProfileRepository, UserRepository
from services.base import BaseService


class UserService(BaseService):
    """Service for user/profile related business logic."""

    def __init__(
        self,
        user_repo: UserRepository,
        profile_repo: UserProfileRepository,
    ):
        self.user_repo = user_repo
        self.profile_repo = profile_repo

    async def get_or_create_guest(self, guest_id: str) -> User:
        """Resolve or provision a guest user."""
        return await self.user_repo.get_or_create_guest(guest_id)

    async def get_user(self, user_id: UUID) -> Optional[User]:
        """Fetch a user by primary key."""
        return await self.user_repo.get_by_id(user_id)

    async def update_profile(
        self,
        user_id: UUID,
        *,
        personal: Optional[dict] = None,
        preferences: Optional[dict] = None,
        frequent_destinations: Optional[list] = None,
    ):
        """Create or update the profile for a user."""
        return await self.profile_repo.create_or_update(
            user_id,
            personal=personal,
            preferences=preferences,
            frequent_destinations=frequent_destinations,
        )

    async def get_profile(self, user_id: UUID):
        """Return the profile for a user."""
        return await self.profile_repo.get_by_user_id(user_id)

    async def create_guest_token(self, device_fingerprint: str) -> tuple[User, str]:
        """Provision a guest user and return a bound access token."""
        user = await self.user_repo.get_or_create_guest(device_fingerprint)
        token = create_guest_token(user.id, device_fingerprint)
        return user, token

    async def register_user(
        self,
        *,
        email: str,
        phone: Optional[str] = None,
        password: str,
    ) -> User:
        """Register a new user account."""
        if await self.user_repo.get_by_email(email) is not None:
            raise ConflictException("Email already registered")
        if phone and await self.user_repo.get_by_phone(phone) is not None:
            raise ConflictException("Phone already registered")

        return await self.user_repo.create_user(
            email=email,
            phone=phone,
            password_hash=hash_password(password),
            role="user",
        )

    async def authenticate_user(
        self,
        *,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        password: str,
    ) -> User:
        """Authenticate by email or phone and password."""
        if email:
            user = await self.user_repo.get_by_email(email)
        elif phone:
            user = await self.user_repo.get_by_phone(phone)
        else:
            raise UnauthorizedException("Email or phone required")

        if user is None or not user.password_hash:
            raise UnauthorizedException("Invalid credentials")
        if not verify_password(password, user.password_hash):
            raise UnauthorizedException("Invalid credentials")
        return user

    async def upgrade_guest_to_user(
        self,
        guest_user_id: UUID,
        *,
        email: str,
        phone: Optional[str] = None,
        password: str,
    ) -> User:
        """Upgrade an existing guest account to a registered user."""
        return await self.user_repo.upgrade_guest(
            guest_user_id,
            email=email,
            phone=phone,
            password_hash=hash_password(password),
        )

    def create_token_pair(self, user: User) -> dict[str, str]:
        """Create access + refresh tokens for a user."""
        return {
            "access_token": create_access_token(user.id, user.role),
            "refresh_token": create_refresh_token(user.id, user.role),
        }

    async def logout(self, token: str, refresh_token: str | None = None) -> None:
        """Revoke access and optional refresh tokens."""
        await blacklist_token(token)
        if refresh_token:
            try:
                await blacklist_token(refresh_token)
            except UnauthorizedException:
                pass

    async def ban_user(self, user_id: UUID) -> None:
        """Revoke all outstanding tokens for a user (admin action)."""
        from core.security import ban_user_tokens

        await ban_user_tokens(str(user_id))
