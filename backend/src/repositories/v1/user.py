"""User repository."""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import ConflictException
from models import User
from repositories.v1.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """Repository for User model."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, User)

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        result = await self.db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    async def get_by_phone(self, phone: str) -> Optional[User]:
        """Get user by phone."""
        result = await self.db.execute(select(User).where(User.phone == phone))
        return result.scalar_one_or_none()

    async def create_user(
        self,
        *,
        email: Optional[str] = None,
        phone: Optional[str] = None,
        password_hash: Optional[str] = None,
        role: str = "user",
    ) -> User:
        """Create a new user."""
        user = User(
            email=email,
            phone=phone,
            password_hash=password_hash,
            role=role,
        )
        return await self.create(user)

    async def get_or_create_guest(self, guest_id: str) -> User:
        """Get existing guest user or create a new one.

        guest_id is used as a stable identifier for anonymous users.
        """
        result = await self.db.execute(select(User).where(User.email == f"guest_{guest_id}@local"))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(
                email=f"guest_{guest_id}@local",
                role="guest",
            )
            user = await self.create(user)
        return user

    async def upgrade_guest(
        self,
        user_id,
        *,
        email: str,
        phone: str | None,
        password_hash: str,
    ) -> User:
        """Convert a guest user into a registered user.

        Raises:
            ConflictException: if the email or phone is already taken by
                another user.
        """
        # Load the target user first to ensure it exists and is a guest.
        user = await self.get_by_id(user_id)
        if user is None:
            raise ConflictException("Guest user not found")
        if user.role != "guest":
            raise ConflictException("User is already registered")

        # Check uniqueness against other users.
        if email:
            existing = await self.get_by_email(email)
            if existing is not None and existing.id != user.id:
                raise ConflictException("Email already registered")
        if phone:
            existing = await self.get_by_phone(phone)
            if existing is not None and existing.id != user.id:
                raise ConflictException("Phone already registered")

        user.email = email
        user.phone = phone
        user.password_hash = password_hash
        user.role = "user"
        await self.db.flush()
        await self.db.refresh(user)
        return user
