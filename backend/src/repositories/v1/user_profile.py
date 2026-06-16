"""User profile repository."""

from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models import UserProfile
from repositories.v1.base import BaseRepository


class UserProfileRepository(BaseRepository[UserProfile]):
    """Repository for UserProfile model."""

    def __init__(self, db: AsyncSession):
        super().__init__(db, UserProfile)

    async def get_by_user_id(self, user_id: UUID) -> Optional[UserProfile]:
        """Get profile by user ID."""
        result = await self.db.execute(select(UserProfile).where(UserProfile.user_id == user_id))
        return result.scalar_one_or_none()

    async def create_or_update(
        self,
        user_id: UUID,
        *,
        personal: Optional[dict] = None,
        preferences: Optional[dict] = None,
        frequent_destinations: Optional[list] = None,
    ) -> UserProfile:
        """Create or update a user profile."""
        profile = await self.get_by_user_id(user_id)
        if profile is None:
            profile = UserProfile(
                user_id=user_id,
                personal=personal or {},
                preferences=preferences or {},
                frequent_destinations=frequent_destinations or [],
            )
            profile = await self.create(profile)
        else:
            if personal is not None:
                profile.personal = personal
            if preferences is not None:
                profile.preferences = preferences
            if frequent_destinations is not None:
                profile.frequent_destinations = frequent_destinations
            await self.db.flush()
            await self.db.refresh(profile)
        return profile
