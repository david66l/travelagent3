"""Memory Management Agent - async persistence of conversations and itineraries."""

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from schemas import UserMemory, DayPlan, UserProfile
from skills.memory_store import MemoryStoreSkill
from skills.memory_retrieve import MemoryRetrieveSkill


class MemoryManagementAgent:
    """Manage all user-related persistent data."""

    @staticmethod
    async def save_conversation_turn(
        db: AsyncSession,
        session_id: str,
        user_msg: str,
        assistant_msg: str,
        intent: Optional[str] = None,
        tools: Optional[list[str]] = None,
    ) -> None:
        """Save a conversation turn (called after each interaction)."""
        await MemoryStoreSkill.save_conversation(
            db=db,
            session_id=session_id,
            user_message=user_msg,
            assistant_response=assistant_msg,
            intent=intent,
            tools_used=tools,
        )

    @staticmethod
    async def save_itinerary(
        db: AsyncSession,
        session_id: str,
        user_id: str,
        itinerary: list[DayPlan],
        profile: UserProfile,
        budget: dict,
        confirmed: bool = False,
    ) -> str:
        """Save an itinerary (called when user confirms)."""
        return await MemoryStoreSkill.save_itinerary(
            db=db,
            session_id=session_id,
            user_id=user_id,
            destination=profile.destination or "",
            travel_days=len(itinerary),
            daily_plans=[day.model_dump() for day in itinerary],
            preference_snapshot=profile.model_dump(),
            budget_snapshot=budget,
            status="confirmed" if confirmed else "draft",
        )

    @staticmethod
    async def get_user_memory(
        db: AsyncSession,
        user_id: str,
        session_id: str,
    ) -> UserMemory:
        """Get complete user memory for a new session."""
        return await MemoryRetrieveSkill.get_user_memory(db, user_id, session_id)

    @staticmethod
    async def save_preference_change(
        db: AsyncSession,
        session_id: str,
        user_id: str,
        change: dict,
        source_message: Optional[str] = None,
    ) -> None:
        """Record a preference change."""
        await MemoryStoreSkill.save_preference_change(
            db=db,
            session_id=session_id,
            user_id=user_id,
            field=change.get("field", ""),
            old_value=change.get("old_value"),
            new_value=change.get("new_value", ""),
            source_message=source_message,
        )
