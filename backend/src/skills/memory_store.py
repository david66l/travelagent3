"""Memory Store Skill - persist data to database."""

from sqlalchemy.ext.asyncio import AsyncSession

from models import Conversation, Itinerary, PreferenceChange


class MemoryStoreSkill:
    """Store conversation, itinerary, and preference data."""

    @staticmethod
    async def save_conversation(
        db: AsyncSession,
        session_id: str,
        user_message: str,
        assistant_response: str,
        intent: str | None = None,
        tools_used: list[str] | None = None,
    ) -> None:
        """Save a conversation turn."""
        conv = Conversation(
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            intent=intent,
            tools_used=tools_used or [],
        )
        db.add(conv)
        await db.commit()

    @staticmethod
    async def save_itinerary(
        db: AsyncSession,
        session_id: str,
        user_id: str,
        destination: str,
        travel_days: int,
        daily_plans: list[dict],
        preference_snapshot: dict,
        budget_snapshot: dict,
        status: str = "draft",
    ) -> str:
        """Save an itinerary and return its ID."""
        itinerary = Itinerary(
            session_id=session_id,
            user_id=user_id,
            destination=destination,
            travel_days=travel_days,
            daily_plans=daily_plans,
            preference_snapshot=preference_snapshot,
            budget_snapshot=budget_snapshot,
            status=status,
        )
        db.add(itinerary)
        await db.commit()
        await db.refresh(itinerary)
        return str(itinerary.id)

    @staticmethod
    async def save_preference_change(
        db: AsyncSession,
        session_id: str,
        user_id: str,
        field: str,
        old_value: str | None,
        new_value: str,
        source_message: str | None = None,
    ) -> None:
        """Save a preference change record."""
        change = PreferenceChange(
            session_id=session_id,
            user_id=user_id,
            field=field,
            old_value=old_value,
            new_value=new_value,
            source_message=source_message,
        )
        db.add(change)
        await db.commit()
