"""Memory Retrieve Skill - retrieve data from database."""

from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from models import Conversation, Itinerary, PreferenceChange
from schemas import UserMemory


class MemoryRetrieveSkill:
    """Retrieve conversation, itinerary, and preference history."""

    @staticmethod
    async def get_recent_conversations(
        db: AsyncSession,
        session_id: str,
        limit: int = 50,
    ) -> list[dict]:
        """Get recent conversations for a session."""
        result = await db.execute(
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(desc(Conversation.created_at))
            .limit(limit)
        )
        conversations = result.scalars().all()
        return [
            {
                "user_message": c.user_message,
                "assistant_response": c.assistant_response,
                "intent": c.intent,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in conversations
        ]

    @staticmethod
    async def get_recent_itineraries(
        db: AsyncSession,
        user_id: str,
        limit: int = 3,
    ) -> list[dict]:
        """Get recent confirmed itineraries for a user."""
        result = await db.execute(
            select(Itinerary)
            .where(
                Itinerary.user_id == user_id,
                Itinerary.status == "confirmed",
            )
            .order_by(desc(Itinerary.created_at))
            .limit(limit)
        )
        itineraries = result.scalars().all()
        return [
            {
                "id": str(i.id),
                "destination": i.destination,
                "travel_days": i.travel_days,
                "daily_plans": i.daily_plans,
                "preference_snapshot": i.preference_snapshot,
                "created_at": i.created_at.isoformat() if i.created_at else None,
            }
            for i in itineraries
        ]

    @staticmethod
    async def get_preference_changes(
        db: AsyncSession,
        session_id: str,
    ) -> list[dict]:
        """Get preference change history for a session."""
        result = await db.execute(
            select(PreferenceChange)
            .where(PreferenceChange.session_id == session_id)
            .order_by(desc(PreferenceChange.created_at))
        )
        changes = result.scalars().all()
        return [
            {
                "field": c.field,
                "old_value": c.old_value,
                "new_value": c.new_value,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in changes
        ]

    @staticmethod
    async def get_user_memory(
        db: AsyncSession,
        user_id: str,
        session_id: str,
    ) -> UserMemory:
        """Get complete user memory (itineraries + patterns + conversations)."""
        recent_itineraries = await MemoryRetrieveSkill.get_recent_itineraries(db, user_id, limit=3)
        recent_conversations = await MemoryRetrieveSkill.get_recent_conversations(
            db, session_id, limit=50
        )

        # Extract preference patterns from itineraries
        patterns = MemoryRetrieveSkill._extract_patterns(recent_itineraries)

        return UserMemory(
            recent_itineraries=recent_itineraries,
            preference_patterns=patterns,
            recent_conversations=recent_conversations,
        )

    @staticmethod
    def _extract_patterns(itineraries: list[dict]) -> dict:
        """Extract preference patterns from historical itineraries."""
        if not itineraries:
            return {}

        pace_counts = {}
        food_set = set()
        total_budget = 0
        cities = []

        for it in itineraries:
            pref = it.get("preference_snapshot", {})
            pace = pref.get("pace", "moderate")
            pace_counts[pace] = pace_counts.get(pace, 0) + 1

            food = pref.get("food_preferences", [])
            food_set.update(food)

            budget = pref.get("budget_range")
            if budget:
                total_budget += budget

            cities.append(it.get("destination", ""))

        preferred_pace = max(pace_counts, key=pace_counts.get) if pace_counts else "moderate"
        avg_budget = total_budget / len(itineraries) if itineraries else 0

        return {
            "preferred_pace": preferred_pace,
            "preferred_food": list(food_set),
            "avg_budget": avg_budget,
            "favorite_cities": list(set(cities)),
            "trip_count": len(itineraries),
        }
