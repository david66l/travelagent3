"""Profile recall agent: fill missing slots from short-term and long-term memory."""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.database import async_session_maker
from core.redis_client import redis_client
from data.profile_service import ProfileService, profile_service
from models.travel_slots import TravelSlots
from schemas import UserProfile

logger = logging.getLogger(__name__)


def _profile_to_dict(profile: UserProfile) -> dict[str, Any]:
    """Serialize UserProfile to a flat dict for conflict resolution."""
    return profile.model_dump(exclude_none=True)


def _slots_from_profile_dict(profile: dict[str, Any]) -> dict[str, Any]:
    """Map profile dict fields to TravelSlots field names."""
    mapping: dict[str, Any] = {}
    budget = profile.get("budget_range") or profile.get("avg_daily_budget")
    if budget is not None:
        mapping["total_budget"] = float(budget)
    if profile.get("travelers_type") is not None:
        mapping["travel_companion"] = profile["travelers_type"]
    if profile.get("food_preferences") is not None:
        mapping["food_prefs"] = list(profile["food_preferences"])
    if profile.get("food_taboos") is not None:
        mapping["food_taboos"] = list(profile["food_taboos"])
    if profile.get("interests") is not None:
        mapping["interests"] = list(profile["interests"])
    if profile.get("pace") is not None:
        mapping["pace"] = profile["pace"]
    if profile.get("transport_mode") is not None:
        mapping["transport_preference"] = profile["transport_mode"]
    if profile.get("max_walk_minutes") is not None:
        mapping["max_walk_minutes"] = profile["max_walk_minutes"]
    if profile.get("max_transit_minutes") is not None:
        mapping["max_transit_minutes"] = profile["max_transit_minutes"]
    if profile.get("has_elderly") is not None:
        mapping["has_elderly"] = bool(profile["has_elderly"])
    if profile.get("has_children") is not None:
        mapping["has_children"] = bool(profile["has_children"])
    if profile.get("has_pregnant") is not None:
        mapping["has_pregnant"] = bool(profile["has_pregnant"])
    if profile.get("has_wheelchair") is not None:
        mapping["has_wheelchair"] = bool(profile["has_wheelchair"])
    if profile.get("avoid_crowds") is not None:
        mapping["avoid_crowds"] = bool(profile["avoid_crowds"])
    if profile.get("prefer_morning") is not None:
        mapping["prefer_morning"] = bool(profile["prefer_morning"])
    return mapping


class ProfileRecallAgent:
    """Recall user profile from Redis short-term memory and pgvector long-term memory."""

    def __init__(
        self,
        profile_service_instance: Optional[ProfileService] = None,
        redis_client_instance: Any = None,
    ):
        self._profile_service = profile_service_instance or profile_service
        self._redis = redis_client_instance or redis_client

    async def recall(
        self,
        user_id: Optional[str],
        current_slots: TravelSlots,
        *,
        session_id: Optional[str] = None,
        short_term_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Return recalled profile layers and inferred slots.

        Returns dict with keys:
        - source: "anonymous" | "short_term" | "long_term" | "mixed"
        - short_term_profile: UserProfile from Redis/session state
        - long_term_profile: UserProfile from pgvector / structured storage
        - merged_profile: UserProfile after merging short + long + similar users
        - recalled_profile: alias of merged_profile (backward compatibility)
        - inferred_slots: dict of slot name -> value
        - confidence: float
        """
        # Anonymous users: no recall.
        if not user_id:
            empty = UserProfile()
            return {
                "source": "anonymous",
                "short_term_profile": empty,
                "long_term_profile": empty,
                "merged_profile": empty,
                "recalled_profile": empty,
                "inferred_slots": {},
                "confidence": 0.0,
            }

        # 1. Short-term memory from Redis/session state
        short_term_dict: dict[str, Any] = {}
        if short_term_state:
            profile_state = short_term_state.get("profile", {})
            if profile_state:
                flat = self._flatten_nested_profile(profile_state)
                short_term_dict.update(_slots_from_profile_dict(flat))

        if session_id:
            try:
                redis_state = await self._redis.get_json(f"session:{session_id}:state")
                if redis_state:
                    flat = self._flatten_nested_profile(redis_state.get("profile", {}))
                    short_term_dict.update(_slots_from_profile_dict(flat))
            except Exception as exc:
                logger.warning("Failed to read Redis short-term memory for %s: %s", session_id, exc)

        short_term_profile = UserProfile(**self._profile_dict_from_slots(short_term_dict))

        # 2. Long-term structured profile
        long_term_dict: dict[str, Any] = {}
        try:
            long_term = await self._profile_service.get_profile(user_id)
            if long_term:
                long_term_dict.update(_slots_from_profile_dict(long_term))
        except Exception as exc:
            logger.warning("Failed to recall long-term profile for %s: %s", user_id, exc)

        # 3. pgvector similar-user preference recall
        similar_dict: dict[str, Any] = {}
        try:
            similar_profiles = await self._recall_similar_users(user_id)
            for profile in similar_profiles:
                similar_dict.update(_slots_from_profile_dict(profile))
        except Exception as exc:
            logger.warning("Failed to recall similar users for %s: %s", user_id, exc)

        # Merge long-term + similar users (similar users only fill gaps)
        merged_long = {**similar_dict, **long_term_dict}
        long_term_profile = UserProfile(**self._profile_dict_from_slots(merged_long))

        # 4. Merge short-term + long-term (short-term wins)
        merged = self._merge_layer(short_term_dict, merged_long)
        merged_profile = UserProfile(**self._profile_dict_from_slots(merged))

        # 5. Infer missing slots from merged profile
        inferred = self._extract_inferred(merged_profile, current_slots)

        source = self._determine_source(
            bool(short_term_dict),
            bool(long_term_dict or similar_dict),
            attempted_long_term=bool(user_id),
        )
        confidence = 0.75 if (long_term_dict or similar_dict) else 0.6 if short_term_dict else 0.25

        return {
            "source": source,
            "short_term_profile": short_term_profile,
            "long_term_profile": long_term_profile,
            "merged_profile": merged_profile,
            "recalled_profile": merged_profile,
            "inferred_slots": inferred,
            "confidence": confidence,
        }

    @staticmethod
    def _flatten_nested_profile(profile: dict[str, Any]) -> dict[str, Any]:
        """Flatten personal + trip nested profile dict."""
        personal = profile.get("personal", {})
        trip = profile.get("trip", {})
        return {**personal, **trip}

    @staticmethod
    def _profile_dict_from_slots(slots_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert slot-oriented dict back to UserProfile-oriented dict."""
        result: dict[str, Any] = {}
        if slots_dict.get("total_budget") is not None:
            result["budget_range"] = slots_dict["total_budget"]
        if slots_dict.get("travel_companion") is not None:
            result["travelers_type"] = slots_dict["travel_companion"]
        if slots_dict.get("food_prefs") is not None:
            result["food_preferences"] = list(slots_dict["food_prefs"])
        if slots_dict.get("food_taboos") is not None:
            result["food_taboos"] = list(slots_dict["food_taboos"])
        if slots_dict.get("interests") is not None:
            result["interests"] = list(slots_dict["interests"])
        if slots_dict.get("pace") is not None:
            result["pace"] = slots_dict["pace"]
        if slots_dict.get("transport_preference") is not None:
            result["transport_mode"] = slots_dict["transport_preference"]
        for key in (
            "destination",
            "travel_days",
            "travel_dates",
            "travelers_count",
            "max_walk_minutes",
            "max_transit_minutes",
            "has_elderly",
            "has_children",
            "has_pregnant",
            "has_wheelchair",
            "avoid_crowds",
            "prefer_morning",
        ):
            if slots_dict.get(key) is not None:
                result[key] = slots_dict[key]
        return result

    @staticmethod
    def _merge_layer(high_priority: dict[str, Any], low_priority: dict[str, Any]) -> dict[str, Any]:
        """Merge two slot-oriented dicts; high_priority wins on conflicts."""
        merged = dict(low_priority)
        for key, value in high_priority.items():
            if value is not None and value != []:
                merged[key] = value
        return merged

    def _extract_inferred(
        self,
        merged_profile: UserProfile,
        current_slots: TravelSlots,
    ) -> dict[str, Any]:
        """Infer missing slots from merged profile without overriding existing ones."""
        inferred: dict[str, Any] = {}
        profile_dict = _profile_to_dict(merged_profile)
        defaults = UserProfile().model_dump()

        def _is_memory_value(key: str, value: Any) -> bool:
            """True if the value differs from the UserProfile default (i.e. it is actual memory)."""
            if value is None:
                return False
            default_value = defaults.get(key)
            if value == default_value:
                return False
            return True

        if current_slots.total_budget is None and _is_memory_value(
            "budget_range", profile_dict.get("budget_range")
        ):
            days = current_slots.travel_days or profile_dict.get("travel_days") or 3
            inferred["total_budget"] = float(profile_dict["budget_range"]) * days

        if current_slots.pace is None and _is_memory_value("pace", profile_dict.get("pace")):
            inferred["pace"] = profile_dict["pace"]

        if not current_slots.interests and profile_dict.get("interests"):
            inferred["interests"] = list(profile_dict["interests"])[:5]

        if not current_slots.food_taboos and profile_dict.get("food_taboos"):
            inferred["food_taboos"] = list(profile_dict["food_taboos"])[:5]

        if not current_slots.food_prefs and profile_dict.get("food_preferences"):
            inferred["food_prefs"] = list(profile_dict["food_preferences"])[:5]

        if (
            current_slots.transport_preference is None
            and _is_memory_value("transport_mode", profile_dict.get("transport_mode"))
            and profile_dict["transport_mode"] != "any"
        ):
            inferred["transport_preference"] = profile_dict["transport_mode"]

        if current_slots.max_walk_minutes is None and _is_memory_value(
            "max_walk_minutes", profile_dict.get("max_walk_minutes")
        ):
            inferred["max_walk_minutes"] = profile_dict["max_walk_minutes"]

        if current_slots.max_transit_minutes is None and _is_memory_value(
            "max_transit_minutes", profile_dict.get("max_transit_minutes")
        ):
            inferred["max_transit_minutes"] = profile_dict["max_transit_minutes"]

        for key in (
            "has_elderly",
            "has_children",
            "has_pregnant",
            "has_wheelchair",
            "avoid_crowds",
            "prefer_morning",
        ):
            current_value = getattr(current_slots, key)
            profile_value = profile_dict.get(key)
            # Only infer positive flags; False is the default and should not be treated as memory.
            if current_value is None and profile_value is True:
                inferred[key] = True

        return inferred

    @staticmethod
    def _determine_source(
        has_short: bool, has_long: bool, attempted_long_term: bool = False
    ) -> str:
        if has_short and has_long:
            return "mixed"
        if has_short:
            return "short_term"
        if has_long:
            return "long_term"
        if attempted_long_term:
            return "long_term"
        return "anonymous"

    async def _recall_similar_users(self, user_id: str) -> list[dict[str, Any]]:
        """Use pgvector to find users with similar preference embeddings.

        Supports both asyncpg-style connections and SQLAlchemy AsyncSession.
        """
        async with async_session_maker() as db:
            if hasattr(db, "fetchrow"):
                return await self._recall_similar_users_asyncpg(db, user_id)
            return await self._recall_similar_users_sqlalchemy(db, user_id)

    @staticmethod
    async def _recall_similar_users_asyncpg(db, user_id: str) -> list[dict[str, Any]]:
        row = await db.fetchrow(
            "SELECT preference_embedding FROM user_profile_vectors WHERE user_id = $1",
            user_id,
        )
        if not row or not row["preference_embedding"]:
            return []

        similar_rows = await db.fetch(
            """
            SELECT user_id, profile_json, visited_cities, favorite_spots,
                   liked_foods, avoided_foods, avg_daily_budget,
                   preferred_transport, preferred_accommodation,
                   preference_embedding <=> $1::vector AS distance
            FROM user_profile_vectors
            WHERE user_id != $2 AND preference_embedding IS NOT NULL
            ORDER BY preference_embedding <=> $1::vector
            LIMIT 3
            """,
            row["preference_embedding"],
            user_id,
        )
        return [_row_to_profile_dict(row) for row in similar_rows]

    @staticmethod
    async def _recall_similar_users_sqlalchemy(db, user_id: str) -> list[dict[str, Any]]:
        from sqlalchemy import text

        result = await db.execute(
            text("SELECT preference_embedding FROM user_profile_vectors WHERE user_id = :uid"),
            {"uid": user_id},
        )
        row = result.fetchone()
        if not row or not row[0]:
            return []
        embedding = row[0]

        result = await db.execute(
            text(
                """
                SELECT user_id, profile_json, visited_cities, favorite_spots,
                       liked_foods, avoided_foods, avg_daily_budget,
                       preferred_transport, preferred_accommodation,
                       preference_embedding <=> :embedding::vector AS distance
                FROM user_profile_vectors
                WHERE user_id != :uid AND preference_embedding IS NOT NULL
                ORDER BY preference_embedding <=> :embedding::vector
                LIMIT 3
                """
            ),
            {"embedding": embedding, "uid": user_id},
        )
        return [_row_to_profile_dict(row) for row in result.fetchall()]


def _row_to_profile_dict(row) -> dict[str, Any]:
    """Convert a pgvector result row to a profile dict."""
    if hasattr(row, "_mapping"):
        row = dict(row._mapping)
    profile = dict(row.get("profile_json") or {})
    profile["visited_cities"] = row.get("visited_cities") or []
    profile["favorite_spots"] = row.get("favorite_spots") or []
    profile["liked_foods"] = row.get("liked_foods") or []
    profile["avoided_foods"] = row.get("avoided_foods") or []
    profile["avg_daily_budget"] = (
        float(row["avg_daily_budget"]) if row.get("avg_daily_budget") else None
    )
    profile["preferred_transport"] = row.get("preferred_transport") or ""
    profile["preferred_accommodation"] = row.get("preferred_accommodation") or ""
    return profile
