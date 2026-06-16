"""Single-path user turn handling — intent recognition + profile merge.

WebSocket uses this before creating a planning job; the pipeline reads the
persisted profile from ``job.user_feedback`` instead of re-running intent LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.intent_recognition import IntentRecognitionAgent
from core.conversation_state import (
    append_message,
    default_conversation_state,
    flatten_profile,
    merge_profile,
)
from schemas import IntentResult, ProfilePatch, UserProfile

logger = logging.getLogger(__name__)


def entities_to_patch(entities: dict[str, Any]) -> ProfilePatch:
    """Convert flat ``user_entities`` into a ``ProfilePatch``."""
    scalar_keys = {
        "destination",
        "travel_days",
        "travel_dates",
        "travelers_count",
        "travelers_type",
        "pace",
        "budget_range",
        "accommodation_preference",
    }
    list_keys = {"interests", "food_preferences", "avoid", "special_requests"}

    set_patch: dict[str, Any] = {}
    add_patch: dict[str, list] = {}
    for key, value in entities.items():
        if value is None:
            continue
        if key in scalar_keys:
            set_patch[key] = value
        elif key in list_keys and isinstance(value, list):
            add_patch[key] = value

    return ProfilePatch(set=set_patch, add=add_patch)


def feedback_from_input_requirements(requirements: dict[str, Any] | None) -> dict[str, Any]:
    """Convert REST ``input_requirements`` flat dict into WS-style ``user_feedback``."""
    if not requirements:
        return default_conversation_state()
    state = default_conversation_state()
    set_fields = {k: v for k, v in requirements.items() if v is not None}
    if set_fields:
        state["profile"] = merge_profile(state["profile"], ProfilePatch(set=set_fields))
    state["phase"] = "planning"
    return state


def effective_user_feedback(
    user_feedback: dict[str, Any] | None,
    input_requirements: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve planning input: prefer full conversation state, else API requirements."""
    fb = user_feedback or {}
    profile = fb.get("profile")
    if profile and flatten_profile(profile).get("destination"):
        return fb
    if input_requirements:
        return feedback_from_input_requirements(input_requirements)
    return fb


def input_requirements_from_feedback(user_feedback: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten ``user_feedback.profile`` for REST / PRD ``input_requirements``."""
    flat = flatten_profile((user_feedback or {}).get("profile", {}))
    return {k: v for k, v in flat.items() if v is not None and v != []}


def user_profile_from_feedback(user_feedback: dict[str, Any] | None) -> UserProfile:
    """Build ``UserProfile`` from persisted conversation state (job.user_feedback)."""
    flat = flatten_profile((user_feedback or {}).get("profile", {}))
    return UserProfile(
        destination=flat.get("destination"),
        travel_days=flat.get("travel_days"),
        travel_dates=flat.get("travel_dates"),
        travelers_count=flat.get("travelers_count") or 1,
        travelers_type=flat.get("travelers_type"),
        budget_range=flat.get("budget_range"),
        food_preferences=flat.get("food_preferences") or [],
        interests=flat.get("interests") or [],
        pace=flat.get("pace") or "moderate",
        accommodation_preference=flat.get("accommodation_preference"),
        special_requests=flat.get("special_requests") or [],
    )


def user_profile_from_job(job: Any) -> UserProfile:
    """Build ``UserProfile`` from a ``PlanningJob`` (WS or REST creation paths)."""
    fb = effective_user_feedback(job.user_feedback, job.input_requirements)
    return user_profile_from_feedback(fb)


async def process_user_turn(state: dict[str, Any], content: str) -> IntentResult:
    """Recognize intent, merge profile, append user message — one code path."""
    recent = state.get("recent_messages", [])
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in recent
        if m.get("role") and m.get("content")
    ][-10:]

    flat = flatten_profile(state.get("profile", {}))
    profile_kwargs = {k: v for k, v in flat.items() if v is not None and v != []}
    user_profile = UserProfile(**profile_kwargs) if profile_kwargs else None

    agent = IntentRecognitionAgent()
    result = await agent.recognize(content, history, user_profile)

    state["profile"] = merge_profile(state["profile"], entities_to_patch(result.user_entities))
    state["last_intent"] = result.intent
    state["missing_required"] = result.missing_required

    append_message(state, "user", content)
    state["turn"] = int(state.get("turn", 0)) + 1

    return result
