"""Single-path user turn handling — intent recognition + profile merge.

WebSocket uses this before creating a planning job; the pipeline reads the
persisted profile from ``job.user_feedback`` instead of re-running intent LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.demand_parser import DemandParserAgent
from agents.disambiguation import DisambiguationEngine
from agents.feasibility import FeasibilityChecker
from agents.memory_conflict_resolver import MemoryConflictResolver
from agents.profile_recall import ProfileRecallAgent
from core.conversation_state import (
    append_message,
    default_conversation_state,
    flatten_profile,
    merge_profile,
)
from core.langsmith_trace import traceable_step
from models.travel_slots import SlotParseOutput, TravelSlots
from schemas import IntentResult, ProfilePatch, UserProfile

logger = logging.getLogger(__name__)


def entities_to_patch(entities: dict[str, Any]) -> ProfilePatch:
    """Convert flat ``user_entities`` into a ``ProfilePatch``."""
    scalar_keys = {
        "origin",
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


def slots_to_patch(slots: TravelSlots) -> ProfilePatch:
    """Convert parsed ``TravelSlots`` into a ``ProfilePatch`` for state merge."""
    scalar_map = {
        "origin": "origin",
        "destination": "destination",
        "travel_days": "travel_days",
        "travel_dates": "travel_dates",
        "travelers_count": "travelers_count",
        "travel_companion": "travelers_type",
        "total_budget": "budget_range",
        "pace": "pace",
    }
    list_map = {
        "interests": "interests",
        "food_prefs": "food_preferences",
        "food_taboos": "avoid",
        "must_not_visit": "avoid",
    }

    set_patch: dict[str, Any] = {}
    add_patch: dict[str, list] = {}
    data = slots.model_dump(exclude_none=True)

    for slot_key, profile_key in scalar_map.items():
        value = data.get(slot_key)
        if value is not None and value != "":
            set_patch[profile_key] = value

    if "has_children" in data and data["has_children"] is not None:
        set_patch["has_children"] = data["has_children"]
    if "has_elderly" in data and data["has_elderly"] is not None:
        set_patch["has_elderly"] = data["has_elderly"]

    for slot_key, profile_key in list_map.items():
        values = data.get(slot_key)
        if values:
            add_patch.setdefault(profile_key, []).extend(values)

    # must_visit 进入 special_requests（本次特殊要求）
    must_visit = data.get("must_visit")
    if must_visit:
        add_patch.setdefault("special_requests", []).extend(
            [f"必去：{v}" for v in must_visit]
        )

    return ProfilePatch(set=set_patch, add=add_patch)


def slots_from_merged_profile(resolved: TravelSlots, flat: dict) -> TravelSlots:
    """Combine current-turn slots with accumulated profile for checks and responses."""
    data = resolved.model_dump()
    field_map = {
        "origin": "origin",
        "destination": "destination",
        "travel_days": "travel_days",
        "travel_dates": "travel_dates",
        "travelers_count": "travelers_count",
        "total_budget": "budget_range",
        "pace": "pace",
        "travel_companion": "travelers_type",
    }
    for slot_key, profile_key in field_map.items():
        if data.get(slot_key) is None and flat.get(profile_key) is not None:
            data[slot_key] = flat[profile_key]
    if data.get("has_children") is None and "has_children" in flat:
        data["has_children"] = flat["has_children"]
    if data.get("has_elderly") is None and "has_elderly" in flat:
        data["has_elderly"] = flat["has_elderly"]
    return TravelSlots(**data)


def slot_parse_output_to_intent_result(
    parsed: SlotParseOutput,
    resolved_slots: TravelSlots,
    inferred_slots: dict[str, Any],
    feasibility: dict[str, Any],
) -> IntentResult:
    """Convert SlotParseOutput + downstream enrichments into IntentResult."""
    clarification_questions: list[str] = []
    if parsed.clarifying_question:
        clarification_questions.append(parsed.clarifying_question)

    if not feasibility["feasible"]:
        for issue in feasibility["issues"]:
            if issue not in clarification_questions:
                clarification_questions.append(issue)

    candidates = []
    if parsed.disambiguation:
        candidates = parsed.disambiguation.get("candidates", [])

    return IntentResult(
        intent=parsed.intent,
        confidence=parsed.confidence,
        sentiment=parsed.sentiment,
        user_entities=resolved_slots.to_flat_dict(),
        slots=resolved_slots.to_flat_dict(),
        missing_required=parsed.missing_slots,
        clarification_questions=clarification_questions,
        disambiguation_candidates=candidates,
        feasibility_report=feasibility,
        reasoning="Parsed by DemandParserAgent with profile recall and feasibility check.",
    )


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
        origin=flat.get("origin"),
        destination=flat.get("destination"),
        travel_days=flat.get("travel_days"),
        travel_dates=flat.get("travel_dates"),
        travelers_count=flat.get("travelers_count"),
        travelers_type=flat.get("travelers_type"),
        budget_range=flat.get("budget_range"),
        food_preferences=flat.get("food_preferences") or [],
        food_taboos=flat.get("food_taboos") or [],
        interests=flat.get("interests") or [],
        pace=flat.get("pace") or "moderate",
        accommodation_preference=flat.get("accommodation_preference"),
        special_requests=flat.get("special_requests") or [],
        **({"has_children": flat["has_children"]} if "has_children" in flat else {}),
    )


def user_profile_from_job(job: Any) -> UserProfile:
    """Build ``UserProfile`` from a ``PlanningJob`` (WS or REST creation paths)."""
    fb = effective_user_feedback(job.user_feedback, job.input_requirements)
    return user_profile_from_feedback(fb)


@traceable_step("intent/demand_parser", run_type="chain")
async def _trace_demand_parser(
    content: str,
    history: list[dict[str, str]],
    user_profile: UserProfile | None,
    flat: dict[str, Any],
) -> SlotParseOutput:
    parser = DemandParserAgent()
    return await parser.parse(content, history, user_profile, known_profile=flat)


@traceable_step("intent/profile_recall", run_type="chain")
async def _trace_profile_recall(
    state: dict[str, Any],
    current_slots: TravelSlots,
) -> dict[str, Any]:
    recall_agent = ProfileRecallAgent()
    return await recall_agent.recall(
        state.get("user_id"),
        current_slots,
        short_term_state=state,
    )


@traceable_step("intent/memory_conflict_resolve", run_type="chain")
def _trace_memory_conflict_resolve(
    current_slots: TravelSlots,
    recall_result: dict[str, Any],
) -> dict[str, Any]:
    resolver = MemoryConflictResolver()
    return resolver.resolve(
        short_term=current_slots.to_flat_dict(),
        long_term=recall_result["recalled_profile"].model_dump(exclude_none=True),
    )


@traceable_step("intent/disambiguation", run_type="chain")
def _trace_disambiguation(
    parsed: SlotParseOutput,
    current_slots: TravelSlots,
    content: str,
) -> SlotParseOutput:
    if parsed.disambiguation:
        return parsed
    parsed.disambiguation = DisambiguationEngine.analyze(current_slots, content)
    if parsed.disambiguation.get("has_ambiguity") and parsed.disambiguation.get("question"):
        parsed.clarifying_question = parsed.disambiguation["question"]
    return parsed


@traceable_step("intent/profile_merge", run_type="chain")
def _trace_profile_merge(
    state: dict[str, Any],
    resolved_slots: TravelSlots,
) -> tuple[dict[str, Any], TravelSlots]:
    state["profile"] = merge_profile(state["profile"], slots_to_patch(resolved_slots))
    merged_flat = flatten_profile(state["profile"])
    merged_slots = slots_from_merged_profile(resolved_slots, merged_flat)
    return merged_flat, merged_slots


@traceable_step("intent/feasibility_check", run_type="chain")
def _trace_feasibility_check(merged_slots: TravelSlots) -> dict[str, Any]:
    return dict(FeasibilityChecker.check(merged_slots))


@traceable_step("intent/build_result", run_type="chain")
def _trace_build_intent_result(
    parsed: SlotParseOutput,
    merged_slots: TravelSlots,
    inferred_slots: dict[str, Any],
    feasibility: dict[str, Any],
    merged_flat: dict[str, Any],
) -> IntentResult:
    result = slot_parse_output_to_intent_result(
        parsed, merged_slots, inferred_slots, feasibility
    )
    missing_required = DemandParserAgent.missing_from_profile(merged_flat)
    if (
        parsed.disambiguation
        and parsed.disambiguation.get("has_ambiguity")
        and parsed.disambiguation.get("question")
    ):
        clarification_questions = [parsed.disambiguation["question"]]
    else:
        clarification_questions = DemandParserAgent.build_clarification_questions(merged_flat)
    if not feasibility["feasible"]:
        for issue in feasibility["issues"]:
            if issue not in clarification_questions:
                clarification_questions.append(issue)

    return result.model_copy(
        update={
            "missing_required": missing_required,
            "clarification_questions": clarification_questions,
            "feasibility_report": feasibility,
            "slots": merged_slots.to_flat_dict(),
            "user_entities": merged_slots.to_flat_dict(),
        }
    )


@traceable_step("intent/process_user_turn", run_type="chain")
async def process_user_turn(state: dict[str, Any], content: str) -> IntentResult:
    """Recognize intent, parse slots, recall profile, check feasibility — one code path."""
    recent = state.get("recent_messages", [])
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in recent
        if m.get("role") and m.get("content")
    ][-10:]

    flat = flatten_profile(state.get("profile", {}))
    profile_kwargs = {k: v for k, v in flat.items() if v is not None and v != []}
    user_profile = UserProfile(**profile_kwargs) if profile_kwargs else None

    parsed = await _trace_demand_parser(content, history, user_profile, flat)
    current_slots = parsed.slots

    recall_result = await _trace_profile_recall(state, current_slots)

    resolved_values = _trace_memory_conflict_resolve(current_slots, recall_result)

    inferred_slots = recall_result["inferred_slots"]
    resolved_values["inferred_slots"] = list(inferred_slots.keys())
    resolved_slots = TravelSlots(**resolved_values)

    parsed = _trace_disambiguation(parsed, current_slots, content)

    merged_flat, merged_slots = _trace_profile_merge(state, resolved_slots)

    feasibility = _trace_feasibility_check(merged_slots)

    result = _trace_build_intent_result(
        parsed, merged_slots, inferred_slots, feasibility, merged_flat
    )
    missing_required = result.missing_required
    state["last_intent"] = result.intent
    state["missing_required"] = missing_required
    state["slots"] = result.slots
    state["inferred_slots"] = inferred_slots
    state["feasibility_report"] = result.feasibility_report

    append_message(state, "user", content)
    state["turn"] = int(state.get("turn", 0)) + 1

    return result
