"""LangGraph State definition for ItineraryState."""

from typing import TypedDict, Annotated
from operator import add


class ItineraryState(TypedDict):
    """Global state shared across all graph nodes."""

    # Session & Input
    session_id: str
    user_id: str
    user_input: str
    messages: Annotated[list[dict], add]  # Accumulate with operator.add

    # Intent Recognition results
    intent: str | None
    intent_confidence: float
    user_entities: dict
    missing_required: list[str]
    missing_recommended: list[str]
    preference_changes: list[dict]
    clarification_questions: list[str]

    # User profile & query results
    user_profile: dict | None
    candidate_pois: list[dict]
    weather_data: list[dict]
    travel_context: dict | None
    price_data: dict

    # Itinerary data
    current_itinerary: list[dict] | None
    itinerary_status: str
    planning_json: dict | None

    # Validation & optimization
    validation_result: dict | None
    optimized_routes: dict

    # Panels & output
    budget_panel: dict | None
    preference_panel: dict | None
    assistant_response: str | None
    proposal_text: str | None

    # Control flow
    needs_clarification: bool
    needs_replan: bool
    waiting_for_confirmation: bool
    error: str | None
