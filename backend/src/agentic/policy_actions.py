"""Policy-visible action contracts, separate from trusted executor payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _PolicyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyArguments(_PolicyArguments):
    pass


class AskUserArguments(_PolicyArguments):
    question: str = Field(min_length=1, description="One concise question for missing user input")


class AbortArguments(_PolicyArguments):
    reason: str = Field(min_length=1, description="Grounded reason the task cannot continue safely")


class ProposeTradeoffArguments(_PolicyArguments):
    reason: str = Field(min_length=1, description="Constraint conflict or unavailable capability")
    options: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three grounded alternatives for the user",
    )


class WeatherPolicyArguments(_PolicyArguments):
    date: str | None = Field(default=None, description="Grounded date in YYYY-MM-DD format")


class SearchPOIsPolicyArguments(_PolicyArguments):
    keywords: list[str] = Field(default_factory=list, max_length=8)
    category: Literal["attraction", "restaurant", "hotel", "shopping"] | None = None


class SolvePolicyArguments(_PolicyArguments):
    strategy: Literal["auto", "cpsat", "greedy"] = "auto"


POLICY_ACTION_MODELS: dict[str, type[BaseModel]] = {
    "abort": AbortArguments,
    "ask_user": AskUserArguments,
    "capability_check": EmptyArguments,
    "compose_draft": EmptyArguments,
    "finish": EmptyArguments,
    "propose_tradeoff": ProposeTradeoffArguments,
    "get_weather": WeatherPolicyArguments,
    "search_pois": SearchPOIsPolicyArguments,
    "get_poi_detail": EmptyArguments,
    "get_route_matrix": EmptyArguments,
    "solve_itinerary": SolvePolicyArguments,
    "validate_itinerary": EmptyArguments,
}

_DESCRIPTIONS = {
    "abort": "Stop safely when the task is unsupported, unsafe, or infeasible.",
    "ask_user": "Ask for information or confirmation that only the user can provide.",
    "capability_check": "Record the controller-computed capability assessment.",
    "compose_draft": "Project a user-facing draft from the verified solver artifact.",
    "finish": "Present the verified draft and wait for confirmation.",
    "propose_tradeoff": "Offer grounded alternatives when constraints conflict.",
    "get_weather": "Read the trusted destination's weather snapshot.",
    "search_pois": "Search POIs in the trusted destination using grounded preferences.",
    "get_poi_detail": "Collect details for the controller-selected POI candidates.",
    "get_route_matrix": "Build a matrix from trusted candidate and constraint artifacts.",
    "solve_itinerary": "Run deterministic constraint solving over trusted artifacts.",
    "validate_itinerary": "Run the programmatic hard-constraint validator.",
}


def policy_action_schema(action: str) -> dict[str, Any]:
    """Return one OpenAI/Transformers-compatible function schema."""
    model = POLICY_ACTION_MODELS.get(action)
    if model is None:
        raise ValueError(f"unknown policy action: {action}")
    parameters = model.model_json_schema()
    parameters.pop("title", None)
    return {
        "type": "function",
        "function": {
            "name": action,
            "description": _DESCRIPTIONS[action],
            "parameters": parameters,
        },
    }


def policy_action_schemas(actions: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    """Build schemas in controller order and reject unknown controller actions."""
    return [policy_action_schema(action) for action in actions]


def validate_policy_arguments(action: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate exactly what the model may choose; trusted fields are never accepted."""
    model = POLICY_ACTION_MODELS.get(action)
    if model is None:
        raise ValueError(f"unknown policy action: {action}")
    try:
        return model.model_validate(arguments).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise ValueError(f"invalid policy arguments for {action}: {exc}") from exc


__all__ = [
    "POLICY_ACTION_MODELS",
    "policy_action_schema",
    "policy_action_schemas",
    "validate_policy_arguments",
]
