"""Policy-visible action contracts, separate from trusted executor payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _PolicyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmptyArguments(_PolicyArguments):
    pass


class AskUserArguments(_PolicyArguments):
    question: str = Field(
        min_length=1,
        description=(
            "One concise user-facing question in the user's language; never mention internal "
            "tools, verifier codes, artifacts, policies, or state fields"
        ),
    )


class AbortArguments(_PolicyArguments):
    reason: str = Field(min_length=1, description="Grounded reason the task cannot continue safely")


class ProposeTradeoffArguments(_PolicyArguments):
    reason: str = Field(
        min_length=1,
        description=(
            "Concise user-facing constraint conflict in the user's language; never expose "
            "internal tools, verifier codes, artifacts, policies, or state fields"
        ),
    )
    options: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Up to three concise, grounded, user-facing alternatives",
    )


class WeatherPolicyArguments(_PolicyArguments):
    date: str | None = Field(default=None, description="Grounded date in YYYY-MM-DD format")


class SearchPOIsPolicyArguments(_PolicyArguments):
    keywords: list[str] = Field(default_factory=list, max_length=8)


class CityKnowledgePolicyArguments(_PolicyArguments):
    topic: str | None = Field(default=None, max_length=80)


class CurrentInfoPolicyArguments(_PolicyArguments):
    query: str = Field(min_length=2, max_length=160)
    info_type: Literal[
        "event", "opening_hours", "restaurant", "seasonal_activity", "closure", "general"
    ] = "general"
    date: str | None = None


class TransportSearchPolicyArguments(_PolicyArguments):
    mode: Literal["flight", "train", "both"] = "both"
    date: str | None = None


class SolvePolicyArguments(_PolicyArguments):
    strategy: Literal["auto", "cpsat", "greedy"] = "auto"


class RetrySolveArguments(_PolicyArguments):
    strategy: Literal["cpsat", "greedy"]
    reason: str = Field(min_length=1, description="Verifier-grounded reason for retrying")


POLICY_ACTION_MODELS: dict[str, type[BaseModel]] = {
    "abort": AbortArguments,
    "accept_candidates": EmptyArguments,
    "accept_itinerary": EmptyArguments,
    "ask_user": AskUserArguments,
    "capability_check": EmptyArguments,
    "compose_draft": EmptyArguments,
    "finish": EmptyArguments,
    "propose_tradeoff": ProposeTradeoffArguments,
    "retry_solve": RetrySolveArguments,
    "get_weather": WeatherPolicyArguments,
    "search_pois": SearchPOIsPolicyArguments,
    "retrieve_city_knowledge": CityKnowledgePolicyArguments,
    "search_current_info": CurrentInfoPolicyArguments,
    "search_transport": TransportSearchPolicyArguments,
    "finalize_research": EmptyArguments,
    "get_poi_detail": EmptyArguments,
    "get_route_matrix": EmptyArguments,
    "solve_itinerary": SolvePolicyArguments,
    "validate_itinerary": EmptyArguments,
}

# Small tool-calling models occasionally copy JSON Schema annotations into the
# argument object. These names never carry trusted executor data, so an exact
# annotation copied from the advertised schema can be removed safely. Unknown
# business fields remain a hard error.
_SCHEMA_ANNOTATION_KEYS = frozenset(
    {
        "$schema",
        "additionalProperties",
        "anyOf",
        "const",
        "default",
        "description",
        "enum",
        "examples",
        "maxItems",
        "maxLength",
        "minItems",
        "minLength",
        "oneOf",
        "properties",
        "required",
        "title",
        "type",
    }
)

# These fields are derived from verified ledger artifacts by the executor. A
# policy may select the action, but it must not replace trusted POI identities
# with model-authored values. Small models commonly echo the visible candidate
# list here, so discard only these explicit controller-owned aliases and keep
# rejecting every other unknown business field.
_CONTROLLER_OWNED_ARGUMENTS: dict[str, frozenset[str]] = {
    "get_poi_detail": frozenset({"candidate_poi_ids", "poi_ids", "poi_names", "city"}),
}

_DESCRIPTIONS = {
    "abort": "Stop safely when the task is unsupported, unsafe, or infeasible.",
    "accept_candidates": (
        "Accept the currently grounded POI candidates when they are sufficient to plan."
    ),
    "accept_itinerary": "Accept an itinerary only when the latest verifier report hard-passes.",
    "ask_user": "Ask for information or confirmation that only the user can provide.",
    "capability_check": "Record the controller-computed capability assessment.",
    "compose_draft": "Project a user-facing draft from the verified solver artifact.",
    "finish": "Present the verified draft and wait for confirmation.",
    "propose_tradeoff": "Offer grounded alternatives when constraints conflict.",
    "retry_solve": "Retry deterministic solving with another bounded solver strategy.",
    "get_weather": "Read the trusted destination's weather snapshot.",
    "search_pois": "Search POIs in the trusted destination using grounded preferences.",
    "retrieve_city_knowledge": (
        "Read stable city and POI facts from the local knowledge base before using live search."
    ),
    "search_current_info": (
        "Search source-backed current facts, including any kind of event, opening hours, "
        "closures, restaurants, or seasonal activities."
    ),
    "search_transport": "Search current flight or train schedule evidence for the grounded route.",
    "finalize_research": (
        "Propose that research is complete; the programmatic evidence verifier may reject it."
    ),
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


def policy_tool_call_json_schema(
    actions: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    """Build one state-scoped JSON Schema for constrained local decoding.

    Each branch binds the selected action name to that action's exact argument
    model. A flat ``name`` enum plus an unrelated argument union would permit
    structurally valid but semantically mismatched pairs such as
    ``ask_user`` with ``search_pois`` arguments.
    """
    if not actions:
        raise ValueError("at least one policy action is required")
    branches = []
    for action in actions:
        function = policy_action_schema(action)["function"]
        branches.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": action},
                    "arguments": function["parameters"],
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
        )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PolicyToolCall",
        "oneOf": branches,
    }


def _schema_contains_annotation(schema: Any, key: str, value: Any) -> bool:
    if isinstance(schema, dict):
        if key in schema and schema[key] == value:
            return True
        return any(_schema_contains_annotation(item, key, value) for item in schema.values())
    if isinstance(schema, list):
        return any(_schema_contains_annotation(item, key, value) for item in schema)
    return False


def strip_policy_schema_artifacts(action: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Drop harmless schema copies and explicit controller-owned fields."""
    model = POLICY_ACTION_MODELS.get(action)
    if model is None:
        raise ValueError(f"unknown policy action: {action}")
    schema = model.model_json_schema()
    controller_owned = _CONTROLLER_OWNED_ARGUMENTS.get(action, frozenset())
    return {
        key: value
        for key, value in arguments.items()
        if not (
            key in controller_owned
            or (
                key not in model.model_fields
                and key in _SCHEMA_ANNOTATION_KEYS
                and _schema_contains_annotation(schema, key, value)
            )
        )
    }


def validate_policy_arguments(action: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Validate exactly what the model may choose; trusted fields are never accepted."""
    model = POLICY_ACTION_MODELS.get(action)
    if model is None:
        raise ValueError(f"unknown policy action: {action}")
    try:
        sanitized = strip_policy_schema_artifacts(action, arguments)
        return model.model_validate(sanitized).model_dump(exclude_none=True)
    except ValidationError as exc:
        raise ValueError(f"invalid policy arguments for {action}: {exc}") from exc


__all__ = [
    "POLICY_ACTION_MODELS",
    "policy_action_schema",
    "policy_action_schemas",
    "policy_tool_call_json_schema",
    "strip_policy_schema_artifacts",
    "validate_policy_arguments",
]
