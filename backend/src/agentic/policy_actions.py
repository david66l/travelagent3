"""Policy-visible action contracts, separate from trusted executor payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class _PolicyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PolicyArgumentValidationError(ValueError):
    """Structured, stable rejection for one model-authored argument object."""

    def __init__(
        self,
        *,
        action: str,
        rejection_code: str,
        validation_errors: list[dict[str, Any]],
    ) -> None:
        super().__init__(f"{rejection_code}: invalid policy arguments for {action}")
        self.action = action
        self.rejection_code = rejection_code
        self.validation_errors = validation_errors


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
    # A retry is authorized only after the controller has observed one failed
    # primary solve.  The fallback solver is therefore an execution-policy
    # decision, not a language-model choice.  Keeping it off the model surface
    # prevents a model from claiming unsupported solver authority.
    "retry_solve": frozenset({"strategy"}),
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
    "retry_solve": (
        "Request one bounded deterministic retry after verifier evidence; the controller "
        "selects the safe fallback solver strategy."
    ),
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


def controller_tradeoff_options(capability: dict[str, Any]) -> list[str] | None:
    """Return the exact controller-authorized options for the current failure."""
    if capability.get("actionable_alternatives") is not True:
        return None
    if capability.get("status") not in {"infeasible", "unsafe", "missing_tool"}:
        return None
    raw = capability.get("alternatives")
    if not isinstance(raw, list) or not raw or len(raw) > 3:
        return None
    if any(not isinstance(item, str) for item in raw):
        return None
    options = [item.strip() for item in raw]
    if any(not option for option in options) or len(set(options)) != len(options):
        return None
    return options


def model_visible_policy_actions(
    actions: list[str] | tuple[str, ...],
    *,
    capability: dict[str, Any],
) -> list[str]:
    """Remove actions whose controller authority is absent from the model surface."""
    return [
        action
        for action in actions
        if action != "propose_tradeoff" or controller_tradeoff_options(capability) is not None
    ]


def policy_action_schemas_for_state(
    actions: list[str] | tuple[str, ...],
    *,
    capability: dict[str, Any],
) -> list[dict[str, Any]]:
    """Hide controller-owned tradeoff options from the model tool surface."""
    visible_actions = model_visible_policy_actions(actions, capability=capability)
    schemas = policy_action_schemas(visible_actions)
    if "propose_tradeoff" not in visible_actions:
        return schemas
    output = deepcopy(schemas)
    for schema in output:
        function = schema.get("function") or {}
        if function.get("name") != "propose_tradeoff":
            continue
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        properties.pop("options", None)
        function["description"] = (
            "Explain the current grounded constraint conflict. The controller adds the "
            "exact verifier-authorized alternatives; do not generate options."
        )
    return output


def project_model_owned_arguments(action: Any) -> dict[str, Any]:
    """Project one recorded action back to fields the model was allowed to author."""
    name = str(getattr(action, "action", ""))
    raw = getattr(action, "model_arguments", None)
    if raw is None:
        raw = getattr(action, "arguments", {})
    arguments = dict(raw or {})
    # ``options`` remains in the legacy global Pydantic model for backwards
    # compatibility, but is never model-owned in the state-scoped contract.
    if name == "propose_tradeoff":
        arguments.pop("options", None)
    return strip_policy_schema_artifacts(name, arguments)


def controller_retry_strategy() -> Literal["greedy"]:
    """Return the single controller-authorized fallback after the primary CP-SAT solve.

    Runtime authorization already limits this action to one retry after a
    verifier-confirmed, solver-adjustable failure.  The executor forces the
    primary solve to CP-SAT, so ``greedy`` is the deterministic, bounded
    alternate path.  This deliberately has no model-visible inputs.
    """
    return "greedy"


def controller_override_attempt(action: str, arguments: dict[str, Any]) -> bool:
    """Whether a model tried to set a controller-owned decision field."""
    return (action == "propose_tradeoff" and "options" in arguments) or (
        action == "retry_solve" and "strategy" in arguments
    )


def unauthorized_tradeoff_alternatives(
    reason: str,
    *,
    capability: dict[str, Any],
    hard_constraints: dict[str, Any],
) -> list[str]:
    """Find globally visible relaxation choices not authorized for this failure."""
    allowed = set(controller_tradeoff_options(capability) or [])
    contract = hard_constraints.get("constraint_flexibility")
    raw_options = contract.get("relaxation_options") if isinstance(contract, dict) else None
    if not isinstance(raw_options, dict):
        return []
    normalized_reason = "".join(character.casefold() for character in reason if not character.isspace())
    violations: list[str] = []
    for values in raw_options.values():
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str):
                continue
            option = value.strip()
            normalized_option = "".join(
                character.casefold() for character in option if not character.isspace()
            )
            if (
                option
                and option not in allowed
                and normalized_option
                and normalized_option in normalized_reason
                and option not in violations
            ):
                violations.append(option)
    return violations


def validate_policy_arguments_for_state(
    action: str,
    arguments: dict[str, Any],
    *,
    capability: dict[str, Any],
) -> dict[str, Any]:
    """Validate only model-owned fields under the current controller authority."""
    if action == "retry_solve":
        model_arguments = dict(arguments)
        model_arguments.pop("strategy", None)
        return validate_policy_arguments(action, model_arguments)
    if action != "propose_tradeoff":
        return validate_policy_arguments(action, arguments)
    options = controller_tradeoff_options(capability)
    if options is None:
        raise ValueError("CONTROLLER_TRADEOFF_AUTHORITY_MISSING")
    model_arguments = dict(arguments)
    model_arguments.pop("options", None)
    validated = validate_policy_arguments(action, model_arguments)
    # ``TradeoffArguments.options`` retains its legacy default for callers that
    # still use the state-agnostic contract.  Under the state-aware contract it
    # is controller-owned, so do not let Pydantic materialize that default back
    # into the model-owned payload.
    validated.pop("options", None)
    return validated


def authorize_policy_action(context: Any, action: Any) -> Any:
    """Apply controller-owned argument authority at the central Agent Loop boundary."""
    action_name = str(action.action)
    if action_name not in {"propose_tradeoff", "retry_solve"}:
        # Other actions keep their existing validation/execution boundary.  In
        # particular, this helper must not widen a tradeoff-specific contract
        # migration into an unrelated Agent Loop behavior change.
        return action
    raw_arguments = dict(
        action.model_arguments if action.model_arguments is not None else action.arguments
    )
    arguments = validate_policy_arguments_for_state(
        action_name,
        raw_arguments,
        capability=dict(context.capability),
    )
    if action_name == "propose_tradeoff":
        unauthorized = unauthorized_tradeoff_alternatives(
            str(arguments.get("reason") or ""),
            capability=dict(context.capability),
            hard_constraints=dict(context.hard_constraints),
        )
        if unauthorized:
            raise ValueError("UNAUTHORIZED_TRADEOFF_ALTERNATIVE_IN_REASON")
        options = controller_tradeoff_options(dict(context.capability))
        if options is None:
            raise ValueError("CONTROLLER_TRADEOFF_AUTHORITY_MISSING")
        arguments["options"] = options
        hydrated_fields = ["options"]
    else:
        arguments["strategy"] = controller_retry_strategy()
        hydrated_fields = ["strategy"]
    override_attempt = controller_override_attempt(action_name, raw_arguments)
    return action.model_copy(
        update={
            "arguments": arguments,
            "model_arguments": raw_arguments,
            "controller_override_attempt": override_attempt,
            "model_contract_compliant": not override_attempt,
            "controller_hydration_exact": True,
            "controller_hydrated_fields": hydrated_fields,
        }
    )


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


def policy_tool_call_json_schema_for_state(
    actions: list[str] | tuple[str, ...],
    *,
    capability: dict[str, Any],
) -> dict[str, Any]:
    """Build constrained decoding grammar from the state-scoped tool contract."""
    schemas = policy_action_schemas_for_state(actions, capability=capability)
    if not schemas:
        raise ValueError("at least one controller-authorized policy action is required")
    branches = []
    for schema in schemas:
        function = schema["function"]
        branches.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": function["name"]},
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
        errors = exc.errors(include_url=False, include_input=False)
        first = errors[0] if errors else {}
        location = ".".join(str(item) for item in first.get("loc") or ()) or "unknown"
        rejection_prefix = {
            "extra_forbidden": "UNEXPECTED_ARGUMENT",
            "missing": "MISSING_ARGUMENT",
            "literal_error": "INVALID_ARGUMENT_ENUM",
        }.get(str(first.get("type") or ""), "INVALID_ARGUMENT")
        raise PolicyArgumentValidationError(
            action=action,
            rejection_code=f"{rejection_prefix}:{location}",
            validation_errors=errors,
        ) from exc


__all__ = [
    "POLICY_ACTION_MODELS",
    "PolicyArgumentValidationError",
    "model_visible_policy_actions",
    "policy_action_schema",
    "policy_action_schemas",
    "policy_action_schemas_for_state",
    "policy_tool_call_json_schema",
    "policy_tool_call_json_schema_for_state",
    "project_model_owned_arguments",
    "authorize_policy_action",
    "controller_override_attempt",
    "controller_retry_strategy",
    "controller_tradeoff_options",
    "strip_policy_schema_artifacts",
    "validate_policy_arguments",
    "validate_policy_arguments_for_state",
    "unauthorized_tradeoff_alternatives",
]
