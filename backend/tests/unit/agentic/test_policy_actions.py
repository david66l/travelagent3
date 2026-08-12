"""Tests for the policy-visible authority boundary."""

import pytest

from agentic.policy_actions import policy_action_schemas, validate_policy_arguments


def test_solve_schema_hides_controller_owned_solver_payloads():
    schema = policy_action_schemas(["solve_itinerary"])[0]
    properties = schema["function"]["parameters"]["properties"]

    assert set(properties) == {"strategy"}
    assert validate_policy_arguments("solve_itinerary", {"strategy": "cpsat"}) == {
        "strategy": "cpsat"
    }


def test_policy_cannot_supply_trusted_city_or_constraint_payload():
    with pytest.raises(ValueError, match="invalid policy arguments"):
        validate_policy_arguments("get_weather", {"city": "Shanghai"})
    with pytest.raises(ValueError, match="invalid policy arguments"):
        validate_policy_arguments("solve_itinerary", {"constraints": {"travel_days": 99}})


def test_controller_actions_have_explicit_function_schemas():
    schemas = policy_action_schemas(["ask_user", "finish", "propose_tradeoff"])

    assert [item["function"]["name"] for item in schemas] == [
        "ask_user",
        "finish",
        "propose_tradeoff",
    ]
    assert schemas[0]["function"]["parameters"]["required"] == ["question"]
