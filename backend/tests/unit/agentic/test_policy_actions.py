"""Tests for the policy-visible authority boundary."""

import pytest

from agentic.policy_actions import (
    policy_action_schemas,
    policy_tool_call_json_schema,
    validate_policy_arguments,
)


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
    with pytest.raises(ValueError, match="invalid policy arguments"):
        validate_policy_arguments("search_pois", {"category": "restaurant"})


def test_policy_arguments_strip_exact_schema_annotations_only():
    assert validate_policy_arguments(
        "retrieve_city_knowledge",
        {"topic": "历史", "default": None},
    ) == {"topic": "历史"}
    assert validate_policy_arguments(
        "search_pois",
        {"keywords": ["博物馆"], "maxItems": 8},
    ) == {"keywords": ["博物馆"]}

    with pytest.raises(ValueError, match="invalid policy arguments"):
        validate_policy_arguments(
            "search_pois",
            {"keywords": ["博物馆"], "maxItems": 99},
        )


def test_policy_arguments_drop_controller_owned_poi_fields():
    assert validate_policy_arguments(
        "get_poi_detail",
        {
            "candidate_poi_ids": ["poi-1", "poi-2"],
            "poi_ids": ["poi-1"],
            "poi_names": ["Museum"],
            "city": "Beijing",
        },
    ) == {}


def test_policy_arguments_still_reject_unknown_poi_fields():
    with pytest.raises(ValueError, match="invalid policy arguments"):
        validate_policy_arguments("get_poi_detail", {"untrusted_override": ["poi-1"]})


def test_unified_search_supports_open_ended_event_queries():
    arguments = validate_policy_arguments(
        "search_current_info",
        {"query": "上海周末音乐节", "info_type": "event"},
    )

    assert arguments == {"query": "上海周末音乐节", "info_type": "event"}


def test_controller_actions_have_explicit_function_schemas():
    schemas = policy_action_schemas(["ask_user", "finish", "propose_tradeoff"])

    assert [item["function"]["name"] for item in schemas] == [
        "ask_user",
        "finish",
        "propose_tradeoff",
    ]
    assert schemas[0]["function"]["parameters"]["required"] == ["question"]


def test_constrained_schema_binds_each_action_to_its_arguments():
    schema = policy_tool_call_json_schema(["ask_user", "search_pois"])

    assert schema["title"] == "PolicyToolCall"
    assert [branch["properties"]["name"]["const"] for branch in schema["oneOf"]] == [
        "ask_user",
        "search_pois",
    ]
    assert set(schema["oneOf"][0]["properties"]["arguments"]["properties"]) == {"question"}
    assert set(schema["oneOf"][1]["properties"]["arguments"]["properties"]) == {"keywords"}
    assert all(branch["additionalProperties"] is False for branch in schema["oneOf"])


def test_constrained_schema_rejects_empty_action_set():
    with pytest.raises(ValueError, match="at least one policy action"):
        policy_tool_call_json_schema([])
