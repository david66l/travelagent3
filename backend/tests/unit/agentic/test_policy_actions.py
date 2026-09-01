"""Tests for the policy-visible authority boundary."""

from types import SimpleNamespace

import pytest

from agentic.loop import PolicyAction
from agentic.policy_actions import (
    PolicyArgumentValidationError,
    authorize_policy_action,
    controller_tradeoff_options,
    policy_action_schemas,
    policy_action_schemas_for_state,
    policy_tool_call_json_schema,
    policy_tool_call_json_schema_for_state,
    project_model_owned_arguments,
    unauthorized_tradeoff_alternatives,
    validate_policy_arguments,
    validate_policy_arguments_for_state,
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


def test_policy_argument_rejections_have_stable_machine_codes():
    with pytest.raises(PolicyArgumentValidationError) as unexpected:
        validate_policy_arguments(
            "retry_solve",
            {"strategy": "greedy", "reason": "grounded", "candidates": ["poi-1"]},
        )
    assert unexpected.value.rejection_code == "UNEXPECTED_ARGUMENT:candidates"
    assert unexpected.value.validation_errors[0]["type"] == "extra_forbidden"

    with pytest.raises(PolicyArgumentValidationError) as missing:
        validate_policy_arguments("retry_solve", {"strategy": "greedy"})
    assert missing.value.rejection_code == "MISSING_ARGUMENT:reason"

    # ``strategy`` is intentionally controller-owned.  It is removed from the
    # model-visible payload here and separately recorded as an override attempt
    # at the central authorization boundary.
    assert validate_policy_arguments(
        "retry_solve",
        {"strategy": "auto", "reason": "grounded"},
    ) == {"reason": "grounded"}


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


def test_tradeoff_options_are_hidden_and_hydrated_from_controller_capability():
    capability = {
        "status": "infeasible",
        "actionable_alternatives": True,
        "alternatives": ["提高总预算"],
    }
    schema = policy_action_schemas_for_state(
        ["propose_tradeoff", "abort"],
        capability=capability,
    )[0]
    properties = schema["function"]["parameters"]["properties"]
    model_arguments = validate_policy_arguments_for_state(
        "propose_tradeoff",
        {
            "reason": "餐饮费用超过预算",
            "options": ["提高总预算", "取消高价餐厅"],
        },
        capability=capability,
    )
    authorized = authorize_policy_action(
        SimpleNamespace(capability=capability, hard_constraints={}),
        PolicyAction(action="propose_tradeoff", arguments=model_arguments),
    )

    assert set(properties) == {"reason"}
    assert model_arguments == {"reason": "餐饮费用超过预算"}
    assert authorized.arguments == {
        "reason": "餐饮费用超过预算",
        "options": ["提高总预算"],
    }


def test_tradeoff_fails_closed_without_exact_controller_authority():
    capability = {
        "status": "infeasible",
        "actionable_alternatives": True,
        "alternatives": [],
    }

    with pytest.raises(ValueError, match="CONTROLLER_TRADEOFF_AUTHORITY_MISSING"):
        validate_policy_arguments_for_state(
            "propose_tradeoff",
            {"reason": "冲突"},
            capability=capability,
        )

    assert [
        schema["function"]["name"]
        for schema in policy_action_schemas_for_state(
            ["propose_tradeoff", "abort"], capability=capability
        )
    ] == ["abort"]


@pytest.mark.parametrize(
    "alternatives",
    [
        [""],
        ["提高预算", "提高预算"],
        ["一", "二", "三", "四"],
        [123],
    ],
)
def test_tradeoff_authority_rejects_malformed_alternatives(alternatives):
    assert (
        controller_tradeoff_options(
            {
                "status": "infeasible",
                "actionable_alternatives": True,
                "alternatives": alternatives,
            }
        )
        is None
    )


def test_controller_override_is_audited_but_never_executed_or_exported():
    capability = {
        "status": "infeasible",
        "actionable_alternatives": True,
        "alternatives": ["提高总预算"],
    }
    action = PolicyAction(
        action="propose_tradeoff",
        arguments={"reason": "餐饮费用超过预算"},
        model_arguments={
            "reason": "餐饮费用超过预算",
            "options": ["提高总预算", "取消高价餐厅"],
        },
    )

    authorized = authorize_policy_action(
        SimpleNamespace(capability=capability, hard_constraints={}),
        action,
    )

    assert authorized.arguments["options"] == ["提高总预算"]
    assert authorized.model_arguments["options"] == ["提高总预算", "取消高价餐厅"]
    assert authorized.controller_override_attempt is True
    assert authorized.model_contract_compliant is False
    assert authorized.controller_hydration_exact is True
    assert project_model_owned_arguments(authorized) == {"reason": "餐饮费用超过预算"}


def test_retry_strategy_is_hidden_hydrated_and_audited_as_controller_owned():
    raw_arguments = {"strategy": "cpsat", "reason": "验证器确认当前排程仍可重排"}
    model_arguments = validate_policy_arguments_for_state(
        "retry_solve", raw_arguments, capability={}
    )
    authorized = authorize_policy_action(
        SimpleNamespace(capability={}, hard_constraints={}),
        PolicyAction(
            action="retry_solve",
            arguments=model_arguments,
            model_arguments=raw_arguments,
        ),
    )

    assert model_arguments == {"reason": "验证器确认当前排程仍可重排"}
    assert authorized.arguments == {
        "reason": "验证器确认当前排程仍可重排",
        "strategy": "greedy",
    }
    assert authorized.model_arguments == raw_arguments
    assert authorized.controller_override_attempt is True
    assert authorized.model_contract_compliant is False
    assert authorized.controller_hydration_exact is True
    assert authorized.controller_hydrated_fields == ["strategy"]
    assert project_model_owned_arguments(authorized) == {
        "reason": "验证器确认当前排程仍可重排"
    }


def test_state_scoped_structured_schema_exposes_reason_only():
    schema = policy_tool_call_json_schema_for_state(
        ["propose_tradeoff", "abort"],
        capability={
            "status": "infeasible",
            "actionable_alternatives": True,
            "alternatives": ["提高总预算"],
        },
    )
    tradeoff = schema["oneOf"][0]

    assert tradeoff["properties"]["name"] == {"const": "propose_tradeoff"}
    assert set(tradeoff["properties"]["arguments"]["properties"]) == {"reason"}
    assert tradeoff["properties"]["arguments"]["additionalProperties"] is False


def test_tradeoff_reason_cannot_smuggle_a_different_relaxation():
    violations = unauthorized_tradeoff_alternatives(
        "预算冲突，建议减少一个非必去活动",
        capability={
            "status": "infeasible",
            "actionable_alternatives": True,
            "alternatives": ["提高总预算"],
        },
        hard_constraints={
            "constraint_flexibility": {
                "relaxation_options": {
                    "total_budget": ["提高总预算"],
                    "activity_set": ["减少一个非必去活动"],
                }
            }
        },
    )

    assert violations == ["减少一个非必去活动"]


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
