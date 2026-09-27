"""Tests for legacy-to-Agent-Loop state projection."""

from agentic.runtime import initialize_agent_ledger
from agentic.state import AgentLedgerState


def test_retired_mode_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="Only agent"):
        initialize_agent_ledger({"user_input": "Plan Shanghai"}, mode="deterministic")


def test_runtime_initializes_one_model_driven_session():
    result = initialize_agent_ledger(
        {"slots": {"destination": "Shanghai", "travel_days": 2}}, mode="agent"
    )
    assert result["current_task_id"] == "travel_agent"
    assert len(result["agent_ledger"]["task_graph"]["tasks"]) == 1


def test_nullable_capability_lists_are_projected_as_empty_lists():
    result = initialize_agent_ledger(
        {
            "user_input": "Plan Shanghai",
            "slots": {"destination": "Shanghai"},
            "missing_slots": [],
            "feasibility_report": {
                "feasible": True,
                "status": "solvable",
                "reasons": None,
                "alternatives": None,
            },
        },
        mode="agent",
    )
    capability = AgentLedgerState(**result["agent_ledger"]).goal.capability

    assert capability.evidence == []
    assert capability.alternatives == []


def test_profile_travel_dates_are_projected_to_agent_date_bounds():
    result = initialize_agent_ledger(
        {
            "user_input": "南京三日游",
            "profile": {
                "trip": {
                    "destination": "南京",
                    "travel_days": 3,
                    "travel_dates": "2026-12-08|2026-12-10",
                }
            },
            "missing_slots": [],
        },
        mode="agent",
    )
    hard = AgentLedgerState(**result["agent_ledger"]).goal.hard_constraints

    assert hard["start_date"] == "2026-12-08"
    assert hard["end_date"] == "2026-12-10"


def test_traveler_context_is_preserved_as_soft_preferences():
    result = initialize_agent_ledger(
        {
            "user_input": "南京历史文化游，不带孩子",
            "slots": {
                "destination": "南京",
                "interests": ["历史文化"],
                "travelers_count": 2,
                "has_children": False,
                "has_elderly": False,
            },
            "missing_slots": [],
        },
        mode="agent",
    )
    soft = AgentLedgerState(**result["agent_ledger"]).goal.soft_preferences

    assert soft["has_children"] is False
    assert soft["has_elderly"] is False
    assert soft["travelers_count"] == 2


def test_accessibility_and_negative_poi_constraints_are_hard_constraints():
    result = initialize_agent_ledger(
        {
            "user_input": "轮椅出行且不要外滩",
            "slots": {
                "destination": "上海",
                "travel_days": 2,
                "has_wheelchair": True,
                "max_walk_minutes": 40,
                "must_not_visit": ["外滩"],
                "food_taboos": ["花生"],
            },
            "missing_slots": [],
        },
        mode="agent",
    )
    hard = AgentLedgerState(**result["agent_ledger"]).goal.hard_constraints

    assert hard["has_wheelchair"] is True
    assert hard["max_walk_minutes"] == 40
    assert hard["must_not_visit"] == ["外滩"]
    assert hard["food_taboos"] == ["花生"]


def test_neutral_constraint_flexibility_is_projected_without_action_labels():
    contract = {
        "schema_version": "constraint-flexibility.v1",
        "locked_constraints": ["total_budget"],
        "solver_adjustable_constraints": ["activity_schedule"],
        "relaxable_constraints": ["activity_set"],
        "relaxation_options": {"activity_set": ["减少一个非必去活动"]},
    }
    result = initialize_agent_ledger(
        {
            "user_input": "预算锁定，活动顺序可以重排，必要时可以问我是否删普通活动",
            "slots": {
                "destination": "上海",
                "travel_days": 2,
                "constraint_flexibility": contract,
            },
            "missing_slots": [],
        },
        mode="agent",
    )
    hard = AgentLedgerState(**result["agent_ledger"]).goal.hard_constraints

    assert hard["constraint_flexibility"] == contract
    serialized = str(hard["constraint_flexibility"])
    assert "target_action" not in serialized
    assert "solver_retry_authorized" not in serialized
    assert "stop_if_unresolved" not in serialized


def test_agent_goal_uses_model_semantics_instead_of_request_keywords():
    without_model_semantics = initialize_agent_ledger(
        {
            "user_input": "去上海看演唱会并坐高铁",
            "slots": {"destination": "上海", "travel_days": 2},
        },
        mode="agent",
    )
    plain_hard = AgentLedgerState(**without_model_semantics["agent_ledger"]).goal.hard_constraints
    assert plain_hard["intent_kind"] == "itinerary"
    assert "transport_modes_requested" not in plain_hard

    with_model_semantics = initialize_agent_ledger(
        {
            "user_input": "表达方式完全可以不含固定关键词",
            "slots": {
                "destination": "上海",
                "travel_days": 2,
                "intent_kind": "event_trip",
                "event_query": "周杰伦上海站",
                "transport_modes_requested": ["train"],
                "information_needs": ["event", "transport"],
            },
        },
        mode="agent",
    )
    model_hard = AgentLedgerState(**with_model_semantics["agent_ledger"]).goal.hard_constraints
    assert model_hard["intent_kind"] == "event_trip"
    assert model_hard["event_query"] == "周杰伦上海站"
    assert model_hard["transport_modes_requested"] == ["train"]


def test_single_profile_date_derives_end_from_trip_duration():
    result = initialize_agent_ledger(
        {
            "profile": {
                "trip": {
                    "destination": "苏州",
                    "travel_days": 3,
                    "travel_dates": "2026-12-08",
                }
            },
            "missing_slots": [],
        },
        mode="agent",
    )
    hard = AgentLedgerState(**result["agent_ledger"]).goal.hard_constraints

    assert hard["start_date"] == "2026-12-08"
    assert hard["end_date"] == "2026-12-10"


def test_existing_ledger_is_resumed_instead_of_reset():
    initial = initialize_agent_ledger(
        {"user_input": "Plan Shanghai", "slots": {"destination": "Shanghai"}},
        mode="agent",
    )
    ledger = AgentLedgerState(**initial["agent_ledger"])
    ledger.budget = ledger.budget.consume(episode_steps=2)

    resumed = initialize_agent_ledger(
        {"user_input": "ignored", "agent_ledger": ledger.model_dump(mode="json")},
        mode="agent",
    )

    assert AgentLedgerState(**resumed["agent_ledger"]).budget.used_episode_steps == 2


def test_material_goal_change_starts_new_version_without_stale_artifacts():
    initial = initialize_agent_ledger(
        {
            "user_input": "南京三日游",
            "slots": {
                "destination": "南京",
                "travel_days": 3,
                "start_date": "2026-12-08",
                "end_date": "2026-12-10",
                "interests": ["历史文化"],
            },
        },
        mode="agent",
    )
    old = AgentLedgerState(**initial["agent_ledger"])
    old.budget = old.budget.consume(episode_steps=3, tool_calls=2)

    revised = initialize_agent_ledger(
        {
            "user_input": "改到明年一月",
            "agent_ledger": old.model_dump(mode="json"),
            "slots": {
                "destination": "南京",
                "travel_days": 3,
                "start_date": "2027-01-12",
                "end_date": "2027-01-14",
                "interests": ["历史文化"],
            },
        },
        mode="agent",
    )
    current = AgentLedgerState(**revised["agent_ledger"])

    assert current.goal.goal_version == old.goal.goal_version + 1
    assert current.task_graph.plan_version == old.task_graph.plan_version + 1
    assert current.goal.hard_constraints["start_date"] == "2027-01-12"
    assert current.trajectory_id != old.trajectory_id
    assert current.budget.used_episode_steps == 0
    assert current.budget.used_tool_calls == 0
    assert current.facts == {}
    assert current.artifacts == {}
