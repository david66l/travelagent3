"""Tests for legacy-to-Agent-Loop state projection."""

import pytest

from agentic.runtime import confirm_agent_ledger, initialize_agent_ledger, resume_agent_ledger
from agentic.state import AgentLedgerState, StateTransitionError, TaskGraphController


def test_deterministic_mode_does_not_create_agent_state():
    result = initialize_agent_ledger({"user_input": "Plan Shanghai"}, mode="deterministic")

    assert result == {"policy_mode": "deterministic", "agent_status": "disabled"}


def test_shadow_mode_creates_versioned_goal_and_default_dag():
    result = initialize_agent_ledger(
        {
            "user_input": "Shanghai for three days under 5000",
            "slots": {
                "destination": "Shanghai",
                "travel_days": 3,
                "budget_range": 5000,
                "interests": ["museum"],
            },
            "missing_slots": [],
        },
        mode="shadow",
    )
    ledger = AgentLedgerState(**result["agent_ledger"])

    assert result["agent_status"] == "initialized"
    assert result["current_task_id"] == "capability_check"
    assert ledger.goal.hard_constraints["destination"] == "Shanghai"
    assert ledger.goal.soft_preferences["interests"] == ["museum"]
    assert ledger.task_graph.get("capability_check").status == "ready"


def test_explicit_react_graph_mode_does_not_depend_on_environment_default():
    result = initialize_agent_ledger(
        {
            "user_input": "上海两日游",
            "slots": {"destination": "上海", "travel_days": 2},
            "missing_slots": [],
        },
        mode="agent",
        task_graph_mode="react",
    )
    ledger = AgentLedgerState(**result["agent_ledger"])

    assert result["current_task_id"] == "research_evidence"
    assert [task.task_id for task in ledger.task_graph.tasks] == [
        "research_evidence",
        "solve_itinerary",
        "validate_itinerary",
        "review_itinerary",
        "compose_draft",
        "await_confirmation",
    ]


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
        mode="shadow",
    )
    ledger = AgentLedgerState(**initial["agent_ledger"])
    ledger.budget = ledger.budget.consume(episode_steps=2)

    resumed = initialize_agent_ledger(
        {"user_input": "ignored", "agent_ledger": ledger.model_dump(mode="json")},
        mode="shadow",
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


def test_user_response_resumes_blocked_task_and_preserves_budget():
    initial = initialize_agent_ledger(
        {
            "user_input": "Plan my trip",
            "missing_slots": ["destination", "travel_dates"],
        },
        mode="shadow",
    )
    ledger = AgentLedgerState(**initial["agent_ledger"])
    controller = TaskGraphController()
    ledger.task_graph = controller.transition(ledger.task_graph, "capability_check", "running")
    ledger.task_graph = controller.transition(
        ledger.task_graph,
        "capability_check",
        "succeeded",
        evidence_refs=["capability-1"],
    )
    ledger.task_graph = controller.refresh_ready(ledger.task_graph)
    ledger.task_graph = controller.transition(
        ledger.task_graph, "resolve_missing_information", "running"
    )
    ledger.task_graph = controller.transition(
        ledger.task_graph, "resolve_missing_information", "blocked"
    )
    ledger.budget = ledger.budget.consume(episode_steps=2)

    resumed = resume_agent_ledger(
        ledger,
        task_id="resolve_missing_information",
        user_value="Shanghai",
        fact_key="user_input.destination",
    )

    assert resumed.task_graph.get("resolve_missing_information").status == "blocked"
    assert resumed.budget.used_episode_steps == 2
    assert next(iter(resumed.facts.values())).value == "Shanghai"


def test_complete_user_response_is_verified_and_unlocks_dependents():
    initial = initialize_agent_ledger(
        {"user_input": "Plan my trip", "missing_slots": ["destination"]},
        mode="shadow",
    )
    ledger = AgentLedgerState(**initial["agent_ledger"])
    controller = TaskGraphController()
    ledger.task_graph = controller.transition(ledger.task_graph, "capability_check", "running")
    ledger.task_graph = controller.transition(
        ledger.task_graph,
        "capability_check",
        "succeeded",
        evidence_refs=["capability-1"],
    )
    ledger.task_graph = controller.refresh_ready(ledger.task_graph)
    ledger.task_graph = controller.transition(
        ledger.task_graph, "resolve_missing_information", "running"
    )
    ledger.task_graph = controller.transition(
        ledger.task_graph, "resolve_missing_information", "blocked"
    )

    resumed = resume_agent_ledger(
        ledger,
        task_id="resolve_missing_information",
        user_value="Shanghai",
        fact_key="user_input.destination",
    )

    assert resumed.task_graph.get("resolve_missing_information").status == "succeeded"
    assert resumed.task_graph.get("collect_weather").status == "ready"


def test_user_response_cannot_resume_non_blocked_task():
    initial = initialize_agent_ledger({"user_input": "Plan Shanghai"}, mode="shadow")
    with pytest.raises(StateTransitionError, match="blocked"):
        resume_agent_ledger(
            initial["agent_ledger"],
            task_id="capability_check",
            user_value="x",
            fact_key="x",
        )


def test_confirmation_closes_task_graph_and_passes_global_guard():
    from agentic.state import ArtifactRecord, GoalLedger, TaskGraph, TaskNode

    ledger = AgentLedgerState(
        goal=GoalLedger(original_request="Plan Shanghai"),
        task_graph=TaskGraph(
            goal_version=1,
            tasks=(
                TaskNode(
                    task_id="await_confirmation",
                    goal="confirm",
                    status="blocked",
                    allowed_actions=("ask_user", "finish"),
                    success_criteria={"required_fact_keys": ["user_confirmation"]},
                    attempts=1,
                ),
            ),
        ),
    )
    ledger.artifacts["validation"] = ArtifactRecord(
        artifact_id="validation",
        artifact_type="validation_report",
        payload={"hard_pass": True, "hard_violations": []},
        goal_version=1,
        plan_version=1,
    )

    confirmed, decision = confirm_agent_ledger(ledger)

    assert confirmed.task_graph.get("await_confirmation").status == "succeeded"
    assert confirmed.termination_reason == "validated_finish"
    assert decision.allowed is True
