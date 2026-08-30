from __future__ import annotations

import json
from datetime import UTC, datetime

from agentic.action_executor import TravelActionExecutor
from agentic.loop import BoundedAgentLoop, PolicyAction, PolicyContext
from agentic.observations import ObservationEnvelope
from agentic.react import ReactTaskGraphPlanner, ResearchSufficiencyVerifier
from agentic.runtime import resume_agent_ledger, revise_agent_ledger
from agentic.state import AgentLedgerState, ArtifactRecord, GoalLedger, TaskGraphController
from models.travel_slots import RevisionOperation, RevisionParseOutput
from vrp_solver_service.models import SolverRequest
from vrp_solver_service.solver import TravelVRPSolver
from evaluation.validator import ItineraryValidator
from schemas import ToolResult


def _goal(request: str = "去上海玩2天") -> GoalLedger:
    return GoalLedger(
        original_request=request,
        hard_constraints={"destination": "上海", "travel_days": 2},
    )


def _ledger(goal: GoalLedger | None = None) -> AgentLedgerState:
    selected = goal or _goal()
    graph = ReactTaskGraphPlanner().plan(selected)
    graph = TaskGraphController().refresh_ready(graph)
    return AgentLedgerState(goal=selected, task_graph=graph)


def _artifact(ledger: AgentLedgerState, artifact_type: str, payload: dict) -> ArtifactRecord:
    return ArtifactRecord(
        artifact_id=f"a:{artifact_type}",
        artifact_type=artifact_type,
        payload=payload,
        goal_version=ledger.goal.goal_version,
        plan_version=ledger.task_graph.plan_version,
    )


def _complete_research(ledger: AgentLedgerState) -> None:
    pois = [{"id": f"p{i}", "name": f"POI {i}", "category": "attraction"} for i in range(4)]
    for artifact in (
        _artifact(ledger, "city_knowledge", {"city": "上海"}),
        _artifact(ledger, "poi_candidate_set", {"pois": pois}),
        _artifact(
            ledger,
            "poi_detail_set",
            {"details": [{"name": f"POI {i}"} for i in range(4)]},
        ),
        _artifact(
            ledger,
            "route_matrix",
            {
                "time_minutes": [
                    [0 if row == column else 10 for column in range(5)] for row in range(5)
                ]
            },
        ),
    ):
        ledger.artifacts[artifact.artifact_id] = artifact


def test_react_graph_has_one_dynamic_research_phase_before_hard_gates():
    graph = ReactTaskGraphPlanner().plan(_goal())
    assert [task.task_id for task in graph.tasks] == [
        "research_evidence",
        "solve_itinerary",
        "validate_itinerary",
        "review_itinerary",
        "compose_draft",
        "await_confirmation",
    ]
    research = graph.get("research_evidence")
    assert {
        "retrieve_city_knowledge",
        "search_pois",
        "search_current_info",
        "search_transport",
        "get_route_matrix",
        "finalize_research",
    } <= set(research.allowed_actions)
    assert research.success_criteria["research_required_artifact_types"] == [
        "city_knowledge",
        "poi_candidate_set",
        "poi_detail_set",
        "route_matrix",
    ]


def test_research_verifier_returns_actionable_evidence_gaps():
    ledger = _ledger()
    report = ResearchSufficiencyVerifier().evaluate(ledger)
    assert report.sufficient is False
    assert "MISSING_ARTIFACT:city_knowledge" in report.missing
    assert "MISSING_ARTIFACT:route_matrix" in report.missing

    _complete_research(ledger)
    report = ResearchSufficiencyVerifier().evaluate(ledger)
    assert report.sufficient is True


def test_event_and_transport_intent_add_only_relevant_live_requirements():
    keyword_only = GoalLedger(
        original_request="从济南坐高铁去上海看演唱会，玩2天",
        hard_constraints={"destination": "上海", "travel_days": 2},
    )
    keyword_only_required = set(
        ResearchSufficiencyVerifier()
        .evaluate(_ledger(keyword_only))
        .requirements.required_artifact_types
    )
    assert "event_search_result" not in keyword_only_required
    assert "transport_search_result" not in keyword_only_required

    goal = GoalLedger(
        original_request="从济南坐高铁去上海看演唱会，玩2天",
        hard_constraints={
            "origin": "济南",
            "destination": "上海",
            "travel_days": 2,
            "intent_kind": "event_trip",
            "transport_modes_requested": ["train"],
        },
    )
    report = ResearchSufficiencyVerifier().evaluate(_ledger(goal))
    required = set(report.requirements.required_artifact_types)
    assert "event_search_result" in required
    assert "transport_search_result" in required
    assert "weather_snapshot" not in required


def test_event_evidence_requires_source_and_grounded_venue_coordinates():
    goal = GoalLedger(
        original_request="去上海看演唱会，玩2天",
        hard_constraints={
            "destination": "上海",
            "travel_days": 2,
            "intent_kind": "event_trip",
        },
    )
    ledger = _ledger(goal)
    _complete_research(ledger)
    ledger.artifacts["event"] = _artifact(
        ledger,
        "event_search_result",
        {
            "event": {
                "complete": True,
                "date": "2026-09-01",
                "start_time": "19:00",
                "venue": "演出场馆",
                "lat": 0,
                "lng": 0,
            },
            "results": [],
            "queried_at": "2026-08-29T04:00:00+00:00",
        },
    )

    report = ResearchSufficiencyVerifier().evaluate(ledger)

    assert "EVENT_VENUE_UNGROUNDED:lat,lng" in report.missing
    assert "EVENT_SOURCE_MISSING" in report.missing


def test_weather_fallback_cannot_satisfy_live_evidence_gate():
    goal = GoalLedger(
        original_request="查天气并规划上海2日游",
        hard_constraints={"destination": "上海", "travel_days": 2},
    )
    ledger = _ledger(goal)
    _complete_research(ledger)
    ledger.artifacts["weather"] = _artifact(
        ledger,
        "weather_snapshot",
        {
            "days": [{"date": "2026-09-02", "condition": "estimated"}],
            "queried_at": datetime.now(UTC).isoformat(),
            "_evidence_source": "fallback",
            "_is_fallback": True,
        },
    )

    report = ResearchSufficiencyVerifier().evaluate(ledger)

    assert "UNVERIFIED_LIVE_EVIDENCE:weather_snapshot" in report.missing


def test_transport_origin_is_clarified_before_research_and_then_resumed():
    goal = GoalLedger(
        original_request="查一下去上海的高铁并规划2天",
        hard_constraints={"destination": "上海", "travel_days": 2},
        missing_information=["origin"],
    )
    ledger = _ledger(goal)
    controller = TaskGraphController()
    assert ledger.task_graph.get("clarify_user_constraints").status == "ready"
    ledger.task_graph = controller.transition(
        ledger.task_graph, "clarify_user_constraints", "running"
    )
    ledger.task_graph = controller.transition(
        ledger.task_graph, "clarify_user_constraints", "blocked"
    )

    resumed = resume_agent_ledger(
        ledger,
        task_id="clarify_user_constraints",
        user_value="济南",
        fact_key="user_input.origin",
    )
    assert resumed.goal.hard_constraints["origin"] == "济南"
    assert resumed.task_graph.get("clarify_user_constraints").status == "succeeded"
    assert resumed.task_graph.get("research_evidence").status == "ready"


def test_transport_evidence_must_produce_applied_planning_boundaries():
    goal = GoalLedger(
        original_request="从济南坐高铁去上海玩2天",
        hard_constraints={
            "origin": "济南",
            "destination": "上海",
            "travel_days": 2,
            "transport_modes_requested": ["train"],
        },
    )
    ledger = _ledger(goal)
    _complete_research(ledger)
    ledger.artifacts["transport"] = _artifact(
        ledger,
        "transport_search_result",
        {
            "results": [{"url": "https://rail.example/schedule"}],
            "queried_at": datetime.now(UTC).isoformat(),
            "planning_constraints": {"applied": False},
        },
    )

    rejected = ResearchSufficiencyVerifier().evaluate(ledger)
    assert "TRANSPORT_SCHEDULE_NOT_PLANNABLE" in rejected.missing

    ledger.artifacts["transport"].payload["planning_constraints"] = {
        "applied": True,
        "daily_start_minutes": [750, 480],
        "daily_end_minutes": [1260, 1260],
    }
    accepted = ResearchSufficiencyVerifier().evaluate(ledger)
    assert accepted.sufficient is True


def test_opening_hours_evidence_must_match_a_candidate_and_contain_a_constraint():
    ledger = _ledger(_goal("规划上海2天并核实POI 0营业时间"))
    _complete_research(ledger)
    ledger.artifacts["current"] = _artifact(
        ledger,
        "current_info_search",
        {
            "info_type": "opening_hours",
            "results": [
                {
                    "title": "其他景点介绍",
                    "snippet": "欢迎参观",
                    "url": "https://official.example/other",
                }
            ],
            "queried_at": datetime.now(UTC).isoformat(),
        },
    )

    rejected = ResearchSufficiencyVerifier().evaluate(ledger)
    assert "CURRENT_INFO_NOT_PLANNABLE" in rejected.missing

    ledger.artifacts["current"].payload["results"] = [
        {
            "title": "POI 0 官方营业时间",
            "snippet": "开放时间09:00-17:00",
            "url": "https://official.example/poi-0",
        }
    ]
    accepted = ResearchSufficiencyVerifier().evaluate(ledger)
    assert accepted.sufficient is True


async def test_finalize_research_is_a_verifier_gate_not_a_model_claim():
    ledger = _ledger()
    task = ledger.task_graph.get("research_evidence")
    executor = TravelActionExecutor()
    rejected = await executor.execute(
        task=task,
        action=PolicyAction(action="finalize_research"),
        ledger=ledger,
    )
    assert rejected.status == "failed"
    assert rejected.error_code == "RESEARCH_EVIDENCE_INSUFFICIENT"

    _complete_research(ledger)
    accepted = await executor.execute(
        task=task,
        action=PolicyAction(action="finalize_research"),
        ledger=ledger,
    )
    assert accepted.status == "completed"
    assert accepted.artifacts[0].artifact_type == "research_bundle"


def test_fixed_event_is_enforced_by_cpsat():
    request = SolverRequest(
        pois=[
            {
                "id": "event:concert",
                "name": "Concert Hall",
                "category": "event",
                "lat": 31.2,
                "lng": 121.4,
                "duration_minutes": 120,
                "open_time": "19:00",
                "close_time": "22:00",
                "must_visit": True,
            },
            {
                "id": "museum",
                "name": "Museum",
                "category": "attraction",
                "lat": 31.21,
                "lng": 121.41,
                "duration_minutes": 120,
                "open_time": "09:00",
                "close_time": "17:00",
            },
        ],
        constraints={
            "travel_days": 1,
            "trip_start_date": "2026-09-01",
            "day_end_min": 22 * 60,
            "must_visit": ["event:concert"],
            "user_reservations": [
                {
                    "poi_id": "event:concert",
                    "date": "2026-09-01",
                    "start_time": "19:00",
                    "end_time": "21:00",
                }
            ],
        },
        strategy="cpsat",
    )
    result = TravelVRPSolver().solve(request)
    concert = next(
        activity
        for day in result.days
        for activity in day.activities
        if activity.poi_id == "event:concert"
    )
    assert concert.start_time == "19:00"
    assert concert.end_time == "21:00"


async def test_user_rejection_creates_new_goal_and_plan_versions():
    ledger = _ledger()
    interpretation = RevisionParseOutput(
        operations=[
            RevisionOperation(field="travel_days", operation="set", value=3),
            RevisionOperation(field="pace", operation="set", value="relaxed"),
        ],
        affected_domains=["schedule"],
        confidence=0.94,
    )
    revised = await revise_agent_ledger(
        ledger,
        revision_reason="上一版太赶，改成3天并轻松一点",
        interpretation=interpretation,
    )
    assert revised.trajectory_id == ledger.trajectory_id
    assert revised.goal.goal_version == ledger.goal.goal_version + 1
    assert revised.task_graph.plan_version == ledger.task_graph.plan_version + 1
    assert revised.goal.hard_constraints["travel_days"] == 3
    assert revised.goal.soft_preferences["pace"] == "relaxed"
    assert revised.task_graph.get("research_evidence").status == "ready"


async def test_revision_does_not_guess_constraints_when_intent_model_is_down(monkeypatch):
    ledger = _ledger()

    async def fail(*args, **kwargs):
        raise RuntimeError("intent model unavailable")

    monkeypatch.setattr("agents.demand_parser.DemandParserAgent.parse_revision", fail)
    revised = await revise_agent_ledger(
        ledger,
        revision_reason="上一版太赶，改成9天并轻松一点",
    )

    assert revised.goal.hard_constraints["travel_days"] == 2
    assert "pace" not in revised.goal.soft_preferences
    assert revised.goal.soft_preferences["revision_feedback"] == ["上一版太赶，改成9天并轻松一点"]
    assert revised.goal.capability.status == "needs_user"


async def test_low_confidence_revision_is_clarified_not_applied():
    ledger = _ledger()
    revised = await revise_agent_ledger(
        ledger,
        revision_reason="那个还是改一下吧",
        interpretation=RevisionParseOutput(
            confidence=0.2,
            operations=[RevisionOperation(field="travel_days", operation="set", value=9)],
            affected_domains=["schedule"],
        ),
    )

    assert revised.goal.hard_constraints["travel_days"] == 2
    assert revised.goal.missing_information == ["revision_clarification"]
    assert revised.task_graph.get("clarify_user_constraints").status == "ready"


async def test_revision_normalizes_model_budget_bound_object():
    revised = await revise_agent_ledger(
        _ledger(),
        revision_reason="预算控制在4500以内",
        interpretation=RevisionParseOutput(
            confidence=0.95,
            operations=[
                RevisionOperation(
                    field="budget_range",
                    operation="set",
                    value={"max": 4500},
                )
            ],
            affected_domains=["budget"],
        ),
    )

    assert revised.goal.hard_constraints["budget_range"] == 4500.0
    assert revised.goal.soft_preferences["revision_parse"]["rejected_operations"] == []


async def test_react_loop_observes_and_selects_multiple_tools_before_cpsat():
    class EvidenceDirectedPolicy:
        def __init__(self) -> None:
            self.actions: list[str] = []

        async def propose(self, context: PolicyContext) -> PolicyAction:
            task_id = context.current_subtask["task_id"]
            types = {item["artifact_type"] for item in context.relevant_artifacts}
            if task_id == "research_evidence":
                if "city_knowledge" not in types:
                    action = "retrieve_city_knowledge"
                elif "poi_candidate_set" not in types:
                    action = "search_pois"
                elif "poi_detail_set" not in types:
                    action = "get_poi_detail"
                elif "route_matrix" not in types:
                    action = "get_route_matrix"
                else:
                    action = "finalize_research"
            elif task_id == "solve_itinerary":
                action = "solve_itinerary"
            elif task_id == "validate_itinerary":
                action = "validate_itinerary"
            elif task_id == "review_itinerary":
                action = "accept_itinerary"
            elif task_id == "compose_draft":
                action = "compose_draft"
            else:
                action = "finish"
            self.actions.append(action)
            return PolicyAction(action=action)

    class FakeTools:
        async def execute(self, calls, guard_context=None):
            records = []
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])
                if name == "retrieve_city_knowledge":
                    data = {"city": "上海", "pois": [{"name": "外滩"}]}
                elif name == "search_pois":
                    data = [
                        {
                            "id": f"p{i}",
                            "name": f"POI {i}",
                            "category": "attraction",
                            "lat": 31.2 + i / 100,
                            "lng": 121.4 + i / 100,
                            "open_time": "09:00",
                            "close_time": "18:00",
                            "duration_minutes": 90,
                        }
                        for i in range(4)
                    ]
                elif name == "get_poi_detail":
                    data = {
                        "name": args["poi_name"],
                        "open_hours": "09:00-18:00",
                        "suggested_hours": 1.5,
                    }
                elif name == "get_route_matrix":
                    size = len(args["pois"]) + 1
                    data = {
                        "poi_ids": ["hotel", *[item["id"] for item in args["pois"]]],
                        "time_minutes": [
                            [0 if i == j else 15 for j in range(size)] for i in range(size)
                        ],
                        "transport_cost": [
                            [0.0 if i == j else 5.0 for j in range(size)] for i in range(size)
                        ],
                    }
                elif name == "solve_itinerary":
                    assert args["strategy"] == "cpsat"
                    data = {
                        "status": "optimal",
                        "days": [
                            {
                                "day_number": 1,
                                "activities": [
                                    {
                                        "poi_id": "p0",
                                        "poi_name": "POI 0",
                                        "category": "attraction",
                                        "start_time": "09:00",
                                        "end_time": "10:30",
                                        "duration_min": 90,
                                    }
                                ],
                            },
                            {
                                "day_number": 2,
                                "activities": [
                                    {
                                        "poi_id": "p1",
                                        "poi_name": "POI 1",
                                        "category": "attraction",
                                        "start_time": "09:00",
                                        "end_time": "10:30",
                                        "duration_min": 90,
                                    }
                                ],
                            },
                        ],
                        "reminders": [],
                        "solve_time_ms": 5,
                    }
                elif name == "validate_itinerary":
                    data = (
                        ItineraryValidator()
                        .validate(args["itinerary"], args["constraints"], args["facts"])
                        .model_dump(mode="json")
                    )
                else:
                    raise AssertionError(name)
                result = ToolResult(data=data, data_source="built_in", confidence=1.0)
                observation = ObservationEnvelope.from_tool_result(
                    tool=name,
                    result=result,
                    tool_call_id=call["id"],
                )
                records.append({"observation": observation.model_dump(mode="json")})
            return records

    ledger = _ledger()
    policy = EvidenceDirectedPolicy()
    result = await BoundedAgentLoop().run(
        ledger,
        policy=policy,
        executor=TravelActionExecutor(tool_executor=FakeTools()),
    )

    assert result.status == "interrupted"
    assert result.termination_reason == "awaiting_user"
    assert policy.actions[:5] == [
        "retrieve_city_knowledge",
        "search_pois",
        "get_poi_detail",
        "get_route_matrix",
        "finalize_research",
    ]
    assert "solve_itinerary" in policy.actions
    assert "validate_itinerary" in policy.actions
    assert result.ledger.task_graph.get("review_itinerary").status == "succeeded"
