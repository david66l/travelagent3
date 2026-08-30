"""Intent-driven ReAct research loop with deterministic planning gates.

The task graph contains only phase boundaries.  Tool order inside
``research_evidence`` is chosen by the policy from live observations; the
controller merely verifies that the resulting evidence bundle is sufficient
before CP-SAT is allowed to run.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, date, datetime, timedelta
from pydantic import BaseModel, Field

from agentic.state import AgentLedgerState, ArtifactRecord, GoalLedger, TaskGraph, TaskNode


def _entity_key(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", normalized)


RESEARCH_ACTIONS = (
    "retrieve_city_knowledge",
    "search_pois",
    "get_poi_detail",
    "get_weather",
    "search_current_info",
    "search_transport",
    "get_route_matrix",
)


class ResearchRequirements(BaseModel):
    intent_kind: str = "itinerary"
    required_artifact_types: list[str] = Field(default_factory=list)
    min_candidate_count: int = 4
    min_detail_count: int = 3
    requires_weather: bool = False
    requires_event: bool = False
    requires_transport: bool = False
    requires_current_info: bool = False


class ResearchSufficiencyReport(BaseModel):
    sufficient: bool
    requirements: ResearchRequirements
    missing: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    coverage: dict[str, bool] = Field(default_factory=dict)


def infer_research_requirements(goal: GoalLedger) -> ResearchRequirements:
    hard = goal.hard_constraints
    days = max(1, int(hard.get("travel_days") or 1))
    intent_kind = str(hard.get("intent_kind") or "itinerary")
    information_needs = set(hard.get("information_needs") or [])
    requires_event = intent_kind == "event_trip" or "event" in information_needs
    modes = list(hard.get("transport_modes_requested") or [])
    requires_transport = bool(modes) or "transport" in information_needs
    requires_current_info = bool(
        information_needs
        & {"opening_hours", "closure", "restaurant", "seasonal_activity", "general"}
    )
    requires_weather = "weather" in information_needs
    start_date = hard.get("start_date")
    if start_date:
        try:
            delta = (date.fromisoformat(str(start_date)) - date.today()).days
            requires_weather = requires_weather or 0 <= delta <= 10
        except ValueError:
            pass

    required = ["city_knowledge", "poi_candidate_set", "poi_detail_set", "route_matrix"]
    if requires_weather:
        required.append("weather_snapshot")
    if requires_event:
        required.append("event_search_result")
    if requires_transport:
        required.append("transport_search_result")
    if requires_current_info:
        required.append("current_info_search")
    return ResearchRequirements(
        intent_kind=intent_kind,
        required_artifact_types=required,
        min_candidate_count=max(4, min(10, days * 2)),
        min_detail_count=max(3, min(8, days * 2)),
        requires_weather=requires_weather,
        requires_event=requires_event,
        requires_transport=requires_transport,
        requires_current_info=requires_current_info,
    )


class ResearchSufficiencyVerifier:
    """Programmatically reject premature ``finalize_research`` proposals."""

    def evaluate(self, ledger: AgentLedgerState) -> ResearchSufficiencyReport:
        requirements = infer_research_requirements(ledger.goal)
        current = [
            artifact
            for artifact in ledger.artifacts.values()
            if artifact.goal_version == ledger.goal.goal_version
            and artifact.plan_version == ledger.task_graph.plan_version
        ]
        by_type: dict[str, list[ArtifactRecord]] = {}
        for artifact in current:
            by_type.setdefault(artifact.artifact_type, []).append(artifact)

        missing: list[str] = []
        coverage: dict[str, bool] = {}
        evidence_refs: list[str] = []
        for artifact_type in requirements.required_artifact_types:
            matches = by_type.get(artifact_type, [])
            covered = bool(matches)
            coverage[artifact_type] = covered
            if not covered:
                missing.append(f"MISSING_ARTIFACT:{artifact_type}")
            else:
                evidence_refs.append(matches[-1].artifact_id)

        freshness_hours = {
            "weather_snapshot": 3,
            "current_info_search": 6,
            "event_search_result": 24,
            "transport_search_result": 2,
        }
        now = datetime.now(UTC)
        for artifact_type, ttl_hours in freshness_hours.items():
            matches = by_type.get(artifact_type) or []
            if not matches:
                continue
            payload = matches[-1].payload
            verified_live_source = (
                payload.get("_evidence_source") not in {"fallback", "unavailable"}
                and payload.get("_is_fallback") is not True
            )
            coverage[f"verified_live_source:{artifact_type}"] = verified_live_source
            if not verified_live_source:
                missing.append(f"UNVERIFIED_LIVE_EVIDENCE:{artifact_type}")
            raw_timestamp = payload.get("queried_at") or payload.get("retrieved_at")
            try:
                queried_at = datetime.fromisoformat(str(raw_timestamp))
                if queried_at.tzinfo is None:
                    queried_at = queried_at.replace(tzinfo=UTC)
                fresh = queried_at > now - timedelta(hours=ttl_hours)
            except (TypeError, ValueError):
                fresh = False
            coverage[f"fresh:{artifact_type}"] = fresh
            if not fresh:
                missing.append(f"STALE_OR_UNTIMED_ARTIFACT:{artifact_type}")

        candidates = (by_type.get("poi_candidate_set") or [None])[-1]
        candidate_items = candidates.payload.get("pois", []) if candidates else []
        plannable_candidates = [
            item
            for item in candidate_items
            if isinstance(item, dict)
            and str(item.get("category") or "attraction").lower()
            not in {"restaurant", "meal", "hotel", "transport"}
        ]
        if len(plannable_candidates) < requirements.min_candidate_count:
            coverage["candidate_count"] = False
            missing.append(
                "INSUFFICIENT_CANDIDATES:"
                f"{len(plannable_candidates)}/{requirements.min_candidate_count}"
            )
        else:
            coverage["candidate_count"] = True

        details = (by_type.get("poi_detail_set") or [None])[-1]
        detail_items = details.payload.get("details", []) if details else []
        if len(detail_items) < requirements.min_detail_count:
            coverage["detail_count"] = False
            missing.append(
                f"INSUFFICIENT_POI_DETAILS:{len(detail_items)}/{requirements.min_detail_count}"
            )
        else:
            coverage["detail_count"] = True

        event = (by_type.get("event_search_result") or [None])[-1]
        if requirements.requires_event and event is not None:
            event_fields = event.payload.get("event") or {}
            complete = bool(event_fields.get("complete"))
            coverage["event_fields_complete"] = complete
            if not complete:
                missing.append("EVENT_FIELDS_INCOMPLETE:date,start_time,venue")
            venue_grounded = bool(event_fields.get("lat") and event_fields.get("lng"))
            coverage["event_venue_grounded"] = venue_grounded
            if not venue_grounded:
                missing.append("EVENT_VENUE_UNGROUNDED:lat,lng")
            source_backed = any(
                isinstance(item, dict) and item.get("url")
                for item in event.payload.get("results") or []
            )
            coverage["event_source_backed"] = source_backed
            if not source_backed:
                missing.append("EVENT_SOURCE_MISSING")

        for artifact_type in ("current_info_search", "transport_search_result"):
            matches = by_type.get(artifact_type) or []
            if not matches:
                continue
            source_backed = any(
                isinstance(item, dict) and item.get("url")
                for item in matches[-1].payload.get("results") or []
            )
            coverage[f"source_backed:{artifact_type}"] = source_backed
            if not source_backed:
                missing.append(f"SOURCE_MISSING:{artifact_type}")
            if artifact_type == "transport_search_result":
                planning_constraints = matches[-1].payload.get("planning_constraints") or {}
                applied = bool(planning_constraints.get("applied"))
                coverage["transport_constraints_applied"] = applied
                if not applied:
                    missing.append("TRANSPORT_SCHEDULE_NOT_PLANNABLE")
            elif str(matches[-1].payload.get("info_type") or "") in {
                "opening_hours",
                "closure",
            }:
                candidate_names = [
                    _entity_key(item.get("name"))
                    for item in candidate_items
                    if isinstance(item, dict) and item.get("name")
                ]
                constraint_bearing = False
                for item in matches[-1].payload.get("results") or []:
                    if not isinstance(item, dict) or not item.get("url"):
                        continue
                    corpus = f"{item.get('title') or ''} {item.get('snippet') or ''}"
                    entity_matched = any(
                        name and name in _entity_key(corpus) for name in candidate_names
                    )
                    has_hours = len(re.findall(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d", corpus)) >= 2
                    has_closure = any(
                        token in corpus
                        for token in ("临时闭馆", "暂停开放", "闭馆", "停业", "关闭")
                    )
                    if entity_matched and (has_hours or has_closure):
                        constraint_bearing = True
                        break
                coverage["current_info_constraints_applicable"] = constraint_bearing
                if not constraint_bearing:
                    missing.append("CURRENT_INFO_NOT_PLANNABLE")

        matrix = (by_type.get("route_matrix") or [None])[-1]
        if matrix is not None:
            rows = matrix.payload.get("time_minutes") or []
            expected_detail_count = int(
                (details.payload.get("expected_count") if details else 0) or len(detail_items)
            )
            expected_matrix_size = min(len(plannable_candidates), expected_detail_count) + 1
            valid_matrix = (
                len(rows) == expected_matrix_size
                and expected_matrix_size >= 2
                and all(len(row) == expected_matrix_size for row in rows)
            )
            coverage["route_matrix_shape"] = valid_matrix
            if not valid_matrix:
                missing.append(
                    f"INVALID_ROUTE_MATRIX:{len(rows)}x?/{expected_matrix_size}x"
                    f"{expected_matrix_size}"
                )

        return ResearchSufficiencyReport(
            sufficient=not missing,
            requirements=requirements,
            missing=missing,
            evidence_refs=list(dict.fromkeys(evidence_refs)),
            coverage=coverage,
        )


class ReactTaskGraphPlanner:
    """Build phase gates while leaving research/recovery decisions to ReAct."""

    def plan(self, goal: GoalLedger, *, plan_version: int = 1) -> TaskGraph:
        requirements = infer_research_requirements(goal)
        research_actions = (
            *RESEARCH_ACTIONS,
            "finalize_research",
            "ask_user",
            "propose_tradeoff",
            "abort",
        )
        tasks: list[TaskNode] = []
        research_dependencies: tuple[str, ...] = ()
        if goal.missing_information:
            tasks.append(
                TaskNode(
                    task_id="clarify_user_constraints",
                    goal="Ask only for required information that cannot be discovered by tools",
                    allowed_actions=("ask_user",),
                    success_criteria={
                        "required_fact_keys": [
                            f"user_input.{field}" for field in goal.missing_information
                        ]
                    },
                    max_attempts=max(2, len(goal.missing_information) + 1),
                    invalidates_on=("goal_changed",),
                )
            )
            research_dependencies = ("clarify_user_constraints",)
        tasks.extend(
            (
                TaskNode(
                    task_id="research_evidence",
                    goal=(
                        "Dynamically gather source-backed evidence, observe each result, and only "
                        "finalize when the intent-specific evidence verifier can pass"
                    ),
                    depends_on=research_dependencies,
                    allowed_actions=research_actions,
                    success_criteria={
                        "required_artifact_types": ["research_bundle"],
                        "research_required_artifact_types": (requirements.required_artifact_types),
                    },
                    max_attempts=14,
                    invalidates_on=("goal_changed", "planning_fact_changed"),
                ),
                TaskNode(
                    task_id="solve_itinerary",
                    goal="Run CP-SAT over the verified evidence bundle and fixed-event constraints",
                    depends_on=("research_evidence",),
                    allowed_actions=("solve_itinerary",),
                    success_criteria={"required_artifact_types": ["solver_result"]},
                    max_attempts=3,
                    invalidates_on=("goal_changed", "planning_fact_changed"),
                ),
                TaskNode(
                    task_id="validate_itinerary",
                    goal="Programmatically validate all hard constraints",
                    depends_on=("solve_itinerary",),
                    allowed_actions=("validate_itinerary",),
                    success_criteria={"required_artifact_types": ["validation_report"]},
                    max_attempts=3,
                    invalidates_on=("goal_changed", "solver_result_changed"),
                ),
                TaskNode(
                    task_id="review_itinerary",
                    goal=(
                        "Use verifier observations to accept, repair with more evidence, retry CP-SAT, "
                        "ask the user for a tradeoff, or stop safely"
                    ),
                    depends_on=("validate_itinerary",),
                    allowed_actions=(
                        "accept_itinerary",
                        "retry_solve",
                        *RESEARCH_ACTIONS,
                        "ask_user",
                        "propose_tradeoff",
                        "abort",
                    ),
                    success_criteria={"required_artifact_types": ["verified_itinerary_acceptance"]},
                    max_attempts=8,
                    invalidates_on=("goal_changed", "validation_result_changed"),
                ),
                TaskNode(
                    task_id="compose_draft",
                    goal="Compose a user-facing draft without changing verified fields",
                    depends_on=("review_itinerary",),
                    allowed_actions=("compose_draft",),
                    success_criteria={"required_artifact_types": ["itinerary_draft"]},
                ),
                TaskNode(
                    task_id="await_confirmation",
                    goal="Present the verified draft and wait for acceptance or revision feedback",
                    depends_on=("compose_draft",),
                    allowed_actions=("ask_user", "finish"),
                    success_criteria={"required_fact_keys": ["user_confirmation"]},
                ),
            )
        )
        return TaskGraph(
            goal_version=goal.goal_version,
            plan_version=plan_version,
            tasks=tuple(tasks),
        )


__all__ = [
    "RESEARCH_ACTIONS",
    "ReactTaskGraphPlanner",
    "ResearchRequirements",
    "ResearchSufficiencyReport",
    "ResearchSufficiencyVerifier",
    "infer_research_requirements",
]
