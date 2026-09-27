"""Evidence validation and the single-session lifecycle envelope."""

from __future__ import annotations

import re
import unicodedata
from datetime import UTC, date, datetime, timedelta
from pydantic import BaseModel, Field

from agentic.clock import reference_now, reference_today
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
    min_candidate_count: int = 1
    min_detail_count: int = 1
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
            delta = (date.fromisoformat(str(start_date)) - reference_today()).days
            requires_weather = requires_weather or 0 <= delta <= 10
        except ValueError:
            pass

    required = ["poi_candidate_set", "poi_detail_set", "route_matrix"]
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
        min_candidate_count=1,
        min_detail_count=1,
        requires_weather=requires_weather,
        requires_event=requires_event,
        requires_transport=requires_transport,
        requires_current_info=requires_current_info,
    )


class ResearchSufficiencyVerifier:
    """Validate evidence prerequisites when the model proposes solving or submission."""

    def evaluate(self, ledger: AgentLedgerState) -> ResearchSufficiencyReport:
        requirements = infer_research_requirements(ledger.goal)
        current = [
            artifact
            for artifact in ledger.artifacts.values()
            if artifact.goal_version == ledger.goal.goal_version
            and artifact.plan_version == ledger.task_graph.plan_version
            and (artifact.expires_at is None or artifact.expires_at > reference_now())
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

            "event_search_result": 24,
            "transport_search_result": 2,
        }
        now = reference_now()
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
        if ledger.goal.hard_constraints.get("require_named_restaurants"):
            dining = [p for p in detail_items if isinstance(p,dict) and p.get("category")=="restaurant"
                and p.get("id") and p.get("lat") and p.get("lng") and p.get("average_cost") is not None
                and p.get("open_time") and p.get("close_time")]
            coverage["named_restaurant_evidence"] = bool(dining)
            if not dining:missing.append("MISSING_NAMED_RESTAURANT_EVIDENCE:search_pois_and_get_poi_detail")
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

        for artifact_type in ("transport_search_result",):
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
        if requirements.requires_current_info:
            from agentic.current_evidence import fresh_searches, source_results, required_opening_targets, opening_target_covered
            searches=[a for a in fresh_searches(ledger) if source_results(a)]
            if not searches:
                missing.append("STALE_OR_UNSOURCED_CURRENT_INFO")
            targets=required_opening_targets(ledger,candidate_items)
            for name,day in targets:
                matched=[a for a in searches if opening_target_covered(a,name,day)]
                coverage[f"current_info:{name}:{day}"]=bool(matched)
                if not matched:
                    missing.append(f"CURRENT_INFO_ENTITY_UNCOVERED:{name}:{day}")
                else:evidence_refs.extend(a.artifact_id for a in matched)
            if not targets and set(ledger.goal.hard_constraints.get("information_needs") or []) & {"opening_hours","closure"}:
                applicable=any(opening_target_covered(a,p.get("name"),str(ledger.goal.hard_constraints.get("start_date") or ""))
                    for a in searches for p in candidate_items if isinstance(p,dict) and p.get("name"))
                if not applicable:missing.append("CURRENT_INFO_NOT_PLANNABLE")

        matrix = (by_type.get("route_matrix") or [None])[-1]
        if matrix is not None:
            rows = matrix.payload.get("time_minutes") or []
            expected_detail_count = int(
                (details.payload.get("expected_count") if details else 0) or len(detail_items)
            )
            expected_matrix_size = min(len(plannable_candidates), expected_detail_count) + 1
            if ledger.goal.hard_constraints.get("require_named_restaurants"):
                from agentic.action_executor import TravelActionExecutor
                expected_matrix_size = len(TravelActionExecutor._planning_candidate_items(ledger))+1
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
    """Create one open agent session; tool order belongs to the model."""

    def plan(self, goal: GoalLedger, *, plan_version: int = 1) -> TaskGraph:
        from agentic.policy_actions import POLICY_ACTION_MODELS

        return TaskGraph(
            goal_version=goal.goal_version,
            plan_version=plan_version,
            tasks=(
                TaskNode(
                    task_id="travel_agent",
                    goal=goal.original_request,
                    allowed_actions=tuple(POLICY_ACTION_MODELS),
                    success_criteria={"require_hard_pass": True},
                    max_attempts=24,
                ),
            ),
        )


__all__ = [
    "RESEARCH_ACTIONS",
    "ReactTaskGraphPlanner",
    "ResearchRequirements",
    "ResearchSufficiencyReport",
    "ResearchSufficiencyVerifier",
    "infer_research_requirements",
]
