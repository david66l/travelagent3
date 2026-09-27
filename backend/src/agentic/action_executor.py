"""Adapter from Agent Loop actions to existing deterministic travel tools."""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, timedelta
from typing import Any

from agentic.clock import reference_now
from agentic.loop import ActionOutcome, PolicyAction
from agentic.observations import ObservationEnvelope
from agentic.state import AgentLedgerState, ArtifactRecord, FactRecord, TaskNode
from core.settings import settings
from planner.preprocessing import PlayTimeManager
from tools.tool_executor import ToolExecutor
from vrp_solver_service.models import ConstraintsInput, POIInput


_ARTIFACT_TYPES = {
    "retrieve_city_knowledge": "city_knowledge",
    "get_weather": "weather_snapshot",
    "search_current_info": "current_info_search",
    "search_transport": "transport_search_result",
    "get_poi_detail": "poi_detail_set",
    "check_reservation": "poi_detail_set",
    "get_route_matrix": "route_matrix",
    "solve_itinerary": "solver_result",
    "validate_itinerary": "validation_report",
}

_FACT_TTL = {
    "candidate_poi_ids": timedelta(hours=24),
    "fixed_events": timedelta(hours=6),
    "transport_time_windows": timedelta(minutes=15),
}

_ARTIFACT_TTL = {
    "city_knowledge": timedelta(days=7),
    "poi_candidate_set": timedelta(hours=24),
    "poi_detail_set": timedelta(hours=24),
    "weather_snapshot": timedelta(hours=1),
    "current_info_search": timedelta(hours=6),
    "event_search_result": timedelta(hours=6),
    "transport_search_result": timedelta(minutes=15),
    "route_matrix": timedelta(minutes=30),
    "research_bundle": timedelta(minutes=15),
}


def _poi_identity(value: Any) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.sub(r"[\s·・,，。.!！?？:：;；()（）\[\]【】'\"“”‘’_-]+", "", normalized)


def _candidate_target_match_quality(item: dict[str, Any], target: Any) -> int | None:
    """Rank exact/id/provider-alias matches ahead of loose name containment."""
    target_identity = _poi_identity(target)
    if not target_identity:
        return None
    identities = [
        _poi_identity(item.get("name")),
        *[
            _poi_identity(alias)
            for alias in item.get("required_aliases") or []
            if str(alias).strip()
        ],
    ]
    if str(item.get("id") or "") == str(target) or target_identity in identities:
        return 0
    if any(
        identity and (target_identity in identity or identity in target_identity)
        for identity in identities
    ):
        return 1
    return None


def _candidate_matches_target(item: dict[str, Any], target: Any) -> bool:
    return _candidate_target_match_quality(item, target) is not None


def _is_plannable_candidate(item: dict[str, Any]) -> bool:
    category = str(item.get("category") or "attraction").lower()
    return category not in {"restaurant", "meal", "hotel", "transport"}


class TravelActionExecutor:
    """Execute policy actions without granting the policy direct state writes."""

    def __init__(self, tool_executor: ToolExecutor | None = None) -> None:
        self.tools = tool_executor or ToolExecutor()

    async def execute(
        self,
        *,
        task: TaskNode,
        action: PolicyAction,
        ledger: AgentLedgerState,
    ) -> ActionOutcome:
        if action.action == "ask_user":
            return ActionOutcome(
                status="awaiting_user",
                artifacts=[
                    self._artifact(
                        ledger,
                        action,
                        "user_question",
                        {"question": action.arguments.get("question")},
                    )
                ],
            )
        if action.action == "abort":
            return ActionOutcome(
                status="failed",
                error_code="POLICY_ABORT",
                error_message=str(action.arguments.get("reason") or "policy aborted"),
            )
        if action.action == "finish":
            # Submission formatting is mechanical; the model chose when to submit.
            solver = self._latest_artifact(ledger, "solver_result")
            report = self._latest_artifact(ledger, "validation_report")
            if solver is None or report is None or not report.payload.get("hard_pass"):
                return ActionOutcome(
                    status="failed", error_code="VALIDATION_NOT_PASSED", retryable=True
                )
            return ActionOutcome(
                status="awaiting_user",
                artifacts=[
                    self._artifact(
                        ledger,
                        action,
                        "verified_itinerary_acceptance",
                        {"validation_report_id": report.artifact_id},
                        evidence_refs=[report.artifact_id],
                    ),
                    self._artifact(
                        ledger,
                        action,
                        "itinerary_draft",
                        solver.payload,
                        evidence_refs=[solver.artifact_id],
                    ),
                    self._artifact(ledger, action, "finish", {}),
                ],
            )
        if action.action == "propose_tradeoff":
            return ActionOutcome(
                status="awaiting_user",
                artifacts=[
                    self._artifact(ledger, action, "propose_tradeoff", action.arguments),
                ],
            )

        if action.action == "get_poi_detail":
            return await self._collect_poi_details(task, action, ledger)

        arguments = self._hydrate_arguments(ledger, action)
        call = {
            "id": action.action_id,
            "type": "function",
            "function": {
                "name": action.action,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
        records = await self.tools.execute(
            [call],
            guard_context={"allowed_tools": set(task.allowed_actions), "max_calls": 1},
        )
        record = records[0]
        observation = ObservationEnvelope(**record["observation"])
        if not observation.ok:
            error = observation.error
            return ActionOutcome(
                status="failed",
                observations=[observation],
                error_code=error.code if error else "TOOL_FAILED",
                error_message=error.message if error else "tool failed",
                retryable=error.retryable if error else False,
                tool_calls_used=1,
            )

        facts: list[FactRecord] = []
        artifacts: list[ArtifactRecord] = []
        artifact_type = _ARTIFACT_TYPES.get(action.action)
        if action.action == "search_current_info" and (
            action.arguments.get("info_type") == "event"
            or (isinstance(observation.data, dict) and observation.data.get("event"))
        ):
            artifact_type = "event_search_result"
        if action.action == "search_pois":
            fresh_items = observation.data if isinstance(observation.data, list) else []
            if not fresh_items:
                return ActionOutcome(
                    status="failed",
                    observations=[observation],
                    error_code="CANDIDATE_POIS_EMPTY",
                    error_message="POI search returned no grounded candidates",
                    retryable=True,
                    tool_calls_used=1,
                )
            previous = self._latest_artifact(ledger, "poi_candidate_set")
            knowledge = self._latest_artifact(ledger, "city_knowledge")
            merged_by_identity: dict[str, dict[str, Any]] = {}
            for item in [
                *((knowledge.payload.get("pois") or []) if knowledge else []),
                *((previous.payload.get("pois") or []) if previous else []),
                *fresh_items,
            ]:
                if not isinstance(item, dict):
                    continue
                identity = str(item.get("id") or "").strip() or _poi_identity(item.get("name"))
                if identity:
                    merged_by_identity[identity] = item
            items = list(merged_by_identity.values())
            items = self._filter_forbidden_candidates(ledger, items)
            items = self._prioritize_required_candidates(ledger, items)
            if not items:
                return ActionOutcome(
                    status="failed",
                    observations=[observation],
                    error_code="CANDIDATE_POIS_ALL_FORBIDDEN",
                    error_message="all grounded candidates violate user exclusions",
                    retryable=True,
                    tool_calls_used=1,
                )
            poi_ids = [
                str(item.get("id") or item.get("name"))
                for item in items
                if isinstance(item, dict) and (item.get("id") or item.get("name"))
            ]
            facts.append(
                self._fact(
                    ledger,
                    action,
                    observation,
                    "candidate_poi_ids",
                    poi_ids,
                )
            )
            artifacts.append(
                self._artifact(
                    ledger,
                    action,
                    "poi_candidate_set",
                    {"pois": items},
                    evidence_refs=list(
                        dict.fromkeys(
                            [
                                action.action_id,
                                *([knowledge.artifact_id] if knowledge else []),
                            ]
                        )
                    ),
                )
            )
        elif artifact_type:
            payload = (
                observation.data
                if isinstance(observation.data, dict)
                else {"data": observation.data}
            )
            if action.action == "get_weather" and isinstance(observation.data, list):
                payload = {
                    "days": observation.data,
                    "queried_at": reference_now().isoformat(),
                }
            if action.action == "search_transport":
                transport_constraints = self._transport_planning_constraints(ledger, payload)
                payload["planning_constraints"] = transport_constraints
                if transport_constraints.get("applied"):
                    facts.append(
                        self._fact(
                            ledger,
                            action,
                            observation,
                            "transport_time_windows",
                            transport_constraints,
                        )
                    )
            payload = {
                **payload,
                "_evidence_source": observation.source,
                "_evidence_confidence": observation.confidence,
                "_is_fallback": observation.is_fallback,
            }
            artifacts.append(
                self._artifact(
                    ledger,
                    action,
                    artifact_type,
                    payload,
                    evidence_refs=[action.action_id],
                )
            )
            if artifact_type == "event_search_result":
                event = (observation.data or {}).get("event") or {}
                if event.get("complete"):
                    event_poi = self._event_candidate(event)
                    previous = self._latest_artifact(ledger, "poi_candidate_set")
                    merged = list((previous.payload.get("pois") or []) if previous else [])
                    merged = [
                        item
                        for item in merged
                        if _poi_identity(item.get("name")) != _poi_identity(event_poi["name"])
                    ]
                    # Fixed events must survive the bounded POI-detail slice.
                    merged.insert(0, event_poi)
                    artifacts.append(
                        self._artifact(
                            ledger,
                            action,
                            "poi_candidate_set",
                            {"pois": merged},
                            evidence_refs=[action.action_id],
                        )
                    )
                    facts.append(
                        self._fact(
                            ledger,
                            action,
                            observation,
                            "fixed_events",
                            [
                                {
                                    "poi_id": event_poi["id"],
                                    "date": event.get("date"),
                                    "start_time": event.get("start_time"),
                                    "end_time": event.get("end_time"),
                                    "note": event.get("name"),
                                }
                            ],
                        )
                    )
        return ActionOutcome(
            observations=[observation],
            facts=facts,
            artifacts=artifacts,
            tool_calls_used=1,
        )

    async def _collect_poi_details(
        self,
        task: TaskNode,
        action: PolicyAction,
        ledger: AgentLedgerState,
    ) -> ActionOutcome:
        items = self._eligible_candidate_items(ledger)
        available_for_details = max(
            1,
            ledger.budget.remaining_tool_calls - settings.agentic_reserved_gate_tool_calls,
        )
        items = items[: min(settings.agentic_poi_detail_limit, available_for_details)]
        names = [
            str(item.get("name")) for item in items if isinstance(item, dict) and item.get("name")
        ]
        if not names:
            return ActionOutcome(
                status="failed",
                error_code="CANDIDATE_POIS_MISSING",
                error_message="POI details require a verified candidate set",
            )
        city = ledger.goal.hard_constraints.get("destination")
        calls = [
            {
                "id": f"{action.action_id}:{index}",
                "type": "function",
                "function": {
                    "name": "get_poi_detail",
                    "arguments": json.dumps({"poi_name": name, "city": city}, ensure_ascii=False),
                },
            }
            for index, name in enumerate(names)
        ]
        action.executed_arguments = {
            "calls": [json.loads(call["function"]["arguments"]) for call in calls]
        }
        action.argument_sources = {"calls": "harness"}
        action.controller_hydrated_fields = ["calls"]
        records = await self.tools.execute(
            calls,
            guard_context={
                "allowed_tools": set(task.allowed_actions),
                "max_calls": len(calls),
            },
        )
        observations = [ObservationEnvelope(**record["observation"]) for record in records]
        usable = [item for item in observations if item.ok]
        if len(usable) != len(names):
            failed = next((item for item in observations if not item.ok), None)
            error = failed.error if failed else None
            return ActionOutcome(
                status="failed",
                observations=observations,
                error_code=error.code if error else "POI_DETAIL_SET_INCOMPLETE",
                error_message=error.message if error else "POI detail set is incomplete",
                retryable=error.retryable if error else True,
                tool_calls_used=len(observations),
            )
        identity_mismatches = [
            {
                "expected": expected_name,
                "actual": (observation.data or {}).get("name"),
            }
            for expected_name, observation in zip(names, usable, strict=True)
            if _poi_identity(expected_name) != _poi_identity((observation.data or {}).get("name"))
        ]
        if identity_mismatches:
            return ActionOutcome(
                status="failed",
                observations=observations,
                error_code="POI_DETAIL_IDENTITY_MISMATCH",
                error_message="POI detail evidence does not match the requested candidates",
                retryable=True,
                tool_calls_used=len(observations),
            )
        return ActionOutcome(
            observations=observations,
            artifacts=[
                self._artifact(
                    ledger,
                    action,
                    "poi_detail_set",
                    {"details": [item.data for item in usable], "expected_count": len(names)},
                    evidence_refs=[
                        item.tool_call_id for item in usable if item.tool_call_id is not None
                    ],
                )
            ],
            tool_calls_used=len(observations),
        )

    def _hydrate_arguments(self, ledger: AgentLedgerState, action: PolicyAction) -> dict[str, Any]:
        """Inject trusted payloads so the policy selects rather than copies facts."""
        arguments = dict(action.arguments)
        destination = ledger.goal.hard_constraints.get("destination")
        if (
            action.action
            in {
                "search_pois",
                "get_weather",
                "retrieve_city_knowledge",
                "search_current_info",
            }
            and destination
        ):
            arguments["city"] = destination
        if action.action == "search_current_info":
            start_date = ledger.goal.hard_constraints.get("start_date")
            if start_date:
                if not arguments.get("date"):
                    arguments["date"] = start_date
        if action.action == "search_transport":
            origin = ledger.goal.hard_constraints.get("origin")
            if not origin:
                raise ValueError("transport search requires a user-grounded origin")
            arguments["origin"] = origin
            arguments["destination"] = destination
            start_date = ledger.goal.hard_constraints.get("start_date")
            end_date = ledger.goal.hard_constraints.get("end_date")
            if start_date:
                if not arguments.get("date"):
                    arguments["date"] = start_date
            if end_date:
                arguments["return_date"] = end_date
        if action.action == "search_pois":
            # Candidate supply must include attractions and dining options. A
            # policy may choose grounded semantic keywords, but cannot narrow
            # the only supply query to a category that makes the solver
            # infeasible.  Preserve an explicit keyword choice so a recovery
            # step can actually narrow a failed query; inject preferences only
            # when the policy leaves the choice empty.
            arguments["category"] = None
            # Empty keywords intentionally mean a broad search, not a hidden preference policy.
            arguments["keywords"] = list(dict.fromkeys(arguments.get("keywords") or []))[:8]
            arguments["required_pois"] = list(
                dict.fromkeys(
                    str(item).strip()
                    for item in ledger.goal.hard_constraints.get("must_visit") or []
                    if str(item).strip()
                )
            )

        if action.action in {"get_route_matrix", "solve_itinerary"}:
            candidate_items = self._planning_candidate_items_with_evidence(ledger)
            if candidate_items:
                candidate_pois = [
                    self._poi_input(item, index)
                    for index, item in enumerate(candidate_items)
                    if isinstance(item, dict)
                    and item.get("name")
                    and str(item.get("category") or "attraction").lower()
                    not in ({"meal", "hotel", "transport"} if ledger.goal.hard_constraints.get("require_named_restaurants") else {"restaurant", "meal", "hotel", "transport"})
                ]
                arguments["pois"] = candidate_pois
                if not candidate_pois:
                    raise ValueError(
                        "verified candidate set contained no attraction POIs; "
                        "provider category normalization is invalid"
                    )
            arguments["constraints"] = self._trusted_constraints(ledger)

        if action.action == "solve_itinerary":
            arguments.setdefault("strategy", "cpsat")
            matrix = self._latest_artifact(ledger, "route_matrix")
            if matrix is not None:
                arguments["dist_matrix"] = matrix.payload.get("time_minutes")
                arguments["tc_matrix"] = matrix.payload.get("transport_cost")

        if action.action == "validate_itinerary":
            solver = self._latest_artifact(ledger, "solver_result")
            arguments["itinerary"] = (solver.payload.get("days") or []) if solver is not None else []
            arguments["constraints"] = self._trusted_constraints(ledger)
            arguments["facts"] = self._effective_planning_facts(ledger)
        action.executed_arguments = dict(arguments)
        action.argument_sources = {
            key: (
                "model"
                if key in (action.model_arguments or action.arguments)
                and (action.model_arguments or action.arguments)[key] == value
                else "harness"
            )
            for key, value in arguments.items()
        }
        action.controller_hydrated_fields = [
            k for k, origin in action.argument_sources.items() if origin == "harness"
        ]
        return arguments

    def _effective_planning_facts(self, ledger: AgentLedgerState) -> list[dict[str, Any]]:
        """Mirror solver preprocessing so validation uses the exact same POI windows."""
        items = self._planning_candidate_items_with_evidence(ledger)
        pois = [
            POIInput(**self._poi_input(item, index))
            for index, item in enumerate(items)
            if isinstance(item, dict) and item.get("name") and (_is_plannable_candidate(item) or (ledger.goal.hard_constraints.get("require_named_restaurants") and item.get("category")=="restaurant"))
        ]
        constraints = ConstraintsInput(**self._trusted_constraints(ledger))
        effective = PlayTimeManager().adjust(pois, constraints)
        return [item.model_dump(mode="json") for item in effective]

    @staticmethod
    def _planning_candidate_items(ledger: AgentLedgerState) -> list[dict[str, Any]]:
        """Use the bounded, evidence-collected POI subset for matrix and solve."""
        raw_items = TravelActionExecutor._eligible_candidate_items(ledger)
        details = TravelActionExecutor._latest_artifact(ledger, "poi_detail_set")
        if details is not None:
            detail_items = [
                item
                for item in details.payload.get("details") or []
                if isinstance(item, dict) and item.get("name")
            ]
            expected_count = int(details.payload.get("expected_count") or len(detail_items))
            selected_raw = [item for item in raw_items[:expected_count] if isinstance(item, dict)]
            by_name = {_poi_identity(item.get("name")): item for item in detail_items}
            merged = []
            for raw in selected_raw:
                detail = by_name.get(_poi_identity(raw.get("name")))
                if detail is None:
                    raise ValueError("verified POI details do not align with candidate order")
                item = {**raw, **detail}
                for identity_key in ("id", "name", "category"):
                    if raw.get(identity_key) is not None:
                        item[identity_key] = raw[identity_key]
                if str(raw.get("category") or "").lower() == "event":
                    for fixed_key in (
                        "lat",
                        "lng",
                        "open_time",
                        "close_time",
                        "duration_minutes",
                        "recommended_hours",
                        "must_visit",
                    ):
                        if raw.get(fixed_key) is not None:
                            item[fixed_key] = raw[fixed_key]
                merged.append(item)
            return merged
        available = max(
            1,
            ledger.budget.remaining_tool_calls - settings.agentic_reserved_gate_tool_calls,
        )
        limit = min(settings.agentic_poi_detail_limit, available)
        return [item for item in raw_items[:limit] if isinstance(item, dict)]

    @staticmethod
    def _eligible_candidate_items(ledger: AgentLedgerState) -> list[dict[str, Any]]:
        """Build one stable candidate order for detail collection, routing, and solving."""
        candidates = TravelActionExecutor._latest_artifact(ledger, "poi_candidate_set")
        items = TravelActionExecutor._prioritize_required_candidates(
            ledger,
            TravelActionExecutor._filter_forbidden_candidates(
                ledger,
                [
                    item
                    for item in (candidates.payload.get("pois", []) if candidates else [])
                    if isinstance(item, dict) and (_is_plannable_candidate(item) or (ledger.goal.hard_constraints.get("require_named_restaurants") and item.get("category")=="restaurant"))
                ],
            ),
        )
        return TravelActionExecutor._deduplicate_candidates(items)

    @staticmethod
    def _deduplicate_candidates(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate by the identity actually sent to the name-based detail tool."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for item in items:
            identity = _poi_identity(item.get("name") or item.get("id"))
            if identity and identity in seen:
                continue
            if identity:
                seen.add(identity)
            unique.append(item)
        return unique

    @staticmethod
    def _filter_forbidden_candidates(
        ledger: AgentLedgerState,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        excluded = [
            *list(ledger.goal.hard_constraints.get("must_not_visit") or []),
            *list(ledger.goal.soft_preferences.get("avoid_pois") or []),
        ]
        excluded_identities = [_poi_identity(item) for item in excluded if item]
        if not excluded_identities:
            return items
        kept: list[dict[str, Any]] = []
        for item in items:
            identity = _poi_identity(item.get("name") or item.get("id"))
            forbidden = any(
                target and (target in identity or identity in target)
                for target in excluded_identities
            )
            if not forbidden:
                kept.append(item)
        return kept

    @staticmethod
    def _prioritize_required_candidates(
        ledger: AgentLedgerState,
        items: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Reserve one best candidate per must-visit inside the bounded slice."""
        required = list(
            dict.fromkeys(
                str(item).strip()
                for item in ledger.goal.hard_constraints.get("must_visit") or []
                if str(item).strip()
            )
        )
        if not required and not ledger.goal.hard_constraints.get("require_named_restaurants"):
            return items

        remaining = set(range(len(items)))
        selected: list[int] = []
        for target in required:
            matches = [
                (quality, index)
                for index, item in enumerate(items)
                if index in remaining
                and (quality := _candidate_target_match_quality(item, target)) is not None
            ]
            if not matches:
                continue
            _quality, chosen = min(matches)
            selected.append(chosen)
            remaining.remove(chosen)
        if ledger.goal.hard_constraints.get("require_named_restaurants"):
            # Reserve dining evidence inside the detail tool's bounded slice.
            for index in sorted(remaining):
                if items[index].get("category")=="restaurant":
                    selected.append(index);remaining.remove(index)
                    if sum(items[i].get("category")=="restaurant" for i in selected)>=2:break
        return [items[index] for index in selected] + [
            item for index, item in enumerate(items) if index in remaining
        ]

    @staticmethod
    def _planning_candidate_items_with_evidence(
        ledger: AgentLedgerState,
    ) -> list[dict[str, Any]]:
        """Apply source-backed live availability to the exact matched POI entity."""
        items = TravelActionExecutor._planning_candidate_items(ledger)
        patched = [dict(item) for item in items]
        by_identity = {_poi_identity(item.get("name")): item for item in patched}
        from agentic.current_evidence import fresh_searches
        current_searches = fresh_searches(ledger)
        weekday_tokens = {
            "周一": 0,
            "星期一": 0,
            "周二": 1,
            "星期二": 1,
            "周三": 2,
            "星期三": 2,
            "周四": 3,
            "星期四": 3,
            "周五": 4,
            "星期五": 4,
            "周六": 5,
            "星期六": 5,
            "周日": 6,
            "周天": 6,
            "星期日": 6,
        }
        for artifact in current_searches:
            payload = artifact.payload
            info_type = str(payload.get("info_type") or "general")
            if info_type not in {"opening_hours", "closure", "general"}:
                continue
            for result in payload.get("results") or []:
                if not isinstance(result, dict) or not result.get("url"):
                    continue
                corpus = f"{result.get('title') or ''} {result.get('snippet') or ''}"
                normalized_corpus = _poi_identity(corpus)
                matches = [
                    (identity, item)
                    for identity, item in by_identity.items()
                    if identity and identity in normalized_corpus
                ]
                if not matches:
                    continue
                times = list(
                    dict.fromkeys(re.findall(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d", corpus))
                )
                raw_date = str(payload.get("date") or "")
                date_match = re.search(r"20\d{2}[-年/.]\d{1,2}[-月/.]\d{1,2}日?", corpus)
                evidence_date = TravelActionExecutor._normalize_iso_date(
                    date_match.group(0) if date_match else raw_date
                )
                closed = any(
                    token in corpus for token in ("临时闭馆", "暂停开放", "闭馆", "停业", "关闭")
                )
                for _identity, item in matches:
                    if closed and evidence_date:
                        item["closed_dates"] = list(
                            dict.fromkeys([*list(item.get("closed_dates") or []), evidence_date])
                        )
                    elif closed:
                        weekdays = list(item.get("closed_weekdays") or [])
                        weekdays.extend(
                            weekday for token, weekday in weekday_tokens.items() if token in corpus
                        )
                        item["closed_weekdays"] = sorted(set(weekdays))
                    if len(times) >= 2:
                        opening = (times[0], times[1])
                        if evidence_date:
                            overrides = dict(item.get("date_opening_hours") or {})
                            overrides[evidence_date] = opening
                            item["date_opening_hours"] = overrides
                        else:
                            item["open_time"], item["close_time"] = opening
                    refs = list(item.get("availability_evidence_urls") or [])
                    item["availability_evidence_urls"] = list(
                        dict.fromkeys([*refs, str(result["url"])])
                    )
        return patched

    @staticmethod
    def _trusted_constraints(ledger: AgentLedgerState) -> dict[str, Any]:
        hard = ledger.goal.hard_constraints
        soft = ledger.goal.soft_preferences
        start_date = hard.get("start_date")
        travel_days = int(hard.get("travel_days") or 1)
        fixed_events = TravelActionExecutor._latest_fact_value(ledger, "fixed_events") or []
        if not start_date and fixed_events and fixed_events[0].get("date"):
            start_date = (
                str(fixed_events[0]["date"])
                .replace("年", "-")
                .replace("月", "-")
                .replace("日", "")
                .replace("/", "-")
                .replace(".", "-")
            )
        day_weekdays: list[int] = []
        if start_date:
            try:
                first = date.fromisoformat(str(start_date))
                day_weekdays = [
                    (first + timedelta(days=offset)).weekday() for offset in range(travel_days)
                ]
            except ValueError:
                day_weekdays = []
        requested_must_visit = list(hard.get("must_visit") or [])
        candidate_items = TravelActionExecutor._planning_candidate_items(ledger)
        must_visit: list[str] = []
        for target in requested_must_visit:
            match = next(
                (item for item in candidate_items if _candidate_matches_target(item, target)),
                None,
            )
            must_visit.append(str((match or {}).get("id") or (match or {}).get("name") or target))
        must_visit.extend(
            str(item.get("poi_id"))
            for item in fixed_events
            if isinstance(item, dict) and item.get("poi_id")
        )
        transport = TravelActionExecutor._latest_fact_value(ledger, "transport_time_windows") or {}
        return {
            "travel_days": travel_days,
            "trip_start_date": start_date,
            "day_weekdays": day_weekdays,
            "daily_start_minutes": list(transport.get("daily_start_minutes") or []),
            "daily_end_minutes": list(transport.get("daily_end_minutes") or []),
            "total_budget": float(hard.get("budget_range") or 0),
            "max_transit_minutes": int(hard.get("max_transit_minutes") or 120),
            "max_walk_minutes": (
                int(hard["max_walk_minutes"]) if hard.get("max_walk_minutes") is not None else None
            ),
            "must_visit": list(dict.fromkeys(must_visit)),
            "must_not_visit": [
                *list(hard.get("must_not_visit") or []),
                *list(soft.get("avoid_pois") or []),
            ],
            "user_reservations": fixed_events,
            "interests": list(soft.get("interests") or []),
            "include_restaurant": True,
            "meals_per_day": 2,
            **({"require_named_restaurants": True,"food_day":0,
                "route_poi_ids": (TravelActionExecutor._latest_artifact(ledger,"route_matrix").payload.get("poi_ids",[]) if TravelActionExecutor._latest_artifact(ledger,"route_matrix") else []),
                "route_time_matrix": (TravelActionExecutor._latest_artifact(ledger,"route_matrix").payload.get("time_minutes",[]) if TravelActionExecutor._latest_artifact(ledger,"route_matrix") else []),
                "route_cost_matrix": (TravelActionExecutor._latest_artifact(ledger,"route_matrix").payload.get("transport_cost",[]) if TravelActionExecutor._latest_artifact(ledger,"route_matrix") else [])}
               if hard.get("require_named_restaurants") else {}),
        }

    @staticmethod
    def _poi_input(item: dict[str, Any], index: int) -> dict[str, Any]:
        location = item.get("location") or {}
        recommended = str(item.get("recommended_hours") or "")
        duration = int(item.get("duration_minutes") or 120)
        try:
            duration = max(15, int(float(recommended) * 60))
        except ValueError:
            pass
        return {
            "id": str(item.get("id") or item.get("name") or f"poi-{index}"),
            "name": str(item.get("name") or f"POI-{index}"),
            "category": str(item.get("category") or "attraction"),
            "tags": list(item.get("tags") or []),
            "lat": float(item.get("lat") or location.get("lat") or 0),
            "lng": float(item.get("lng") or location.get("lng") or 0),
            "score": float(item.get("score") or 0.5),
            "ticket_price": float(item.get("ticket_price") or item.get("price") or 0),
            "average_cost": item.get("average_cost"),
            "duration_minutes": duration,
            "open_time": str(item.get("open_time") or "08:00"),
            "close_time": str(item.get("close_time") or "18:00"),
            "closed_weekdays": list(item.get("closed_weekdays") or []),
            "closed_dates": list(item.get("closed_dates") or []),
            "date_opening_hours": dict(item.get("date_opening_hours") or {}),
            "availability_evidence_urls": list(item.get("availability_evidence_urls") or []),
            "must_visit": bool(item.get("must_visit")),
        }

    @staticmethod
    def _normalize_iso_date(value: Any) -> str | None:
        normalized = (
            str(value or "")
            .strip()
            .replace("年", "-")
            .replace("月", "-")
            .replace("日", "")
            .replace("/", "-")
            .replace(".", "-")
        )
        try:
            return date.fromisoformat(normalized).isoformat()
        except ValueError:
            return None

    @staticmethod
    def _transport_planning_constraints(
        ledger: AgentLedgerState,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Translate selected source-backed intercity legs into daily tour bounds."""
        travel_days = max(1, int(ledger.goal.hard_constraints.get("travel_days") or 1))
        starts = [8 * 60 for _ in range(travel_days)]
        ends = [21 * 60 for _ in range(travel_days)]
        applied_legs: list[dict[str, Any]] = []
        for leg in payload.get("legs") or []:
            if not isinstance(leg, dict):
                continue
            selected = leg.get("selected_option") or {}
            if not isinstance(selected, dict) or not selected.get("source_url"):
                continue
            direction = str(leg.get("direction") or "")
            if direction == "inbound":
                arrival = TravelActionExecutor._time_minutes(selected.get("arrival_time"))
                if arrival is None:
                    continue
                starts[0] = max(starts[0], min(23 * 60, arrival + 60))
            elif direction == "outbound":
                departure = TravelActionExecutor._time_minutes(selected.get("departure_time"))
                if departure is None:
                    continue
                ends[-1] = min(ends[-1], max(0, departure - 90))
            else:
                continue
            applied_legs.append(
                {
                    "direction": direction,
                    "date": leg.get("date"),
                    "service_code": selected.get("service_code"),
                    "departure_time": selected.get("departure_time"),
                    "arrival_time": selected.get("arrival_time"),
                    "source_url": selected.get("source_url"),
                }
            )
        feasible = all(start < end for start, end in zip(starts, ends, strict=True))
        return {
            "applied": bool(applied_legs) and feasible,
            "daily_start_minutes": starts,
            "daily_end_minutes": ends,
            "legs": applied_legs,
            "conflict": None if feasible else "transport leaves no usable planning window",
        }

    @staticmethod
    def _time_minutes(value: Any) -> int | None:
        try:
            hour, minute = (int(part) for part in str(value).split(":"))
        except (TypeError, ValueError):
            return None
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        return hour * 60 + minute

    @staticmethod
    def _event_candidate(event: dict[str, Any]) -> dict[str, Any]:
        start = str(event.get("start_time") or "19:30")
        end = str(event.get("end_time") or "22:00")
        try:
            sh, sm = (int(part) for part in start.split(":"))
            eh, em = (int(part) for part in end.split(":"))
            duration = max(30, (eh * 60 + em) - (sh * 60 + sm))
        except (TypeError, ValueError):
            duration = 150
        name = str(event.get("venue") or event.get("name") or "固定活动")
        return {
            "id": f"event:{_poi_identity(name)}",
            "name": name,
            "category": "event",
            "score": 1.0,
            "lat": float(event.get("lat") or 0),
            "lng": float(event.get("lng") or 0),
            "open_time": start,
            "close_time": end,
            "duration_minutes": duration,
            "recommended_hours": str(duration / 60),
            "ticket_price": 0,
            "tags": ["固定活动"],
            "must_visit": True,
        }

    @staticmethod
    def _latest_fact_value(ledger: AgentLedgerState, key: str) -> Any:
        now = reference_now()
        matches = [
            fact
            for fact in ledger.facts.values()
            if fact.key == key
            and fact.goal_version == ledger.goal.goal_version
            and fact.plan_version == ledger.task_graph.plan_version
            and (fact.expires_at is None or fact.expires_at > now)
        ]
        return matches[-1].value if matches else None

    @staticmethod
    def _latest_artifact(ledger: AgentLedgerState, artifact_type: str) -> ArtifactRecord | None:
        now = reference_now()
        matches = [
            artifact
            for artifact in ledger.artifacts.values()
            if artifact.artifact_type == artifact_type
            and artifact.goal_version == ledger.goal.goal_version
            and artifact.plan_version == ledger.task_graph.plan_version
            and (artifact.expires_at is None or artifact.expires_at > now)
        ]
        return matches[-1] if matches else None

    @staticmethod
    def _artifact(
        ledger: AgentLedgerState,
        action: PolicyAction,
        artifact_type: str,
        payload: dict[str, Any],
        *,
        evidence_refs: list[str] | None = None,
    ) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=f"{ledger.trajectory_id}:{action.action_id}:{artifact_type}",
            artifact_type=artifact_type,
            payload=payload,
            evidence_refs=evidence_refs or [action.action_id],
            goal_version=ledger.goal.goal_version,
            plan_version=ledger.task_graph.plan_version,
            expires_at=(
                reference_now() + _ARTIFACT_TTL[artifact_type]
                if artifact_type in _ARTIFACT_TTL
                else None
            ),
        )

    @staticmethod
    def _fact(
        ledger: AgentLedgerState,
        action: PolicyAction,
        observation: ObservationEnvelope,
        key: str,
        value: Any,
    ) -> FactRecord:
        return FactRecord(
            fact_id=f"{ledger.trajectory_id}:{action.action_id}:{key}",
            key=key,
            value=value,
            observation_ref=action.action_id,
            goal_version=ledger.goal.goal_version,
            plan_version=ledger.task_graph.plan_version,
            source=observation.source,
            confidence=observation.confidence,
            expires_at=(reference_now() + _FACT_TTL[key] if key in _FACT_TTL else None),
        )
