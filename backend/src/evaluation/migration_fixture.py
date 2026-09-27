"""Strict, offline migration of legacy travel fixtures to the current Harness.

Only provider facts cross this boundary. Old solver answers, verifier answers,
policy prefixes, and private oracle metadata never enter a rebuilt case.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math

from agentic.observations import ObservationEnvelope
from agentic.environment import SnapshotToolExecutor, SnapshotToolResponse
from evaluation import research_fixture
from schemas import ToolResult
from tools.tool_executor import ToolExecutor

VERSION = "legacy-provider-facts-migration.v2"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def rebuild_case(selected):
    """Return a proposal and explicit blockers; readiness requires real execution."""
    source = selected["source"]
    task, snapshot = source["task"], source["snapshot"]
    request = selected["request"]
    tools = snapshot["tool_responses"]
    blockers, notes = [], []
    if task["user_request"] != request["request"]:
        blockers.append("SOURCE_REQUEST_MISMATCH")
    if task.get("missing_slots"):
        blockers.append("LEGACY_MISSING_SLOTS_REQUIRE_USER_TEXT_RECONCILIATION")
    family = task["template_family"]
    # These old tasks inject their decisive facts via solver/verifier answers.
    # Replacing them by unrelated generic POIs would remove the task itself.
    if "verifier-repair" in family or "necessary-abort" in family:
        blockers.append("DECISIVE_CONSTRAINT_NOT_RECONSTRUCTED_FROM_PROVIDER_FACTS")
    if task.get("feasibility_report", {}).get("feasible") is False:
        notes.append("OLD_INFEASIBILITY_LABEL_EXCLUDED_RECOMPUTE_FROM_FACTS")
    responses = tools.get("search_pois", [])
    successful = [r for r in responses if isinstance(r.get("data"), list) and not r.get("error_code")]
    raw = successful[0]["data"] if successful else []
    if not raw:
        blockers.append("NO_SEARCH_PROVIDER_FACTS")
    facts = []
    for index, poi in enumerate(raw):
        p = deepcopy(poi)
        location = p.get("location") or {}
        for key in ("lat", "lng"):
            value = p.get(key, location.get(key))
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                blockers.append("MISSING_POI_COORDINATES")
            else:
                p[key] = value
        if not p.get("open_time") or not p.get("close_time"):
            blockers.append("MISSING_EXPLICIT_OPENING_HOURS")
        duration = p.get("duration_minutes")
        if duration is None:
            hours = p.get("recommended_hours")
            if isinstance(hours, (int, float)) and hours > 0:
                duration = int(hours * 60)
        if duration is None:
            blockers.append("MISSING_NUMERIC_VISIT_DURATION")
        else:
            p["duration_minutes"] = duration
        if p.get("ticket_price") is None:
            blockers.append("MISSING_EXPLICIT_POI_COST")
        p["id"] = p.get("id") or f"{selected['id']}-source-poi-{index}"
        # This URL names a frozen fixture, never a purported real travel source.
        p["source_url"] = f"https://migration.example.invalid/{selected['id']}/{index}"
        p["fixture_record"] = True
        facts.append(p)
    profile = deepcopy(task.get("profile") or {})
    slots = {**profile, **deepcopy(task["slots"])}
    for key, value in request["hard_constraints"].items():
        if key in slots and slots[key] != value:
            blockers.append("SOURCE_CONSTRAINT_MISMATCH")
        elif key not in slots:
            blockers.append("CANDIDATE_CONSTRAINT_ABSENT_FROM_SOURCE")
    faults = []
    queries = []
    for r in responses:
        if r.get("error_code"):
            if r["error_code"] not in {"UPSTREAM_TIMEOUT", "QUERY_TOO_BROAD"}:
                blockers.append("UNSUPPORTED_PROVIDER_FAULT")
            faults.append({**{k: deepcopy(r.get(k)) for k in ("expected_arguments", "error_code", "fallback_reason", "retryable")},
                           "argument_match_mode": r.get("argument_match_mode", "exact"),
                           "ignored_keyword_values": deepcopy(r.get("ignored_keyword_values", []))})
        elif isinstance(r.get("data"), list):
            queries.append({"keywords": r.get("expected_arguments", {}).get("keywords", []), "names": [p["name"] for p in r["data"]],
                            "expected_arguments": deepcopy(r.get("expected_arguments", {})),
                            "argument_match_mode": r.get("argument_match_mode", "exact"),
                            "ignored_keyword_values": deepcopy(r.get("ignored_keyword_values", []))})
    case = {
        "id": selected["id"], "family": "migration", "variant": 0,
        "request": request["request"], "slots": slots, "missing_slots": [],
        "missing_field": None, "expected": "plan", "reference_time": "2026-09-05T00:00:00+00:00",
        "pois": [p for p in facts if p.get("category") != "restaurant"],
        "restaurants": [p for p in facts if p.get("category") == "restaurant"],
        "source_group": selected["audit"]["group"]["source_group"], "split": "train",
        "synthetic": True, "dataset_version": VERSION, "training_targets_present": False,
        "provider_faults": faults, "provider_query_facts": queries,
        "source_case_sha256": fingerprint(source),
    }
    return case, {"id": case["id"], "blockers": sorted(set(blockers)), "notes": notes,
                  "training_ready": False, "v9_rollout_ready": False,
                  "source_group": case["source_group"], "source_case_sha256": case["source_case_sha256"]}


class MigrationResearchExecutor(ToolExecutor):
    """Closed provider registry; route matrix, solver and validator remain real."""

    def __init__(self, case):
        super().__init__()
        self.case = deepcopy(case)
        self.counts = Counter()
        self.calls = []
        self.first_info_query = None
        self._search_outcomes = []
        self._seen_faults = set()
        self._contracts = [SnapshotToolResponse.model_validate({k: r[k] for k in ("expected_arguments", "argument_match_mode", "ignored_keyword_values")})
                           for r in case["provider_faults"] + case["provider_query_facts"]]
        real = {name: self._handlers[name] for name in ("get_route_matrix", "solve_itinerary", "validate_itinerary")}
        # No fallback to a live provider, even for unsupported tools.
        self._handlers = {name: self._unavailable for name in self._handlers}
        self._handlers.update(real)
        self._handlers["search_pois"] = self._search
        self._handlers["get_poi_detail"] = self._detail
        self._handlers["retrieve_city_knowledge"] = self._knowledge

    async def _unavailable(self, args):
        raise ValueError("This offline migration fixture has no grounded provider response for this tool")

    async def _knowledge(self, args):
        self.counts["retrieve_city_knowledge"] += 1
        return ToolResult(data={"city": self.case["slots"].get("destination"),
            "summary": "离线迁移场景，仅包含下列冻结 POI 资料。", "pois": research_fixture.catalog(self.case)}, data_source="built_in", confidence=1.0)

    async def _detail(self, args):
        self.counts["get_poi_detail"] += 1
        return ToolResult(data=research_fixture.detail(self.case, args["poi_name"]), data_source="built_in", confidence=1.0)

    async def _search(self, args):
        self.counts["search_pois"] += 1
        self._search_outcomes.append(None)
        if args.get("city") and args["city"] != self.case["slots"].get("destination"):
            return ToolResult(data=[], data_source="built_in", confidence=1.0)
        keywords = list(args.get("keywords") or [])
        for i, fault in enumerate(self.case["provider_faults"]):
            matches = SnapshotToolExecutor._arguments_match(args, self._contracts[i], self._contracts)
            if not matches:
                continue
            if fault["error_code"] == "UPSTREAM_TIMEOUT" and i in self._seen_faults:
                continue
            self._seen_faults.add(i)
            self._search_outcomes[-1] = fault
            return ToolResult(data=None, data_source="unavailable", confidence=0.0, fallback_reason=fault.get("fallback_reason"))
        found = research_fixture.search_pois(self.case, args)
        # Retain observed query->result facts even where old POI tags were sparse.
        for i, fact in enumerate(self.case["provider_query_facts"], len(self.case["provider_faults"])):
            if keywords and SnapshotToolExecutor._arguments_match(args, self._contracts[i], self._contracts):
                found = [p for p in research_fixture.catalog(self.case) if p["name"] in fact["names"]]
        return ToolResult(data=found, data_source="built_in", confidence=1.0)

    async def execute(self, tool_calls, guard_context=None):
        # Evaluate the entire batch once so call limits and duplicate detection
        # remain authoritative. Only executed search handlers enqueue outcomes.
        self._search_outcomes = []
        records = await super().execute(tool_calls, guard_context)
        outcomes = iter(self._search_outcomes)
        for call, record in zip(tool_calls, records, strict=True):
            if record["name"] == "search_pois" and record["guard"]["allowed"]:
                f = next(outcomes, None)
            else:
                f = None
            if f:
                code = "TOOL_TIMEOUT" if f["error_code"] == "UPSTREAM_TIMEOUT" else f["error_code"]
                record["observation"] = ObservationEnvelope.failure(tool="search_pois", code=code,
                    message=f.get("fallback_reason") or code, retryable=bool(f.get("retryable")),
                    tool_call_id=record["tool_call_id"], details={"fixture_revision": VERSION}).model_dump()
            self.calls.append({"call": deepcopy(call), "record": deepcopy(record)})
        return records
