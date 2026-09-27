"""Execution authority for the single model-driven loop.

The harness validates proposals and invalidates derived evidence. It never
chooses the next tool, fills semantic queries, or advances a hidden policy.
"""

from __future__ import annotations

from typing import Any

ARCHITECTURE_VERSION = "agent-harness-v1"
HARNESS_REVISION = "evidence-progress-retry-v9"
SESSION_TASK_ID = "travel_agent"


def resolved_precondition_failure_ids(ledger: Any, failures: list[dict]) -> list[str]:
    """Recheck actual prerequisites without executing or selecting an action.

    This internal guard result does not rewrite the visible failure history.
    A changed query or an unrelated successful tool call alone cannot clear it.
    """
    from agentic.loop import PolicyAction

    codes = {
        "CANDIDATES_REQUIRED", "RESEARCH_EVIDENCE_INSUFFICIENT",
        "SOLVER_ARTIFACT_MISSING", "VALIDATION_NOT_PASSED", "STALE_VALIDATION",
    }
    resolved = []
    for failure in failures:
        if failure.get("code") not in codes or not failure.get("failure_id"):
            continue
        action = PolicyAction(action=failure["attempted_strategy"],
                              arguments=failure.get("attempted_arguments") or {})
        if preflight_action(ledger, action) is None:
            resolved.append(failure["failure_id"])
    return resolved


def capability_from_feasibility(ledger: Any, assessment: dict) -> Any:
    from agentic.state import GoalCapability
    from contracts.constraint_flexibility import ConstraintFlexibilityContract
    witnesses = assessment.get("witnesses") or []
    if assessment.get("status") != "infeasible" or not witnesses:
        return GoalCapability(status="undetermined")
    try:
        raw = ledger.goal.hard_constraints.get("constraint_flexibility")
        contract = ConstraintFlexibilityContract.model_validate(raw) if raw else None
    except ValueError:
        contract = None
    option_sets = [set(contract.relaxation_options.get(w["dimension"], []))
                   if contract else set() for w in witnesses]
    options = sorted(set.intersection(*option_sets)) if option_sets else []
    return GoalCapability(status="infeasible", evidence=[w["message"] for w in witnesses],
        evidence_ids=[w["evidence_id"] for w in witnesses],
        actionable_alternatives=bool(options), alternatives=options)


def refresh_constraint_evidence(ledger: Any) -> None:
    """Verify newly observed required facts; never select a model action.

Proofs are available as soon as candidate details arrive. Requiring a model to
run a knowingly impossible solver before stopping would be a hidden workflow.
"""
    from agentic.action_executor import TravelActionExecutor
    from agentic.clock import reference_now
    from agentic.state import ArtifactRecord
    from evaluation.feasibility import assess_required_facts
    details = latest_artifact(ledger, "poi_detail_set")
    if details is None or latest_artifact(ledger, "poi_candidate_set") is None:
        return
    items = TravelActionExecutor._planning_candidate_items_with_evidence(ledger)
    facts = [TravelActionExecutor._poi_input(item,i) for i,item in enumerate(items)]
    assessment = assess_required_facts(TravelActionExecutor._trusted_constraints(ledger), facts)
    if assessment["status"] != "infeasible":
        return
    previous = latest_artifact(ledger, "constraint_evidence")
    if previous is None or previous.payload.get("feasibility") != assessment:
        for key,a in list(ledger.artifacts.items()):
            if a.artifact_type == "constraint_evidence":
                del ledger.artifacts[key]
        artifact = ArtifactRecord(
            artifact_id="constraint-evidence:" + assessment["witnesses"][0]["evidence_id"],
            artifact_type="constraint_evidence", payload={"feasibility":assessment},
            goal_version=ledger.goal.goal_version, plan_version=ledger.task_graph.plan_version,
            created_at=reference_now(), evidence_refs=[details.artifact_id],
            expires_at=min((a.expires_at for a in ledger.artifacts.values()
                            if a.artifact_type in {"poi_candidate_set", "poi_detail_set", "current_info_search"}
                            and a.expires_at is not None), default=None),
        )
        ledger.artifacts[artifact.artifact_id] = artifact
    ledger.goal = ledger.goal.model_copy(update={"capability":capability_from_feasibility(ledger,assessment)})


def assert_session_contract(ledger: Any) -> None:
    tasks = ledger.task_graph.tasks
    if len(tasks) != 1 or tasks[0].task_id != SESSION_TASK_ID or tasks[0].depends_on:
        raise ValueError(
            "ARCHITECTURE_MISMATCH: this runtime accepts agent-harness-v1 sessions only; "
            "start a fresh episode from the user goal, not a legacy DAG checkpoint"
        )


def latest_artifact(ledger: Any, kind: str) -> Any:
    from agentic.clock import reference_now

    now = reference_now()
    return next(
        (
            a
            for a in reversed(list(ledger.artifacts.values()))
            if a.artifact_type == kind
            and a.goal_version == ledger.goal.goal_version
            and a.plan_version == ledger.task_graph.plan_version
            and (a.expires_at is None or a.expires_at > now)
        ),
        None,
    )


def preflight_action(ledger: Any, action: Any) -> Any:
    """Reject invalid execution, returning evidence to the same policy on the next turn."""
    from agentic.loop import ActionOutcome
    from agentic.policy_actions import controller_tradeoff_options

    def reject(code: str, detail: str):
        return ActionOutcome(status="failed", error_code=code, error_message=detail, retryable=True)

    constraints = ledger.goal.hard_constraints
    args = action.arguments
    if action.action == "propose_tradeoff":
        if controller_tradeoff_options(ledger.goal.capability.model_dump(mode="json")) is None:
            return reject(
                "TRADEOFF_NOT_AUTHORIZED", "No user-authorized alternatives are available."
            )
    if action.action == "search_transport" and not constraints.get("origin"):
        return reject("ORIGIN_REQUIRED", "Transport search requires a user-grounded origin.")
    if action.action in {"search_transport", "search_current_info", "get_weather"}:
        # Omitted facts may be injected; conflicting authored values are never silently fixed.
        if args.get("date") and constraints.get("start_date"):
            from datetime import date, timedelta

            try:
                start = date.fromisoformat(str(constraints["start_date"]))
                end = (
                    date.fromisoformat(str(constraints["end_date"]))
                    if constraints.get("end_date")
                    else start
                    + timedelta(days=max(1, int(constraints.get("travel_days") or 1)) - 1)
                )
                proposed = date.fromisoformat(str(args["date"]))
            except (TypeError, ValueError):
                return reject(
                    "DATE_CONSTRAINT_MISMATCH",
                    "Date must be an ISO date within the user's travel window.",
                )
            if not start <= proposed <= end:
                return reject(
                    "DATE_CONSTRAINT_MISMATCH",
                    "The proposed date is outside the user's travel window.",
                )
    if action.action == "search_transport":
        modes = constraints.get("transport_modes_requested") or []
        if len(modes) == 1 and args.get("mode", "both") not in {modes[0]}:
            return reject(
                "TRANSPORT_MODE_CONSTRAINT_MISMATCH",
                "The proposed mode conflicts with the user's requested transport mode.",
            )
    if action.action == "get_poi_detail" and latest_artifact(ledger, "poi_candidate_set") is None:
        return reject("CANDIDATES_REQUIRED", "No candidate POIs have been obtained yet.")
    if action.action == "get_route_matrix" and latest_artifact(ledger, "poi_candidate_set") is None:
        return reject("CANDIDATES_REQUIRED", "A route matrix requires grounded candidate POIs.")
    if action.action == "solve_itinerary":
        missing = [k for k in ("destination", "travel_days") if not constraints.get(k)]
        if missing:
            return reject(
                "PLANNING_INPUT_MISSING", "Missing planning inputs: " + ", ".join(missing)
            )
        from agentic.react import ResearchSufficiencyVerifier

        report = ResearchSufficiencyVerifier().evaluate(ledger)
        if not report.sufficient:
            return reject("RESEARCH_EVIDENCE_INSUFFICIENT", ", ".join(report.missing))
    if (action.action == "validate_itinerary" and latest_artifact(ledger, "solver_result") is None
            and latest_artifact(ledger, "poi_detail_set") is None):
        return reject("SOLVER_ARTIFACT_MISSING", "There is no current itinerary to validate.")
    if action.action in {"abort", "propose_tradeoff"}:
        expected = set(ledger.goal.capability.evidence_ids)
        supplied = set(args.get("evidence_ids") or [])
        if expected or supplied:
            report = latest_artifact(ledger, "constraint_evidence") or latest_artifact(ledger, "validation_report")
            assessment = (report.payload.get("feasibility") or {}) if report else {}
            current_ids = {w.get("evidence_id") for w in assessment.get("witnesses") or []}
            expired_proof = latest_artifact(ledger, "constraint_evidence") is None and any(
                a.artifact_type == "constraint_evidence" and a.goal_version == ledger.goal.goal_version
                and a.plan_version == ledger.task_graph.plan_version for a in ledger.artifacts.values())
            if (assessment.get("status") != "infeasible"
                    or expired_proof or not supplied or not supplied <= expected & current_ids):
                return reject("TERMINAL_EVIDENCE_REFERENCE_INVALID", "Cite current capability.evidence_ids supporting the stop or conflict; prior references are invalid after evidence changes.")
    if action.action == "finish":
        report = latest_artifact(ledger, "validation_report")
        solver = latest_artifact(ledger, "solver_result")
        if solver is None or report is None or report.payload.get("hard_pass") is not True:
            return reject(
                "VALIDATION_NOT_PASSED",
                "A current itinerary and hard-passed validation are required before submission.",
            )
        if report.created_at < solver.created_at:
            return reject(
                "STALE_VALIDATION",
                "The itinerary changed after validation; validate the current version.",
            )
        from agentic.react import ResearchSufficiencyVerifier

        evidence = ResearchSufficiencyVerifier().evaluate(ledger)
        if not evidence.sufficient:
            return reject("RESEARCH_EVIDENCE_INSUFFICIENT", ", ".join(evidence.missing))
    return None


def invalidate_derived_evidence(
    ledger: Any, action: str, *, artifacts: list[Any] | None = None, facts: list[Any] | None = None
) -> None:
    """A changed input invalidates its outputs; this does not schedule replacement actions."""
    from agentic.react import RESEARCH_ACTIONS

    derived = {
        "solver_result",
        "validation_report",
        "verified_itinerary_acceptance",
        "itinerary_draft",
        "finish",
    }
    invalid = set()
    if action in RESEARCH_ACTIONS:
        # IDs, receipt times and expiry extensions do not change the content of
        # current evidence. An expired predecessor is not equivalent evidence.
        # Without an outcome, callers request conservative invalidation.
        changed_types = None
        if artifacts is not None:
            changed_types = {
                item.artifact_type
                for item in artifacts
                if (previous := latest_artifact(ledger, item.artifact_type)) is None
                or previous.payload != item.payload
                or (
                    item.expires_at is not None
                    and (previous.expires_at is None or item.expires_at < previous.expires_at)
                )
            }
            from agentic.clock import reference_now

            now = reference_now()
            current_facts = {
                fact.key: fact
                for fact in ledger.facts.values()
                if fact.goal_version == ledger.goal.goal_version
                and fact.plan_version == ledger.task_graph.plan_version
                and (fact.expires_at is None or fact.expires_at > now)
            }
            changed_facts = any(
                (previous := current_facts.get(fact.key)) is None
                or (previous.value, previous.source, previous.confidence)
                != (fact.value, fact.source, fact.confidence)
                or (
                    fact.expires_at is not None
                    and (previous.expires_at is None or fact.expires_at < previous.expires_at)
                )
                for fact in facts or []
            )
            if not changed_types and not changed_facts:
                return
        invalid |= derived | {"research_bundle", "constraint_evidence"}
        if (changed_types is None and action == "search_pois") or (
            changed_types and "poi_candidate_set" in changed_types
        ):
            invalid |= {"poi_detail_set", "route_matrix"}
        elif changed_types and "poi_detail_set" in changed_types:
            invalid.add("route_matrix")
    elif action == "solve_itinerary":
        invalid |= derived
    elif action == "validate_itinerary":
        invalid |= {
            "validation_report",
            "verified_itinerary_acceptance",
            "itinerary_draft",
            "finish",
        }
    for key, artifact in list(ledger.artifacts.items()):
        if (
            artifact.artifact_type in invalid
            and artifact.goal_version == ledger.goal.goal_version
            and artifact.plan_version == ledger.task_graph.plan_version
        ):
            del ledger.artifacts[key]
            if artifact.artifact_type in {"validation_report", "constraint_evidence"}:
                from agentic.state import GoalCapability
                ledger.goal = ledger.goal.model_copy(update={"capability": GoalCapability(status="undetermined")})
