"""Controller-owned action policy logic.

Production-active: the controller decides permission-free deterministic
transitions (CONTROLLER_TASK_ACTIONS), constrains the model-visible context
and drives research-gap recovery.  Wired from ``agentic.integration``.
"""

from __future__ import annotations
from typing import Any
from agentic.loop import PolicyAction, PolicyContext
from agentic.policy_actions import (
    controller_tradeoff_options,
    model_visible_policy_actions,
)

class PolicyOutputError(ValueError):
    """Raised when a policy proposes an action outside controller authority."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "POLICY_OUTPUT_ERROR",
        detail_code: str | None = None,
        raw_output: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail_code = detail_code
        # Keep offline diagnostics bounded; callers must not rely on this as
        # part of the online policy contract.
        self.raw_output = raw_output[:2000] if raw_output is not None else None


# These tasks have one correct controller-owned transition after gathering has
# established that the request is solvable. Calling a model for them creates
CONTROLLER_TASK_ACTIONS: dict[str, str] = {
    "capability_check": "capability_check",
    "collect_weather": "get_weather",
    "collect_poi_details": "get_poi_detail",
    "collect_route_matrix": "get_route_matrix",
    "solve_itinerary": "solve_itinerary",
    "validate_itinerary": "validate_itinerary",
    "compose_draft": "compose_draft",
    "await_confirmation": "finish",
}

_RESEARCH_ARTIFACT_ACTIONS = {
    "city_knowledge": "retrieve_city_knowledge",
    "poi_candidate_set": "search_pois",
    "poi_detail_set": "get_poi_detail",
    "weather_snapshot": "get_weather",
    "current_info_search": "search_current_info",
    "event_search_result": "search_current_info",
    "transport_search_result": "search_transport",
    "route_matrix": "get_route_matrix",
}

_RESEARCH_ACTION_ATTEMPT_LIMIT = 2



class ControllerFirstPolicy:
    """Execute mandatory transitions directly and delegate real choices.

    Search strategy, clarification and recovery remain model decisions. Solver,
    validation and completion gates remain deterministic controller decisions.
    """

    def __init__(self, delegate) -> None:
        self.delegate = delegate

    async def propose(self, context: PolicyContext) -> PolicyAction:
        decision = controller_policy_action(context)
        if decision is not None:
            return decision
        return await self.delegate.propose(context)



def controller_policy_action(context: PolicyContext) -> PolicyAction | None:
    """Return the production controller transition, or defer a real choice.

    The helper is shared by online inference and the stateful RL environment so
    training cannot accidentally optimize actions that production never asks a
    model to choose.
    """
    task_id = str(context.current_subtask.get("task_id") or "")
    action = CONTROLLER_TASK_ACTIONS.get(task_id)
    if task_id == "research_evidence":
        gap_actions = _research_gap_actions(context) or _declared_research_gap_actions(context)
        if gap_actions and _research_recovery_exhausted(context, gap_actions):
            action = None
        elif len(gap_actions) == 1:
            action = gap_actions[0]
        elif not gap_actions and _declared_research_requirements_satisfied(context):
            action = "finalize_research"
        else:
            action = None
    if task_id == "capability_check" and context.capability.get("status") != "solvable":
        action = None
    if task_id == "review_itinerary":
        latest_report = next(
            (
                item
                for item in reversed(context.relevant_artifacts)
                if item.get("artifact_type") == "validation_report"
            ),
            None,
        )
        action = "accept_itinerary" if latest_report and latest_report.get("hard_pass") else None
    if action and action in context.allowed_actions:
        arguments: dict[str, Any] = {}
        if action == "get_weather":
            start_date = context.hard_constraints.get("start_date")
            if start_date:
                arguments["date"] = str(start_date)
        if action == "search_current_info":
            arguments = _current_info_arguments(context)
        return PolicyAction(
            action=action,
            arguments=arguments,
            decision_source="controller",
        )
    return None


def _current_info_arguments(context: PolicyContext) -> dict[str, Any]:
    """Ground one generic web-search call in the current unresolved evidence gap."""
    present = {str(item.get("artifact_type") or "") for item in context.relevant_artifacts}
    latest_failure = next(
        (
            str(failure.get("message") or "")
            for failure in reversed(context.failure_summary)
            if failure.get("code") == "RESEARCH_EVIDENCE_INSUFFICIENT"
        ),
        "",
    )
    criteria = context.current_subtask.get("success_criteria") or {}
    required = set(criteria.get("research_required_artifact_types") or [])
    needs_event = "event_search_result" not in present and (
        "MISSING_ARTIFACT:event_search_result" in latest_failure
        or "event_search_result" in required
        or context.hard_constraints.get("intent_kind") == "event_trip"
    )
    information_needs = list(context.hard_constraints.get("information_needs") or [])
    current_queries = list(context.hard_constraints.get("current_info_queries") or [])
    query = (
        str(
            context.hard_constraints.get("event_query")
            if needs_event
            else (current_queries[0] if current_queries else context.original_request)
        ).strip()
        or context.original_request.strip()
    )
    if needs_event:
        info_type = "event"
    else:
        info_type = next(
            (
                item
                for item in information_needs
                if item
                in {
                    "opening_hours",
                    "closure",
                    "restaurant",
                    "seasonal_activity",
                    "general",
                }
            ),
            "general",
        )
    arguments: dict[str, Any] = {"query": query[:160], "info_type": info_type}
    start_date = context.hard_constraints.get("start_date")
    if start_date:
        arguments["date"] = str(start_date)
    return arguments


def _research_gap_actions(context: PolicyContext) -> list[str]:
    """Return currently unresolved actions from the latest evidence-verifier failure.

    The mapping is applied only after ``finalize_research`` produced a concrete
    programmatic failure.  Multiple gaps remain a policy decision within this
    dynamic subset; a single gap can be closed directly by the controller.
    """
    latest = next(
        (
            failure
            for failure in reversed(context.failure_summary)
            if failure.get("code") == "RESEARCH_EVIDENCE_INSUFFICIENT"
        ),
        None,
    )
    if latest is None:
        return []
    message = str(latest.get("message") or "")
    present_artifacts = {
        str(item.get("artifact_type") or "") for item in context.relevant_artifacts
    }
    gap_actions: list[str] = []
    for marker, artifact_type, action, requires_missing_artifact in (
        (
            "MISSING_ARTIFACT:city_knowledge",
            "city_knowledge",
            "retrieve_city_knowledge",
            True,
        ),
        (
            "MISSING_ARTIFACT:poi_candidate_set",
            "poi_candidate_set",
            "search_pois",
            True,
        ),
        (
            "MISSING_ARTIFACT:poi_detail_set",
            "poi_detail_set",
            "get_poi_detail",
            True,
        ),
        (
            "MISSING_ARTIFACT:weather_snapshot",
            "weather_snapshot",
            "get_weather",
            True,
        ),
        (
            "MISSING_ARTIFACT:event_search_result",
            "event_search_result",
            "search_current_info",
            True,
        ),
        (
            "MISSING_ARTIFACT:transport_search_result",
            "transport_search_result",
            "search_transport",
            True,
        ),
        (
            "MISSING_ARTIFACT:route_matrix",
            "route_matrix",
            "get_route_matrix",
            True,
        ),
        ("INSUFFICIENT_CANDIDATES:", None, "search_pois", False),
        ("INSUFFICIENT_POI_DETAILS:", None, "get_poi_detail", False),
        ("INVALID_ROUTE_MATRIX:", None, "get_route_matrix", False),
        ("CURRENT_INFO_NOT_PLANNABLE", None, "search_current_info", False),
        (
            "UNVERIFIED_LIVE_EVIDENCE:current_info_search",
            None,
            "search_current_info",
            False,
        ),
        (
            "STALE_OR_UNTIMED_ARTIFACT:current_info_search",
            None,
            "search_current_info",
            False,
        ),
        ("SOURCE_MISSING:current_info_search", None, "search_current_info", False),
        ("EVENT_FIELDS_INCOMPLETE", None, "search_current_info", False),
        ("EVENT_VENUE_UNGROUNDED", None, "search_current_info", False),
        ("EVENT_SOURCE_MISSING", None, "search_current_info", False),
        (
            "UNVERIFIED_LIVE_EVIDENCE:event_search_result",
            None,
            "search_current_info",
            False,
        ),
        (
            "STALE_OR_UNTIMED_ARTIFACT:event_search_result",
            None,
            "search_current_info",
            False,
        ),
        ("TRANSPORT_SCHEDULE_NOT_PLANNABLE", None, "search_transport", False),
        (
            "UNVERIFIED_LIVE_EVIDENCE:transport_search_result",
            None,
            "search_transport",
            False,
        ),
        (
            "STALE_OR_UNTIMED_ARTIFACT:transport_search_result",
            None,
            "search_transport",
            False,
        ),
        ("SOURCE_MISSING:transport_search_result", None, "search_transport", False),
    ):
        if marker not in message or action not in context.allowed_actions:
            continue
        if requires_missing_artifact and artifact_type in present_artifacts:
            continue
        gap_actions.append(action)
    return list(dict.fromkeys(gap_actions))


def _research_recovery_exhausted(
    context: PolicyContext,
    gap_actions: list[str],
) -> bool:
    """Stop a missing/low-quality evidence action after two observed attempts.

    The task still has a larger total budget because a healthy research pass
    needs several different tools.  Per-action counts prevent one unavailable
    provider from consuming that whole budget.
    """
    raw_counts = context.current_subtask.get("action_attempt_counts") or {}
    if not isinstance(raw_counts, dict):
        return False
    return any(
        int(raw_counts.get(action, 0) or 0) >= _RESEARCH_ACTION_ATTEMPT_LIMIT
        for action in gap_actions
    )


def _declared_research_gap_actions(context: PolicyContext) -> list[str]:
    """Project intent-specific missing evidence into the current action space."""
    criteria = context.current_subtask.get("success_criteria") or {}
    required = list(criteria.get("research_required_artifact_types") or [])
    present = {str(item.get("artifact_type") or "") for item in context.relevant_artifacts}
    return list(
        dict.fromkeys(
            action
            for artifact_type in required
            if artifact_type not in present
            if (action := _RESEARCH_ARTIFACT_ACTIONS.get(str(artifact_type)))
            and action in context.allowed_actions
        )
    )


def _declared_research_requirements_satisfied(context: PolicyContext) -> bool:
    """True only when a non-empty declared evidence set is fully present."""
    criteria = context.current_subtask.get("success_criteria") or {}
    required = {
        str(item) for item in criteria.get("research_required_artifact_types") or [] if item
    }
    if not required:
        return False
    present = {str(item.get("artifact_type") or "") for item in context.relevant_artifacts}
    return required.issubset(present)


def constrain_policy_context(context: PolicyContext) -> PolicyContext:
    """Remove actions that contradict controller-known capability state."""
    status = str(context.capability.get("status") or "")
    allowed = list(context.allowed_actions)
    task_id = str(context.current_subtask.get("task_id") or "")
    narrowed_missing_information = list(context.missing_information)
    if status == "missing_tool":
        recovery_action = _missing_tool_recovery_action(context)
        if not _missing_tool_recovery_exhausted(context) and recovery_action is not None:
            constrained = [recovery_action]
        else:
            constrained = _terminal_capability_actions(context, allowed)
    elif status == "needs_user" or context.missing_information:
        constrained = [action for action in allowed if action == "ask_user"]
        narrowed_missing_information = narrowed_missing_information[:1]
    elif status in {"infeasible", "unsafe"}:
        constrained = _terminal_capability_actions(context, allowed)
    else:
        constrained = allowed
        if recovery_actions := _search_failure_recovery_actions(context, allowed):
            constrained = recovery_actions
        elif task_id == "research_evidence" and (
            gap_actions := (
                _research_gap_actions(context) or _declared_research_gap_actions(context)
            )
        ):
            constrained = (
                _research_tradeoff_actions(context, allowed)
                if _research_recovery_exhausted(context, gap_actions)
                else gap_actions
            )
        elif task_id == "research_evidence" and _declared_research_requirements_satisfied(context):
            constrained = [action for action in allowed if action == "finalize_research"]
        elif task_id == "search_candidates":
            has_candidates = any(
                item.get("artifact_type") == "poi_candidate_set"
                and int(item.get("poi_count") or 0) > 0
                for item in context.relevant_artifacts
            )
            if not has_candidates:
                constrained = [
                    action for action in allowed if action in {"search_pois", "ask_user"}
                ]
        elif task_id == "review_itinerary":
            latest_report = next(
                (
                    item
                    for item in reversed(context.relevant_artifacts)
                    if item.get("artifact_type") == "validation_report"
                ),
                None,
            )
            if latest_report and latest_report.get("hard_pass") is True:
                constrained = [action for action in allowed if action == "accept_itinerary"]
            else:
                constrained = [action for action in allowed if action != "accept_itinerary"]
    constrained = model_visible_policy_actions(
        constrained,
        capability=context.capability,
    )
    updates: dict[str, Any] = {}
    if constrained != allowed:
        current_subtask = dict(context.current_subtask)
        current_subtask["allowed_actions"] = constrained
        updates.update(
            {
                "allowed_actions": constrained,
                "current_subtask": current_subtask,
            }
        )
    if narrowed_missing_information != context.missing_information:
        updates["missing_information"] = narrowed_missing_information
    if not updates:
        return context
    return context.model_copy(deep=True, update=updates)


def _search_failure_recovery_actions(
    context: PolicyContext,
    allowed: list[str],
) -> list[str]:
    """Mask unrelated tools for one verifier-grounded search recovery turn.

    The model still owns the semantically important argument decision: narrow
    after a broad-query error or preserve arguments after a transient timeout.
    The controller only removes actions that cannot repair the failed POI
    candidate search, keeping the online protocol aligned with state-scoped
    SFT/GRPO training.
    """
    if str(context.current_subtask.get("task_id") or "") != "research_evidence":
        return []
    latest_decision = next(
        (
            decision
            for decision in reversed(context.decision_history)
            if decision.get("task_id") == "research_evidence"
        ),
        None,
    )
    if (
        latest_decision is None
        or latest_decision.get("action") != "search_pois"
        or latest_decision.get("outcome_status") != "failed"
    ):
        return []
    latest = next(
        (
            failure
            for failure in reversed(context.failure_summary)
            if failure.get("attempted_strategy") == "search_pois"
        ),
        None,
    )
    if latest is None or not latest.get("retryable"):
        return []
    if int(latest.get("retry_budget_remaining") or 0) <= 0:
        return []
    if str(latest.get("code") or "") not in {
        "QUERY_TOO_BROAD",
        "TOOL_TIMEOUT",
        "UPSTREAM_TIMEOUT",
    }:
        return []
    return [action for action in allowed if action == "search_pois"]


def _terminal_capability_actions(context: PolicyContext, allowed: list[str]) -> list[str]:
    """Use controller-grounded alternative availability to bound termination."""
    terminal = [action for action in allowed if action in {"propose_tradeoff", "abort"}]
    if context.capability.get("actionable_alternatives") is False:
        abort_only = [action for action in terminal if action == "abort"]
        return abort_only or terminal
    return terminal


def _research_tradeoff_actions(context: PolicyContext, allowed: list[str]) -> list[str]:
    """Prefer asking the user over silently abandoning an evidence-limited plan."""
    # A research failure does not by itself authorize changing user constraints.
    # If the capability report supplies exact alternatives the model may explain
    # them; otherwise retain the fail-closed abort action instead of presenting
    # an empty policy surface after model-visible authority filtering.
    if controller_tradeoff_options(context.capability) is not None:
        tradeoff = [action for action in allowed if action == "propose_tradeoff"]
        if tradeoff:
            return tradeoff
    return [action for action in allowed if action == "abort"]



def _missing_tool_recovery_exhausted(context: PolicyContext) -> bool:
    """Match the frozen Stage29 router using policy-visible retry state.

    External benchmark rows carry ``retry_budget_remaining`` explicitly. Live
    trajectories can infer the same value from controller-owned task attempts.
    Missing or malformed recovery evidence is routed conservatively to the
    teacher instead of guessing that another tool call is safe.
    """
    failures = context.failure_summary
    if not failures:
        return True

    task = context.current_subtask
    try:
        inferred_remaining = max(
            0,
            int(task.get("max_attempts", 0)) - int(task.get("attempts", 0)),
        )
    except (TypeError, ValueError):
        inferred_remaining = 0

    for failure in failures:
        if not bool(failure.get("retryable", False)):
            return True
        explicit_remaining = failure.get("retry_budget_remaining")
        if explicit_remaining is None:
            remaining = inferred_remaining
        else:
            try:
                remaining = int(explicit_remaining)
            except (TypeError, ValueError):
                return True
        if remaining <= 0:
            return True
    return False


def _missing_tool_recovery_action(context: PolicyContext) -> str | None:
    """Return the controller-observed failed action that a retry may repeat.

    Live failures use ``attempted_strategy`` while frozen external cases use
    ``action``.  Never infer an action from natural language: without explicit
    controller evidence, the recovery belongs on the conservative teacher path.
    """
    allowed = set(context.allowed_actions)
    for failure in reversed(context.failure_summary):
        candidate = failure.get("attempted_strategy") or failure.get("action")
        if isinstance(candidate, str) and candidate in allowed:
            return candidate
    return None
