"""Routing decisions and routed policy implementations.

Mixed-active: ``route_policy_context`` feeds the generalist/specialist
routing used by student/teacher inference; the ``RoutedAgentPolicy`` family
wraps local checkpoints for specialist surfaces (POI detail, verifier
repair) under shadow/canary evaluation.
"""

from __future__ import annotations
import logging
from contextvars import ContextVar
from typing import Any, Literal
from pydantic import BaseModel, Field
from agentic.loop import PolicyAction, PolicyContext, PolicyRouteTrace
from agentic.policy_actions import (
    controller_override_attempt,
    validate_policy_arguments_for_state,
)
from agentic.verifier_repair import (
    VERIFIER_REPAIR_ACTIONS,
    VERIFIER_REPAIR_VIOLATION_CODES,
    constraint_handling_for_violation,
    parse_constraint_flexibility,
    relaxation_options_for_violation,
)
from core.inference_metrics import InferenceMetrics
from agentic.policy_controller import (
    PolicyOutputError,
    _missing_tool_recovery_action,
    _missing_tool_recovery_exhausted,
)

logger = logging.getLogger(__name__)

class PolicyDecision(BaseModel):
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class PolicyRouteDecision(BaseModel):
    """Auditable student/teacher decision made before model inference."""

    target: Literal["student", "teacher"]
    family: Literal["clarification", "search", "recovery", "tradeoff", "complex"]
    reason: str
    fallback_used: bool = False
    fallback_error_code: str | None = None


def route_policy_context(context: PolicyContext) -> PolicyRouteDecision:
    """Route frequent bounded actions to the student and rare choices to the teacher."""
    status = str(context.capability.get("status") or "")
    task_id = str(context.current_subtask.get("task_id") or "")
    allowed = set(context.allowed_actions)

    if status == "missing_tool":
        if (
            not _missing_tool_recovery_exhausted(context)
            and _missing_tool_recovery_action(context) is not None
        ):
            return PolicyRouteDecision(
                target="student",
                family="recovery",
                reason="retryable tool failure has controller-owned retry budget remaining",
            )
        return PolicyRouteDecision(
            target="student",
            family="tradeoff",
            reason=("bounded recovery termination is covered by the distilled student curriculum"),
        )
    if status == "needs_user" or context.missing_information:
        return PolicyRouteDecision(
            target="student",
            family="clarification",
            reason="missing user-provided information",
        )
    if status in {"infeasible", "unsafe"}:
        return PolicyRouteDecision(
            target="student",
            family="tradeoff",
            reason="bounded tradeoff or safe termination is in the student curriculum",
        )
    if task_id == "search_candidates" or allowed == {"search_pois"}:
        family: Literal["search", "recovery"] = "recovery" if context.failure_summary else "search"
        return PolicyRouteDecision(
            target="student",
            family=family,
            reason=(
                "recover a failed search with a bounded retry"
                if family == "recovery"
                else "high-frequency candidate search"
            ),
        )
    if task_id == "review_itinerary":
        return PolicyRouteDecision(
            target="teacher",
            family="complex",
            reason="verifier failure requires a grounded repair or tradeoff decision",
        )
    if {
        "propose_tradeoff",
        "abort",
    } & allowed:
        return PolicyRouteDecision(
            target="student",
            family="tradeoff",
            reason="bounded tradeoff or safe termination is in the student curriculum",
        )
    return PolicyRouteDecision(
        target="teacher",
        family="complex",
        reason="action is outside the student's bounded high-frequency curriculum",
    )



class RoutedAgentPolicy:
    """Route policy calls between a distilled student and a stronger teacher.

    A failed student inference receives exactly one teacher fallback. Teacher
    failures are never swallowed.
    """

    def __init__(self, student: Any, teacher: Any) -> None:
        self.student = student
        self.teacher = teacher
        self._last_route: ContextVar[PolicyRouteDecision | None] = ContextVar(
            f"agent_policy_route_{id(self)}", default=None
        )

    @property
    def last_route(self) -> PolicyRouteDecision | None:
        return self._last_route.get()

    async def propose(self, context: PolicyContext) -> PolicyAction:
        route = route_policy_context(context)
        self._last_route.set(route)
        if route.target == "teacher":
            action = await self.teacher.propose(context)
            return _with_route_trace(action, route, executed_target="teacher")
        try:
            action = await self.student.propose(context)
            return _with_route_trace(action, route, executed_target="student")
        except Exception as exc:
            error_code = str(getattr(exc, "code", type(exc).__name__))
            self._last_route.set(
                route.model_copy(
                    update={
                        "fallback_used": True,
                        "fallback_error_code": error_code,
                    }
                )
            )
            logger.warning(
                "Student policy failed for %s; falling back to teacher: %s",
                route.family,
                error_code,
            )
            fallback_route = self._last_route.get()
            action = await self.teacher.propose(context)
            return _with_route_trace(
                action,
                fallback_route or route,
                executed_target="teacher",
            )


def is_poi_detail_specialist_state(context: PolicyContext) -> bool:
    """Return whether verified state matches the narrow RL specialist contract."""
    if context.failure_summary or "get_poi_detail" not in context.allowed_actions:
        return False
    artifact_types = {
        str(artifact.get("artifact_type") or "") for artifact in context.relevant_artifacts
    }
    if not {"city_knowledge", "poi_candidate_set"}.issubset(artifact_types):
        return False
    if "poi_detail_set" in artifact_types:
        return False

    hard = context.hard_constraints
    information_needs = set(hard.get("information_needs") or [])
    required_before_detail: set[str] = set()
    if "weather" in information_needs:
        required_before_detail.add("weather_snapshot")
    if str(hard.get("intent_kind") or "") == "event_trip" or "event" in information_needs:
        required_before_detail.add("event_search_result")
    if hard.get("transport_modes_requested") or "transport" in information_needs:
        required_before_detail.add("transport_search_result")
    if information_needs.intersection(
        {"opening_hours", "closure", "restaurant", "seasonal_activity", "general"}
    ):
        required_before_detail.add("current_info_search")
    return required_before_detail.issubset(artifact_types)


class DecisionSpecialistRoutedAgentPolicy:
    """Use a GRPO adapter only inside its measured decision-state support.

    The SFT policy remains the general production policy. The specialist gets
    one bounded decision and falls back to SFT on any inference or validation
    error. Both adapters can be served by one vLLM base model with multi-LoRA.
    """

    def __init__(self, generalist: Any, poi_detail_specialist: Any) -> None:
        self.generalist = generalist
        self.poi_detail_specialist = poi_detail_specialist
        self._last_route: ContextVar[PolicyRouteDecision | None] = ContextVar(
            f"agent_decision_specialist_route_{id(self)}", default=None
        )

    @property
    def last_route(self) -> PolicyRouteDecision | None:
        return self._last_route.get()

    def set_rollout_seed(self, seed: int) -> None:
        """Keep paired benchmark sampling identical across both routed arms."""
        for policy in (self.generalist, self.poi_detail_specialist):
            setter = getattr(policy, "set_rollout_seed", None)
            if callable(setter):
                setter(seed)

    async def propose(self, context: PolicyContext) -> PolicyAction:
        if not is_poi_detail_specialist_state(context):
            route = PolicyRouteDecision(
                target="teacher",
                family="complex",
                reason="state is outside the measured GRPO decision-specialist support",
            )
            self._last_route.set(route)
            action = await self.generalist.propose(context)
            return _with_route_trace(action, route, executed_target="teacher")

        route = PolicyRouteDecision(
            target="student",
            family="search",
            reason="verified POI candidates are ready for the GRPO detail-decision specialist",
        )
        self._last_route.set(route)
        specialist_token_usage = 0
        try:
            action = await self.poi_detail_specialist.propose(context)
            specialist_token_usage = int(getattr(action, "token_usage", 0) or 0)
            if action.action != "get_poi_detail":
                raise PolicyOutputError(
                    f"POI decision specialist returned unsupported action: {action.action}",
                    code="SPECIALIST_SCOPE_VIOLATION",
                )
            return _with_route_trace(action, route, executed_target="student")
        except Exception as exc:
            error_code = str(getattr(exc, "code", type(exc).__name__))
            fallback_route = route.model_copy(
                update={"fallback_used": True, "fallback_error_code": error_code}
            )
            self._last_route.set(fallback_route)
            logger.warning(
                "POI decision specialist failed; falling back to SFT generalist: %s",
                error_code,
            )
            action = await self.generalist.propose(context)
            fallback_action = _with_route_trace(
                action,
                fallback_route,
                executed_target="teacher",
            )
            return fallback_action.model_copy(
                update={
                    "token_usage": fallback_action.token_usage + specialist_token_usage,
                }
            )


def is_verifier_repair_specialist_state(context: PolicyContext) -> bool:
    """Recognize bounded review states using policy-visible verifier artifacts only."""
    if str(context.current_subtask.get("task_id") or "") != "review_itinerary":
        return False
    if context.missing_information or context.capability.get("status") == "needs_user":
        return False
    if not (VERIFIER_REPAIR_ACTIONS & set(context.allowed_actions)):
        return False
    contract = parse_constraint_flexibility(
        context.hard_constraints.get("constraint_flexibility")
    )
    if contract is None:
        return False
    reports = [
        artifact
        for artifact in context.relevant_artifacts
        if str(artifact.get("artifact_type") or "") == "validation_report"
    ]
    if not reports or reports[-1].get("hard_pass") is not False:
        return False
    violations = reports[-1].get("violations") or []
    if not isinstance(violations, list) or len(violations) != 1:
        return False
    violation_codes = {
        str(code)
        for code in reports[-1].get("violation_codes") or []
        if str(code).strip()
    }
    visible_violation_codes = {
        str(item.get("code"))
        for item in violations
        if isinstance(item, dict) and str(item.get("code") or "").strip()
    }
    # Fail closed for mixed reports.  A supported scheduling violation plus a
    # closure/reservation failure still requires the generalist's search or
    # clarification actions, which this bounded specialist was not trained on.
    if (
        violation_codes != visible_violation_codes
        or len(violation_codes) != 1
        or not violation_codes <= VERIFIER_REPAIR_VIOLATION_CODES
    ):
        return False
    violation = violations[0]
    code = next(iter(violation_codes))
    message = str(violation.get("message") or "").strip()
    if not message or context.capability.get("evidence") != [message]:
        return False
    handling = constraint_handling_for_violation(code, contract)
    if handling is None:
        return False
    retry_count = int(
        (context.current_subtask.get("action_attempt_counts") or {}).get(
            "retry_solve", 0
        )
        or 0
    )
    if handling == "solver_adjustable":
        capability_matches = (
            context.capability.get("status") == "solvable" and retry_count == 0
        )
    elif handling == "relaxable":
        authorized_options = relaxation_options_for_violation(code, contract)
        capability_matches = (
            bool(authorized_options)
            and context.capability.get("status") == "infeasible"
            and context.capability.get("actionable_alternatives") is True
            and context.capability.get("alternatives") == authorized_options
        )
    else:
        capability_matches = (
            context.capability.get("status") == "infeasible"
            and context.capability.get("actionable_alternatives") is False
            and not context.capability.get("alternatives")
        )
    if not capability_matches:
        return False
    return any(
        str(artifact.get("artifact_type") or "") == "solver_result"
        for artifact in context.relevant_artifacts
    )


class VerifierRepairSpecialistRoutedAgentPolicy:
    """Bound one verifier-repair adapter to its measured three-action support."""

    def __init__(self, generalist: Any, specialist: Any) -> None:
        self.generalist = generalist
        self.specialist = specialist
        self._last_route: ContextVar[PolicyRouteDecision | None] = ContextVar(
            f"agent_verifier_repair_specialist_route_{id(self)}", default=None
        )

    @property
    def last_route(self) -> PolicyRouteDecision | None:
        return self._last_route.get()

    def set_rollout_seed(self, seed: int) -> None:
        for policy in (self.generalist, self.specialist):
            setter = getattr(policy, "set_rollout_seed", None)
            if callable(setter):
                setter(seed)

    async def propose(self, context: PolicyContext) -> PolicyAction:
        if not is_verifier_repair_specialist_state(context):
            route = PolicyRouteDecision(
                target="teacher",
                family="complex",
                reason="state is outside verified verifier-repair specialist support",
            )
            self._last_route.set(route)
            action = await self.generalist.propose(context)
            return _with_route_trace(action, route, executed_target="teacher")

        route = PolicyRouteDecision(
            target="student",
            family="recovery",
            reason="failed verifier report is inside bounded three-action repair support",
        )
        self._last_route.set(route)
        specialist_token_usage = 0
        try:
            action = _normalize_policy_action(await self.specialist.propose(context))
            specialist_token_usage = action.token_usage
            if (
                action.action not in VERIFIER_REPAIR_ACTIONS
                or action.action not in context.allowed_actions
            ):
                raise PolicyOutputError(
                    f"Verifier-repair specialist returned unsupported action: {action.action}",
                    code="SPECIALIST_SCOPE_VIOLATION",
                )
            raw_arguments = dict(
                action.model_arguments if action.model_arguments is not None else action.arguments
            )
            try:
                arguments = validate_policy_arguments_for_state(
                    action.action,
                    raw_arguments,
                    capability=context.capability,
                )
            except ValueError as exc:
                raise PolicyOutputError(
                    str(exc),
                    code="ARGUMENT_VALIDATION_FAILED",
                    detail_code=getattr(exc, "rejection_code", None),
                ) from exc
            override_attempt = controller_override_attempt(action.action, raw_arguments)
            action = action.model_copy(
                update={
                    "arguments": arguments,
                    "model_arguments": raw_arguments,
                    "controller_override_attempt": override_attempt,
                    "model_contract_compliant": not override_attempt,
                }
            )
            return _with_route_trace(action, route, executed_target="student")
        except Exception as exc:
            error_code = str(getattr(exc, "code", type(exc).__name__))
            fallback_route = route.model_copy(
                update={"fallback_used": True, "fallback_error_code": error_code}
            )
            self._last_route.set(fallback_route)
            logger.warning(
                "Verifier-repair specialist failed; falling back to SFT generalist: %s",
                error_code,
            )
            fallback = _with_route_trace(
                await self.generalist.propose(context),
                fallback_route,
                executed_target="teacher",
            )
            return fallback.model_copy(
                update={"token_usage": fallback.token_usage + specialist_token_usage}
            )



def _normalize_policy_action(action: Any) -> PolicyAction:
    if isinstance(action, PolicyAction):
        return action
    return PolicyAction(
        action=str(action.action),
        arguments=dict(getattr(action, "arguments", {}) or {}),
    )


def _with_route_trace(
    action: Any,
    route: PolicyRouteDecision,
    *,
    executed_target: Literal["student", "teacher"],
) -> PolicyAction:
    """Normalize structural policy results and attach auditable route evidence."""
    action = _normalize_policy_action(action)
    return action.model_copy(
        update={
            "route_trace": PolicyRouteTrace(
                requested_target=route.target,
                executed_target=executed_target,
                family=route.family,
                reason=route.reason,
                fallback_used=route.fallback_used,
                fallback_error_code=route.fallback_error_code,
            )
        }
    )


def _last_inference_metrics(client: Any) -> InferenceMetrics | None:
    """Return only a completed metrics snapshot from a compatible client."""
    metrics = getattr(client, "last_request_metrics", None)
    return metrics if isinstance(metrics, InferenceMetrics) else None
