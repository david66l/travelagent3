"""One bounded self-repair policy wrapper.

Production-active: wraps the runtime policy so a single repairable policy
failure can be retried with feedback before the batch fails.  Wired from
``agentic.integration``.
"""

from __future__ import annotations
import json
from typing import Any
from agentic.loop import PolicyAction, PolicyContext
from agentic.policy_actions import (
    controller_override_attempt,
    validate_policy_arguments_for_state,
)
from agentic.policy_routing import _normalize_policy_action
from agentic.policy_controller import PolicyOutputError

_REPAIRABLE_POLICY_ERROR_CODES = frozenset(
    {
        "ACTION_NOT_ALLOWED",
        "ARGUMENT_VALIDATION_FAILED",
        "POLICY_OUTPUT_ERROR",
        "POLICY_OUTPUT_MALFORMED",
        "POLICY_ARGUMENT_INVALID",
        "REPEATED_NO_PROGRESS_ACTION",
        "TOOL_CALL_PARSE_ERROR",
        "TOOL_CALL_SHAPE_ERROR",
    }
)
_ARGUMENT_CHANGE_REQUIRED_CODES = frozenset(
    {
        "ACTION_NOT_ALLOWED",
        "ARGUMENT_VALIDATION_FAILED",
        "INVALID_ARGUMENTS",
        "INVALID_TOOL_ARGUMENTS",
        "QUERY_TOO_BROAD",
        "SNAPSHOT_ARGUMENT_MISMATCH",
        "TOOL_NOT_ALLOWED",
    }
)


# Only a prerequisite-producing action can unlock a retry. A successful weather
# lookup cannot resolve a missing candidate set. Other failures retain the
# existing retry bound until their recovery condition is defined.
_PREREQUISITE_PROGRESS_ACTIONS = {
    "CANDIDATES_REQUIRED": frozenset({"search_pois"}),
}
_MISSING_ARTIFACT_PROGRESS_ACTIONS = {
    "poi_candidate_set": "search_pois",
    "poi_detail_set": "get_poi_detail",
    "route_matrix": "get_route_matrix",
    "weather_snapshot": "get_weather",
    "current_info_search": "search_current_info",
    "event_search_result": "search_current_info",
    "transport_search_result": "search_transport",
}


def _prerequisite_progress_actions(failure: dict[str, Any]) -> set[str]:
    code = str(failure.get("code") or "")
    actions = set(_PREREQUISITE_PROGRESS_ACTIONS.get(code, ()))
    if code == "RESEARCH_EVIDENCE_INSUFFICIENT":
        # The verifier names the missing evidence. Do not unlock missing-route
        # retries merely because an unrelated research call succeeded.
        for issue in str(failure.get("message") or "").split(", "):
            prefix, _, kind = issue.partition(":")
            if prefix == "MISSING_ARTIFACT" and kind in _MISSING_ARTIFACT_PROGRESS_ACTIONS:
                actions.add(_MISSING_ARTIFACT_PROGRESS_ACTIONS[kind])
    return actions


class SelfRepairingAgentPolicy:
    """Give malformed or no-progress model decisions one bounded repair turn.

    This wrapper repairs only policy-owned output errors. Provider failures and
    controller contract errors still fail fast, so a retry cannot hide an
    outage or manufacture authority that the task graph did not grant.
    """

    def __init__(self, delegate: Any, *, max_repair_attempts: int = 1) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must not be negative")
        self.delegate = delegate
        self.max_repair_attempts = max_repair_attempts

    async def propose(self, context: PolicyContext) -> PolicyAction:
        repair_context = context
        repair_codes: list[str] = []
        failed_token_usage = 0

        for attempt in range(self.max_repair_attempts + 1):
            proposed: Any = None
            try:
                proposed = await self.delegate.propose(repair_context)
                action = _normalize_policy_action(proposed)
                action = _validate_repairable_action(repair_context, action)
                if not repair_codes:
                    return action
                return action.model_copy(
                    update={
                        "token_usage": action.token_usage + failed_token_usage,
                        "repair_attempts": len(repair_codes),
                        "repair_error_codes": repair_codes,
                    }
                )
            except PolicyOutputError as exc:
                failed_token_usage += int(
                    getattr(proposed, "token_usage", 0) or _policy_last_token_usage(self.delegate)
                )
                if (
                    attempt >= self.max_repair_attempts
                    or exc.code not in _REPAIRABLE_POLICY_ERROR_CODES
                ):
                    raise
                repair_codes.append(exc.code)
                repair_context = _with_policy_repair_feedback(
                    repair_context,
                    code=exc.code,
                    message=str(exc),
                    attempt=attempt + 1,
                )


def _validate_repairable_action(context: PolicyContext, action: PolicyAction) -> PolicyAction:
    if action.action not in context.allowed_actions:
        raise PolicyOutputError(
            f"policy proposed {action.action}, allowed: {context.allowed_actions}",
            code="ACTION_NOT_ALLOWED",
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
    normalized = action.model_copy(
        update={
            "arguments": arguments,
            "model_arguments": raw_arguments,
            "controller_override_attempt": override_attempt,
            "model_contract_compliant": not override_attempt,
        }
    )
    if _repeats_failed_action_without_progress(context, normalized):
        raise PolicyOutputError(
            "policy repeated an action and arguments that already failed without progress",
            code="REPEATED_NO_PROGRESS_ACTION",
            output_summary={"tool_call_count": 1, "actions": [action.action]},
        )
    return normalized


def _repeats_failed_action_without_progress(context: PolicyContext, action: PolicyAction) -> bool:
    exact_failures = []
    for failure in context.failure_summary:
        if failure.get("failure_id") in context.resolved_precondition_failures:
            continue
        if failure.get("attempted_strategy") != action.action:
            continue
        attempted_arguments = failure.get("attempted_arguments")
        if not isinstance(attempted_arguments, dict):
            continue
        if _canonical_policy_arguments(attempted_arguments) != _canonical_policy_arguments(
            action.arguments
        ):
            continue
        exact_failures.append(failure)
    if not exact_failures:
        return False
    if str(exact_failures[-1].get("code") or "") in _ARGUMENT_CHANGE_REQUIRED_CODES:
        return True
    # New verified observations can resolve a missing prerequisite. Count
    # retries since that progress, rather than permanently banning an action
    # because the same arguments failed in a different evidence state.
    progress_actions = set().union(
        *(_prerequisite_progress_actions(failure) for failure in exact_failures)
    )
    task_id = context.current_subtask.get("task_id")
    history = [item for item in context.decision_history if item.get("task_id") == task_id]
    last_progress = next(
        (
            index
            for index in range(len(history) - 1, -1, -1)
            if history[index].get("outcome_status") == "completed"
            and history[index].get("progress_made") is True
            and history[index].get("action") in progress_actions
        ),
        None,
    )
    if last_progress is not None:
        return (
            sum(
                item.get("action") == action.action
                and item.get("outcome_status") == "failed"
                and _canonical_policy_arguments(item.get("arguments") or {})
                == _canonical_policy_arguments(action.arguments)
                for item in history[last_progress + 1 :]
            )
            >= 2
        )
    return len(exact_failures) >= 2


def _canonical_policy_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)


def _with_policy_repair_feedback(
    context: PolicyContext,
    *,
    code: str,
    message: str,
    attempt: int,
) -> PolicyContext:
    feedback = [
        *context.policy_feedback,
        {
            "code": code,
            "attempt": attempt,
            "message": message[:300],
            "instruction": (
                "Return exactly one allowed action with schema-valid, grounded arguments. "
                "When the same arguments already failed, change the strategy or arguments."
            ),
        },
    ]
    return context.model_copy(deep=True, update={"policy_feedback": feedback})


def _policy_last_token_usage(policy: Any) -> int:
    client = getattr(policy, "client", None)
    if client is not None:
        return int(getattr(client, "last_token_usage", 0) or 0)
    delegate = getattr(policy, "delegate", None)
    if delegate is not None and delegate is not policy:
        return _policy_last_token_usage(delegate)
    return 0
