"""Adapters that project the existing TravelAgent state into Agent Loop state."""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any, Literal

from agentic.planner import DefaultTaskGraphPlanner
from agentic.state import (
    AgentLedgerState,
    BudgetLedger,
    FactRecord,
    GoalCapability,
    GoalLedger,
    PlanVersion,
    StateTransitionError,
    TaskGraphController,
)
from agentic.verifier import SubtaskVerifier
from agentic.termination import CompletionDecision, CompletionGuard
from core.conversation_state import flatten_profile
from core.settings import settings


PolicyMode = Literal["deterministic", "shadow", "agent"]
TaskGraphMode = Literal["configured", "legacy", "react"]

logger = logging.getLogger(__name__)


def _configured_task_graph_planner(mode: TaskGraphMode = "configured"):
    """Select the production ReAct graph while preserving legacy baselines."""
    if mode == "react" or (mode == "configured" and settings.agentic_execution_mode == "react"):
        from agentic.react import ReactTaskGraphPlanner

        return ReactTaskGraphPlanner()
    return DefaultTaskGraphPlanner()


def _project_goal(state: dict[str, Any]) -> GoalLedger:
    flat = flatten_profile(state.get("profile") or {})
    slots = state.get("slots") or {}

    def value(key: str) -> Any:
        slot_value = slots.get(key)
        return slot_value if slot_value not in (None, "", []) else flat.get(key)

    start_date = value("start_date")
    end_date = value("end_date")
    if not start_date:
        start_date, derived_end = _travel_date_bounds(value("travel_dates"), value("travel_days"))
        end_date = end_date or derived_end

    historical_messages = state.get("messages") or (
        (state.get("_conversation_state") or {}).get("recent_messages") or []
    )
    user_history = [
        str(message.get("content") or "")
        for message in historical_messages[-8:]
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    request_text = "\n".join(
        dict.fromkeys([*user_history, str(state.get("user_input") or "")])
    ).strip()
    intent_kind = str(value("intent_kind") or "itinerary")
    transport_modes = list(value("transport_modes_requested") or [])
    information_needs = list(value("information_needs") or [])
    current_info_queries = list(value("current_info_queries") or [])

    hard_constraints = {
        key: item
        for key in (
            "origin",
            "destination",
            "travel_days",
            "start_date",
            "end_date",
            "budget_range",
            "must_visit",
            "must_not_visit",
            "mobility_constraints",
            "max_walk_minutes",
            "max_transit_minutes",
            "has_wheelchair",
            "has_pregnant",
            "food_taboos",
        )
        if (item := value(key)) not in (None, "", [])
    }
    hard_constraints["intent_kind"] = intent_kind
    event_query = value("event_query")
    if event_query:
        hard_constraints["event_query"] = str(event_query)
    if transport_modes:
        hard_constraints["transport_modes_requested"] = transport_modes
    if information_needs:
        hard_constraints["information_needs"] = information_needs
    if current_info_queries:
        hard_constraints["current_info_queries"] = current_info_queries
    if start_date:
        hard_constraints["start_date"] = start_date
    if end_date:
        hard_constraints["end_date"] = end_date
    soft_preferences = {
        key: item
        for key in (
            "interests",
            "food_preferences",
            "transport_preference",
            "hotel_preference",
            "pace_preference",
            "pace",
            "travelers_type",
            "travelers_count",
            "has_children",
            "has_elderly",
            "fatigue_preference",
            "avoid_crowds",
        )
        if (item := value(key)) not in (None, "", [])
    }
    feasibility = state.get("feasibility_report") or {}
    feasible = feasibility.get("feasible", True)
    missing = list(state.get("missing_slots") or [])
    if transport_modes and not value("origin") and "origin" not in missing:
        missing.append("origin")
    reported_status = str(feasibility.get("status") or "")
    if reported_status not in {"solvable", "needs_user", "missing_tool", "infeasible", "unsafe"}:
        reported_status = ""
    capability_status = (
        "needs_user" if missing else (reported_status or ("solvable" if feasible else "infeasible"))
    )
    return GoalLedger(
        original_request=request_text or "Travel planning request",
        success_definition=[
            "generate an executable itinerary",
            "pass deterministic hard-constraint validation",
            "wait for user confirmation",
        ],
        hard_constraints=hard_constraints,
        soft_preferences=soft_preferences,
        missing_information=missing,
        capability=GoalCapability(
            status=capability_status,
            evidence=[
                str(item)
                for item in (feasibility.get("reasons") or feasibility.get("issues") or [])
            ],
            actionable_alternatives=feasibility.get("actionable_alternatives"),
            alternatives=[str(item) for item in (feasibility.get("alternatives") or [])],
        ),
    )


def _travel_date_bounds(raw_dates: Any, travel_days: Any) -> tuple[str | None, str | None]:
    """Project the canonical profile date value into Agent hard constraints."""
    matches = re.findall(r"\d{4}-\d{2}-\d{2}", str(raw_dates or ""))
    if not matches:
        return None, None
    try:
        start = date.fromisoformat(matches[0])
        if len(matches) > 1:
            end = date.fromisoformat(matches[1])
        else:
            duration = max(1, int(travel_days or 1))
            end = start + timedelta(days=duration - 1)
    except (TypeError, ValueError):
        return None, None
    if end < start:
        return None, None
    return start.isoformat(), end.isoformat()


def initialize_agent_ledger(
    state: dict[str, Any],
    *,
    mode: PolicyMode,
    task_graph_mode: TaskGraphMode = "configured",
) -> dict[str, Any]:
    """Initialize authoritative state without changing the legacy output path."""
    if mode == "deterministic":
        return {"policy_mode": mode, "agent_status": "disabled"}
    if state.get("agent_ledger"):
        ledger = AgentLedgerState(**state["agent_ledger"])
        has_goal_projection = bool(state.get("slots") or state.get("profile"))
        projected_goal = _project_goal(state) if has_goal_projection else ledger.goal
        if has_goal_projection and (
            ledger.goal.hard_constraints != projected_goal.hard_constraints
            or ledger.goal.soft_preferences != projected_goal.soft_preferences
            or ledger.goal.missing_information != projected_goal.missing_information
        ):
            projected_goal = projected_goal.model_copy(
                update={"goal_version": ledger.goal.goal_version + 1}
            )
            graph = _configured_task_graph_planner(task_graph_mode).plan(
                projected_goal, plan_version=ledger.task_graph.plan_version + 1
            )
            graph = TaskGraphController().refresh_ready(graph)
            ready = TaskGraphController.ready_tasks(graph)
            # A materially changed goal starts a new episode. Historical facts,
            # artifacts and failures stay in the persisted old ledger/checkpoint,
            # while the new trajectory cannot accidentally consume them.
            revised = AgentLedgerState(goal=projected_goal, task_graph=graph)
            revised.budget = revised.budget.model_copy(
                update={"max_tool_calls": settings.agentic_tool_call_budget}
            )
            return {
                "policy_mode": mode,
                "agent_ledger": revised.model_dump(mode="json"),
                "current_task_id": ready[0].task_id if ready else None,
                "agent_step": 0,
                "subtask_step": 0,
                "agent_status": "initialized",
            }
        ready = TaskGraphController.ready_tasks(ledger.task_graph)
        return {
            "policy_mode": mode,
            "agent_ledger": ledger.model_dump(mode="json"),
            "current_task_id": ready[0].task_id if ready else ledger.current_task_id,
            "agent_status": "initialized",
        }

    projected_goal = _project_goal(state)
    graph = _configured_task_graph_planner(task_graph_mode).plan(projected_goal)
    graph = TaskGraphController().refresh_ready(graph)
    ready = TaskGraphController.ready_tasks(graph)
    ledger = AgentLedgerState(goal=projected_goal, task_graph=graph)
    ledger.budget = ledger.budget.model_copy(
        update={"max_tool_calls": settings.agentic_tool_call_budget}
    )
    return {
        "policy_mode": mode,
        "agent_ledger": ledger.model_dump(mode="json"),
        "current_task_id": ready[0].task_id if ready else None,
        "agent_step": 0,
        "subtask_step": 0,
        "agent_status": "initialized",
    }


def resume_agent_ledger(
    ledger: AgentLedgerState | dict[str, Any],
    *,
    task_id: str,
    user_value: Any,
    fact_key: str,
) -> AgentLedgerState:
    """Resume a blocked task without resetting graph versions or budgets."""
    state = ledger if isinstance(ledger, AgentLedgerState) else AgentLedgerState(**ledger)
    task = state.task_graph.get(task_id)
    if task.status != "blocked":
        raise StateTransitionError("only a blocked task can be resumed by user input")
    observation_ref = f"user:{state.trajectory_id}:{task_id}:{task.attempts}"
    fact = FactRecord(
        fact_id=observation_ref,
        key=fact_key,
        value=user_value,
        observation_ref=observation_ref,
        goal_version=state.goal.goal_version,
        plan_version=state.task_graph.plan_version,
        source="user",
        confidence=1.0,
    )
    state.facts[fact.fact_id] = fact
    if fact_key.startswith("user_input."):
        field = fact_key.removeprefix("user_input.")
        remaining_missing = [item for item in state.goal.missing_information if item != field]
        hard_constraints = dict(state.goal.hard_constraints)
        if field in {
            "origin",
            "destination",
            "travel_days",
            "start_date",
            "end_date",
            "budget_range",
            "must_visit",
            "mobility_constraints",
        }:
            hard_constraints[field] = user_value
        state.goal = state.goal.model_copy(
            update={
                "missing_information": remaining_missing,
                "hard_constraints": hard_constraints,
                "capability": state.goal.capability.model_copy(
                    update={"status": "solvable" if not remaining_missing else "needs_user"}
                ),
            }
        )
    controller = TaskGraphController()
    state.task_graph = controller.transition(
        state.task_graph, task_id, "ready", evidence_refs=[observation_ref]
    )
    required_user_keys = set(task.required_facts) | set(
        task.success_criteria.get("required_fact_keys") or []
    )
    if fact_key not in required_user_keys:
        # Open-ended clarification/tradeoff replies are observations for the
        # next policy turn, not proof that the blocked planning task succeeded.
        state.task_graph = controller.refresh_ready(state.task_graph)
        state.current_task_id = task_id
        state.termination_reason = None
        return state
    state.task_graph = controller.transition(state.task_graph, task_id, "running")
    verification = SubtaskVerifier().verify(
        state.task_graph.get(task_id), facts=state.facts, artifacts=state.artifacts
    )
    if verification.passed:
        state.task_graph = controller.transition(
            state.task_graph,
            task_id,
            "succeeded",
            evidence_refs=verification.evidence_refs,
        )
        state.task_graph = controller.refresh_ready(state.task_graph)
        ready = controller.ready_tasks(state.task_graph)
        state.current_task_id = ready[0].task_id if ready else None
    else:
        state.task_graph = controller.transition(state.task_graph, task_id, "blocked")
        state.current_task_id = task_id
    state.termination_reason = None
    return state


_HARD_REVISION_FIELDS = frozenset(
    {
        "origin",
        "destination",
        "travel_days",
        "start_date",
        "end_date",
        "budget_range",
        "must_visit",
        "must_not_visit",
        "mobility_constraints",
        "max_transit_minutes",
        "intent_kind",
        "event_query",
        "transport_modes_requested",
        "information_needs",
        "current_info_queries",
    }
)
_SOFT_REVISION_FIELDS = frozenset(
    {
        "interests",
        "food_preferences",
        "transport_preference",
        "hotel_preference",
        "pace",
        "travelers_type",
        "travelers_count",
        "has_children",
        "has_elderly",
        "avoid_pois",
    }
)
_LIST_REVISION_FIELDS = frozenset(
    {
        "must_visit",
        "must_not_visit",
        "mobility_constraints",
        "transport_modes_requested",
        "information_needs",
        "current_info_queries",
        "interests",
        "food_preferences",
        "avoid_pois",
    }
)


def _validated_revision_value(field: str, value: Any) -> Any:
    """Validate model-proposed values before the controller mutates the ledger."""
    if field == "travel_days":
        parsed = int(value)
        if not 1 <= parsed <= 30:
            raise StateTransitionError("travel_days must be between 1 and 30")
        return parsed
    if field == "travelers_count":
        parsed = int(value)
        if not 1 <= parsed <= 50:
            raise StateTransitionError("travelers_count must be between 1 and 50")
        return parsed
    if field in {"budget_range", "max_transit_minutes"}:
        scalar = value
        if field == "budget_range" and isinstance(value, dict):
            unsupported = set(value) - {"max", "value", "amount", "total"}
            candidates = [
                value.get(key)
                for key in ("max", "value", "amount", "total")
                if value.get(key) is not None
            ]
            if unsupported or len(candidates) != 1:
                raise StateTransitionError("budget_range object must contain one numeric bound")
            scalar = candidates[0]
        parsed = float(scalar)
        if parsed < 0:
            raise StateTransitionError(f"{field} cannot be negative")
        return parsed
    if field in {"start_date", "end_date"}:
        parsed = str(value).strip()
        try:
            date.fromisoformat(parsed)
        except ValueError as exc:
            raise StateTransitionError(f"{field} must be an ISO date") from exc
        return parsed
    if field in {"has_children", "has_elderly"}:
        if not isinstance(value, bool):
            raise StateTransitionError(f"{field} must be boolean")
        return value
    if field == "intent_kind" and value not in {"itinerary", "event_trip"}:
        raise StateTransitionError("unsupported intent_kind")
    if field == "pace" and value not in {"relaxed", "moderate", "intensive"}:
        raise StateTransitionError("unsupported pace")
    if field == "transport_preference" and value not in {
        "public",
        "taxi",
        "walk",
        "rental_car",
        "mixed",
        "any",
    }:
        raise StateTransitionError("unsupported transport_preference")
    if field in _LIST_REVISION_FIELDS:
        values = value if isinstance(value, list) else [value]
        cleaned = [item for item in values if item not in (None, "")]
        if field == "transport_modes_requested" and any(
            item not in {"flight", "train", "bus", "ferry"} for item in cleaned
        ):
            raise StateTransitionError("unsupported transport mode")
        if field == "information_needs" and any(
            item
            not in {
                "event",
                "transport",
                "weather",
                "opening_hours",
                "closure",
                "restaurant",
                "seasonal_activity",
                "general",
            }
            for item in cleaned
        ):
            raise StateTransitionError("unsupported information need")
        if any(not isinstance(item, str) or len(item) > 200 for item in cleaned):
            raise StateTransitionError(f"{field} must contain short strings")
        return cleaned
    if field in {
        "origin",
        "destination",
        "event_query",
        "transport_preference",
        "hotel_preference",
        "travelers_type",
    }:
        parsed = str(value).strip()
        if not parsed or len(parsed) > 200:
            raise StateTransitionError(f"{field} must be a short non-empty string")
        return parsed
    return value


def _apply_revision_operation(
    hard: dict[str, Any],
    soft: dict[str, Any],
    *,
    field: str,
    operation: str,
    value: Any,
) -> None:
    target = hard if field in _HARD_REVISION_FIELDS else soft
    if field not in _HARD_REVISION_FIELDS | _SOFT_REVISION_FIELDS:
        raise StateTransitionError(f"revision field is not allowed: {field}")
    if operation == "clear":
        target.pop(field, None)
        return
    checked = _validated_revision_value(field, value)
    if operation == "set":
        target[field] = checked
        return
    if field not in _LIST_REVISION_FIELDS:
        raise StateTransitionError(f"{operation} is only valid for list fields: {field}")
    current = list(target.get(field) or [])
    values = list(checked)
    if operation == "add":
        target[field] = list(dict.fromkeys([*current, *values]))
    elif operation == "remove":
        target[field] = [item for item in current if item not in set(values)]
    else:
        raise StateTransitionError(f"unsupported revision operation: {operation}")


async def revise_agent_ledger(
    ledger: AgentLedgerState | dict[str, Any],
    *,
    revision_reason: str,
    interpretation: Any | None = None,
) -> AgentLedgerState:
    """Use model-parsed feedback to version the goal and restart ReAct gates."""
    from models.travel_slots import RevisionParseOutput

    state = ledger if isinstance(ledger, AgentLedgerState) else AgentLedgerState(**ledger)
    reason = str(revision_reason or "").strip()
    if not reason:
        raise StateTransitionError("revision reason cannot be empty")

    parse_source = "provided"
    if interpretation is None:
        from agents.demand_parser import DemandParserAgent

        try:
            interpretation = await DemandParserAgent().parse_revision(
                reason,
                current_goal=state.goal.model_dump(mode="json"),
            )
            parse_source = "llm"
        except Exception as exc:
            # Safe degradation: preserve the feedback, but never guess a
            # constraint mutation from keywords when the intent model is down.
            logger.warning("LLM revision parsing failed; preserving feedback only: %s", exc)
            interpretation = RevisionParseOutput(
                operations=[],
                affected_domains=["schedule"],
                needs_clarification=True,
                clarification_question="我暂时没能可靠理解这条修改，请再具体说明要改哪一部分。",
            )
            parse_source = "safe_fallback"
    elif not isinstance(interpretation, RevisionParseOutput):
        interpretation = RevisionParseOutput.model_validate(interpretation)

    hard = dict(state.goal.hard_constraints)
    soft = dict(state.goal.soft_preferences)
    feedback = list(soft.get("revision_feedback") or [])
    feedback.append(reason)
    soft["revision_feedback"] = feedback[-5:]
    changed_constraints: dict[str, Any] = {"revision_reason": reason}
    rejected_operations: list[dict[str, Any]] = []
    should_apply = (
        interpretation.intent in {"revise_itinerary", "start_new_trip"}
        and interpretation.confidence >= 0.55
        and not interpretation.needs_clarification
    )
    operations = interpretation.operations if should_apply else []
    for operation in operations:
        try:
            _apply_revision_operation(
                hard,
                soft,
                field=operation.field,
                operation=operation.operation,
                value=operation.value,
            )
            changed_constraints[operation.field] = {
                "operation": operation.operation,
                "value": operation.value,
            }
        except (TypeError, ValueError, StateTransitionError) as exc:
            rejected_operations.append({"field": operation.field, "reason": str(exc)})
    soft["revision_parse"] = {
        "source": parse_source,
        "confidence": interpretation.confidence,
        "affected_domains": interpretation.affected_domains,
        "rejected_operations": rejected_operations,
    }

    missing_information: list[str] = []
    capability_status = "solvable"
    if interpretation.needs_clarification or interpretation.confidence < 0.55:
        missing_information = ["revision_clarification"]
        capability_status = "needs_user"
        soft["revision_clarification_question"] = (
            interpretation.clarification_question or "你希望具体修改行程的哪一部分？"
        )

    new_goal = state.goal.model_copy(
        update={
            "goal_version": state.goal.goal_version + 1,
            "original_request": (
                f"{state.goal.original_request}\n用户对上一版的修改意见：{reason}"
            ),
            "hard_constraints": hard,
            "soft_preferences": soft,
            "missing_information": missing_information,
            "capability": state.goal.capability.model_copy(update={"status": capability_status}),
        }
    )
    new_plan_version = state.task_graph.plan_version + 1
    if any(task.task_id == "research_evidence" for task in state.task_graph.tasks):
        from agentic.react import ReactTaskGraphPlanner

        planner = ReactTaskGraphPlanner()
    else:
        planner = _configured_task_graph_planner()
    graph = planner.plan(new_goal, plan_version=new_plan_version)
    graph = TaskGraphController().refresh_ready(graph)
    ready = TaskGraphController.ready_tasks(graph)
    history = list(state.plan_versions)
    history.append(
        PlanVersion(
            plan_version=new_plan_version,
            goal_version=new_goal.goal_version,
            trigger="user_rejection_feedback",
            changed_constraints=changed_constraints,
            invalidated_task_ids=[task.task_id for task in state.task_graph.tasks],
            preserved_task_ids=[],
        )
    )
    revised = AgentLedgerState(
        trajectory_id=state.trajectory_id,
        goal=new_goal,
        task_graph=graph,
        facts=dict(state.facts),
        artifacts=dict(state.artifacts),
        failures=list(state.failures),
        decision_history=list(state.decision_history),
        plan_versions=history,
        budget=BudgetLedger(max_tool_calls=settings.agentic_tool_call_budget),
        current_task_id=ready[0].task_id if ready else None,
    )
    return revised


def confirm_agent_ledger(
    ledger: AgentLedgerState | dict[str, Any],
) -> tuple[AgentLedgerState, CompletionDecision]:
    """Close the confirmation task and enforce the global completion gate."""
    state = ledger if isinstance(ledger, AgentLedgerState) else AgentLedgerState(**ledger)
    state = resume_agent_ledger(
        state,
        task_id="await_confirmation",
        user_value=True,
        fact_key="user_confirmation",
    )
    reports = [
        artifact
        for artifact in state.artifacts.values()
        if artifact.artifact_type == "validation_report"
        and artifact.goal_version == state.goal.goal_version
        and artifact.plan_version == state.task_graph.plan_version
    ]
    report = reports[-1].payload if reports else None
    decision = CompletionGuard(mode="enforce").evaluate(report, ledger=state)
    if not decision.allowed:
        codes = ", ".join(block.code for block in decision.blocks)
        raise StateTransitionError(f"global completion guard rejected confirmation: {codes}")
    state.termination_reason = "validated_finish"
    return state, decision
