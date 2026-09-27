"""LangGraph-facing integration for the bounded Agent Loop."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

from agentic.action_executor import TravelActionExecutor
from agentic.loop import (
    ActionExecutor,
    AgentLoopEvent,
    AgentLoopResult,
    AgentPolicy,
    BoundedAgentLoop,
)
from agentic.policy import (
    ApiAgentPolicy,
    NativeToolAgentPolicy,
    SelfRepairingAgentPolicy,
)
from agentic.state import AgentLedgerState
from agentic.trajectory import AgentEpisode, EpisodeRecorder
from evaluation.validator import VALIDATOR_VERSION
from core.settings import settings
from vrp_solver_service.models import SolverResponse


logger = logging.getLogger(__name__)
AGENT_ENVIRONMENT_VERSION = "agent-harness-v1"


async def run_agent_branch(
    state: dict[str, Any],
    *,
    policy: AgentPolicy | None = None,
    executor: ActionExecutor | None = None,
    execution_mode: Literal["react"] | None = None,
    single_step: bool = False,
) -> dict[str, Any]:
    """Run an Agent episode or one checkpointable production action batch."""
    raw_ledger = state.get("agent_ledger")
    if not raw_ledger:
        return _fallback("AGENT_LEDGER_MISSING", "agent ledger was not initialized")

    ledger = AgentLedgerState(**raw_ledger)
    selected_policy = policy or _configured_policy()
    selected_execution_mode = execution_mode or settings.agentic_execution_mode
    runtime_policy: AgentPolicy = SelfRepairingAgentPolicy(
        selected_policy,
        max_repair_attempts=settings.agentic_policy_repair_attempts,
    )
    if selected_execution_mode != "react":
        raise ValueError("Only the unified agent harness is supported")
    policy_name, policy_version = _policy_identity(selected_policy)
    existing_episode = state.get("agent_episode")
    recorder = (
        EpisodeRecorder.resume(existing_episode)
        if existing_episode
        else EpisodeRecorder(
            ledger,
            environment_version=AGENT_ENVIRONMENT_VERSION,
            validator_version=VALIDATOR_VERSION,
            policy_name=policy_name,
            policy_version=policy_version,
        )
    )
    try:
        result = await BoundedAgentLoop().run(
            ledger,
            policy=runtime_policy,
            executor=executor or TravelActionExecutor(),
            recorder=recorder,
            max_batches=1 if single_step else None,
        )
    except Exception as exc:
        logger.exception("Agent branch failed before a terminal result: %s", exc)
        message = f"{type(exc).__name__}: {exc}"
        failed = AgentLoopResult(
            ledger=ledger,
            status="failed",
            termination_reason="runtime_error_fallback",
            events=[
                AgentLoopEvent(
                    sequence=1,
                    event_type="episode_terminated",
                    payload={
                        "status": "failed",
                        "reason": "runtime_error_fallback",
                        "error": message,
                        "failure_class": "runtime_error",
                        "error_type": type(exc).__name__,
                    },
                )
            ],
        )
        recorder.finalize(failed)
        return _fallback(
            "runtime_error_fallback",
            message,
            ledger=ledger,
            episode=recorder.episode.model_dump(mode="json"),
            agent_error=message,
        )

    patch: dict[str, Any] = {
        "agent_ledger": result.ledger.model_dump(mode="json"),
        "agent_episode": recorder.episode.model_dump(mode="json"),
        "agent_status": result.status,
        "termination_reason": result.termination_reason,
        "agent_step": result.ledger.budget.used_episode_steps,
        "agent_execution_mode": selected_execution_mode,
        "stage": "agent_loop_done",
    }
    if result.status == "failed":
        termination_error = next(
            (
                event.payload.get("error")
                for event in reversed(recorder.episode.events)
                if event.event_type == "episode_terminated" and event.payload.get("error")
            ),
            None,
        )
        if termination_error:
            patch["agent_error"] = str(termination_error)
    validation = _latest_artifact(result.ledger, "validation_report")
    if validation is not None:
        patch["validation_report"] = validation.payload

    solver = _latest_artifact(result.ledger, "solver_result")
    if solver is not None:
        try:
            from graph.node_impl import _vrp_response_to_itinerary
            from vrp_solver_service.models import POIInput

            response = SolverResponse(**solver.payload)
            candidates = _latest_artifact(result.ledger, "poi_candidate_set")
            restaurant_pois = []
            for index, item in enumerate(
                (candidates.payload.get("pois") if candidates else []) or []
            ):
                if not isinstance(item, dict) or not item.get("name"):
                    continue
                if str(item.get("category") or "").lower() != "restaurant":
                    continue
                restaurant_pois.append(POIInput(**TravelActionExecutor._poi_input(item, index)))
            hard = result.ledger.goal.hard_constraints
            start_date = hard.get("start_date")
            end_date = hard.get("end_date") or start_date
            travel_dates = f"{start_date}|{end_date}" if start_date else None
            days = max(1, int(hard.get("travel_days") or len(response.days) or 1))
            total_budget = float(hard.get("budget_range") or 0)
            meal_budget = max(80.0, total_budget * 0.35 / (days * 2) * 2) if total_budget else 0.0
            patch["itinerary"] = _vrp_response_to_itinerary(
                response,
                restaurant_pois=restaurant_pois,
                meal_budget=meal_budget,
                travel_dates=travel_dates,
            )
            patch["solve_status"] = response.status
            patch["solve_time_ms"] = response.solve_time_ms
        except Exception as exc:
            logger.warning("Agent solver artifact could not be projected: %s", exc)
            return _fallback(
                "SOLVER_PROJECTION_FAILED",
                str(exc),
                ledger=result.ledger,
                episode=recorder.episode.model_dump(mode="json"),
            )

    if result.status == "running":
        patch.update(
            {
                "stage": "agent_loop_step",
                "next_action": "agent_continue",
                "current_task_id": _next_task_id(result.ledger),
            }
        )
        return patch

    if result.status == "interrupted" and result.termination_reason == "awaiting_user":
        if patch.get("itinerary") and validation and validation.payload.get("hard_pass"):
            patch.update(
                {
                    "agent_status": "awaiting_confirmation",
                    "stage": "agent_draft_ready",
                    "next_action": "agent_draft",
                }
            )
            return patch
        patch.update(
            {
                "agent_status": "awaiting_information",
                "stage": "agent_awaiting_information",
                "next_action": "clarify",
            }
        )
        return patch

    if result.status == "finished" and patch.get("itinerary"):
        patch.update({"stage": "agent_validated", "next_action": "agent_draft"})
        return patch

    fallback = _fallback(
        result.termination_reason,
        "agent episode did not produce a validated draft",
        ledger=result.ledger,
        episode=recorder.episode.model_dump(mode="json"),
    )
    if patch.get("agent_error"):
        fallback["agent_error"] = patch["agent_error"]
    return fallback


def _next_task_id(ledger: AgentLedgerState) -> str | None:
    from agentic.state import TaskGraphController

    graph = TaskGraphController().refresh_ready(ledger.task_graph)
    ready = TaskGraphController.ready_tasks(graph)
    return ready[0].task_id if ready else None


def _latest_artifact(ledger: AgentLedgerState, artifact_type: str):
    matches = [
        artifact
        for artifact in ledger.artifacts.values()
        if artifact.artifact_type == artifact_type
        and artifact.goal_version == ledger.goal.goal_version
        and artifact.plan_version == ledger.task_graph.plan_version
    ]
    return matches[-1] if matches else None


@lru_cache(maxsize=1)
def _configured_local_policy() -> AgentPolicy:
    checkpoint = settings.agentic_local_checkpoint.strip()
    if not checkpoint:
        raise RuntimeError(
            "AGENTIC_LOCAL_CHECKPOINT is required when AGENTIC_POLICY_BACKEND=local_checkpoint"
        )
    from agentic.local_policy import LocalCheckpointAgentPolicy

    policy_options: dict[str, Any] = {
        "max_new_tokens": settings.agentic_local_max_new_tokens,
        "do_sample": False,
        "load_in_4bit": settings.agentic_local_load_in_4bit,
        "structured_decoding": settings.agentic_local_structured_decoding,
    }
    revision = settings.agentic_local_revision.strip()
    if revision:
        policy_options["revision"] = revision
    return LocalCheckpointAgentPolicy(checkpoint, **policy_options)


def _configured_policy() -> AgentPolicy:
    if settings.agentic_policy_backend == "local_checkpoint":
        return _configured_local_policy()
    if settings.agentic_policy_protocol == "native_tool":
        return NativeToolAgentPolicy()
    return ApiAgentPolicy()


def _policy_identity(policy: AgentPolicy) -> tuple[str, str]:
    checkpoint = getattr(policy, "checkpoint", None)
    if checkpoint:
        return "local-checkpoint-agent-policy", str(checkpoint)
    if isinstance(policy, ApiAgentPolicy):
        model = (
            getattr(policy, "model", None)
            or getattr(getattr(policy, "client", None), "model", None)
            or settings.agentic_policy_model
            or "configured"
        )
        return "api-json-agent-policy", str(model)
    return type(policy).__name__, str(getattr(policy, "model", None) or "configured")


def _fallback(
    code: str,
    message: str,
    *,
    ledger: AgentLedgerState | None = None,
    episode: dict[str, Any] | None = None,
    agent_error: str | None = None,
) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "agent_status": "failed",
        "termination_reason": code,
        "stage": "agent_failed",
        "next_action": "agent_error",
        "warnings": [f"Agent mode stopped safely: {code}: {message}"],
    }
    if ledger is not None:
        patch["agent_ledger"] = ledger.model_dump(mode="json")
    if episode is not None:
        patch["agent_episode"] = episode
    if agent_error is not None:
        patch["agent_error"] = agent_error
    return patch
