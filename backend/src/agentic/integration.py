"""LangGraph-facing integration for the bounded Agent Loop."""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

from agentic.action_executor import TravelActionExecutor
from agentic.loop import ActionExecutor, AgentPolicy, BoundedAgentLoop
from agentic.policy import (
    ApiAgentPolicy,
    ControllerFirstPolicy,
    DecisionSpecialistRoutedAgentPolicy,
    NativeToolAgentPolicy,
    RoutedAgentPolicy,
    SelfRepairingAgentPolicy,
    ShadowComparingAgentPolicy,
)
from agentic.state import AgentLedgerState
from agentic.trajectory import AgentEpisode, EpisodeRecorder
from evaluation.validator import VALIDATOR_VERSION
from core.settings import settings
from vrp_solver_service.models import SolverResponse


logger = logging.getLogger(__name__)
AGENT_ENVIRONMENT_VERSION = "travel-agent-env.v1"


async def run_agent_branch(
    state: dict[str, Any],
    *,
    policy: AgentPolicy | None = None,
    executor: ActionExecutor | None = None,
    execution_mode: Literal["controller_first", "policy_driven", "react"] | None = None,
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
    if selected_execution_mode in {"controller_first", "react"}:
        # ReAct governs the open-ended research/recovery choices.  Solver,
        # validation, composition and confirmation are single-action system
        # gates, so invoking an LLM there adds cost without adding agency.
        runtime_policy = ControllerFirstPolicy(runtime_policy)
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
        return _fallback(type(exc).__name__, str(exc), ledger=ledger)

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
    routing_summary = summarize_policy_routing(recorder.episode)
    if routing_summary is not None:
        patch["agent_policy_routing"] = routing_summary
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
                routing_summary=routing_summary,
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
        routing_summary=routing_summary,
    )
    if patch.get("agent_error"):
        fallback["agent_error"] = patch["agent_error"]
    return fallback


def _next_task_id(ledger: AgentLedgerState) -> str | None:
    from agentic.state import TaskGraphController

    graph = TaskGraphController().refresh_ready(ledger.task_graph)
    ready = TaskGraphController.ready_tasks(graph)
    return ready[0].task_id if ready else None


def summarize_policy_routing(episode: AgentEpisode) -> dict[str, Any] | None:
    """Build a UI-safe route summary from persisted per-action evidence."""
    decisions: list[dict[str, Any]] = []
    counts = {"student": 0, "teacher": 0}
    family_counts: dict[str, int] = {}
    fallback_count = 0
    completion_tokens = 0
    request_latency_ms = 0.0
    for step in episode.steps:
        trace = step.action.route_trace
        if trace is None:
            continue
        counts[trace.executed_target] += 1
        family_counts[trace.family] = family_counts.get(trace.family, 0) + 1
        fallback_count += int(trace.fallback_used)
        metrics = step.action.inference_metrics
        if metrics is not None:
            completion_tokens += metrics.completion_tokens
            request_latency_ms += metrics.request_latency_ms
        decisions.append(
            {
                "step_index": step.step_index,
                "task_id": step.task_id,
                "action": step.action.action,
                **trace.model_dump(mode="json"),
                "model": metrics.model if metrics is not None else None,
                "completion_tokens": metrics.completion_tokens if metrics is not None else 0,
                "request_latency_ms": metrics.request_latency_ms if metrics is not None else 0.0,
            }
        )
    if not decisions:
        return None
    return {
        "schema_version": "agent-policy-routing-summary.v1",
        "decisions": decisions,
        "route_counts": counts,
        "family_counts": dict(sorted(family_counts.items())),
        "fallback_count": fallback_count,
        "completion_tokens": completion_tokens,
        "request_latency_ms": round(request_latency_ms, 3),
    }


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
        if settings.agentic_decision_specialist_enabled:
            if settings.agentic_policy_routing_enabled:
                raise RuntimeError(
                    "decision-specialist and student/teacher routing cannot be enabled together"
                )
            generalist_model = settings.agentic_policy_model.strip()
            specialist_model = settings.agentic_decision_specialist_model.strip()
            if not generalist_model or not specialist_model:
                raise RuntimeError(
                    "AGENTIC_POLICY_MODEL and AGENTIC_DECISION_SPECIALIST_MODEL are required "
                    "when AGENTIC_DECISION_SPECIALIST_ENABLED=true"
                )
            from core.llm_client import LLMClient

            shared_client = LLMClient(
                base_url=settings.vllm_base_url,
                api_key=settings.vllm_api_key,
                using_vllm=True,
            )
            return DecisionSpecialistRoutedAgentPolicy(
                NativeToolAgentPolicy(
                    shared_client,
                    model=generalist_model,
                    temperature=0.0,
                    max_tokens=settings.agentic_student_max_tokens,
                ),
                NativeToolAgentPolicy(
                    shared_client,
                    model=specialist_model,
                    temperature=0.0,
                    max_tokens=settings.agentic_student_max_tokens,
                ),
            )
        if settings.agentic_policy_routing_enabled:
            student_model = settings.agentic_student_policy_model.strip()
            teacher_model = settings.agentic_teacher_policy_model.strip()
            if not student_model or not teacher_model:
                raise RuntimeError(
                    "AGENTIC_STUDENT_POLICY_MODEL and AGENTIC_TEACHER_POLICY_MODEL "
                    "are required when AGENTIC_POLICY_ROUTING_ENABLED=true"
                )
            from core.llm_client import LLMClient

            student_url = settings.agentic_student_base_url.strip()
            teacher_url = settings.agentic_teacher_base_url.strip()
            shared_client = LLMClient() if not student_url and not teacher_url else None
            student_client = shared_client or LLMClient(
                base_url=student_url or settings.vllm_base_url,
                api_key=settings.vllm_api_key,
                using_vllm=True,
            )
            teacher_client = shared_client or LLMClient(
                base_url=teacher_url or settings.vllm_base_url,
                api_key=settings.vllm_api_key,
                using_vllm=True,
            )
            champion = RoutedAgentPolicy(
                NativeToolAgentPolicy(
                    student_client,
                    model=student_model,
                    temperature=0.0,
                    max_tokens=settings.agentic_student_max_tokens,
                ),
                NativeToolAgentPolicy(
                    teacher_client,
                    model=teacher_model,
                    temperature=0.0,
                    max_tokens=settings.agentic_teacher_max_tokens,
                ),
            )
            if not settings.agentic_challenger_shadow_enabled:
                return champion
            challenger_model = settings.agentic_challenger_policy_model.strip()
            if not challenger_model:
                raise RuntimeError(
                    "AGENTIC_CHALLENGER_POLICY_MODEL is required when "
                    "AGENTIC_CHALLENGER_SHADOW_ENABLED=true"
                )
            challenger_url = (
                settings.agentic_challenger_base_url.strip()
                or student_url
                or settings.vllm_base_url
            )
            challenger_client = LLMClient(
                base_url=challenger_url,
                api_key=settings.vllm_api_key,
                using_vllm=True,
            )
            challenger_teacher_client = LLMClient(
                base_url=teacher_url or settings.vllm_base_url,
                api_key=settings.vllm_api_key,
                using_vllm=True,
            )
            challenger = RoutedAgentPolicy(
                NativeToolAgentPolicy(
                    challenger_client,
                    model=challenger_model,
                    temperature=0.0,
                    max_tokens=settings.agentic_challenger_max_tokens,
                ),
                NativeToolAgentPolicy(
                    challenger_teacher_client,
                    model=teacher_model,
                    temperature=0.0,
                    max_tokens=settings.agentic_teacher_max_tokens,
                ),
            )
            return ShadowComparingAgentPolicy(
                champion,
                challenger,
                challenger_model=challenger_model,
            )
        return NativeToolAgentPolicy()
    return ApiAgentPolicy()


def _policy_identity(policy: AgentPolicy) -> tuple[str, str]:
    checkpoint = getattr(policy, "checkpoint", None)
    if checkpoint:
        return "local-checkpoint-agent-policy", str(checkpoint)
    if isinstance(policy, NativeToolAgentPolicy):
        return "api-native-tool-agent-policy", str(policy.model or "configured")
    if isinstance(policy, RoutedAgentPolicy):
        student = getattr(policy.student, "model", None) or "configured"
        teacher = getattr(policy.teacher, "model", None) or "configured"
        return "routed-native-tool-agent-policy", f"student={student};teacher={teacher}"
    if isinstance(policy, DecisionSpecialistRoutedAgentPolicy):
        generalist = getattr(policy.generalist, "model", None) or "configured"
        specialist = getattr(policy.poi_detail_specialist, "model", None) or "configured"
        return (
            "decision-specialist-native-tool-agent-policy",
            f"generalist={generalist};poi_detail_specialist={specialist}",
        )
    if isinstance(policy, ShadowComparingAgentPolicy):
        champion_name, champion_version = _policy_identity(policy.champion)
        return (
            f"shadow-comparing-{champion_name}",
            f"champion={champion_version};challenger={policy.challenger_model}",
        )
    if isinstance(policy, ApiAgentPolicy):
        model = (
            getattr(policy, "model", None)
            or getattr(getattr(policy, "client", None), "model", None)
            or settings.agentic_policy_model
            or "configured"
        )
        return "api-json-agent-policy", str(model)
    return type(policy).__name__, "injected"


def _fallback(
    code: str,
    message: str,
    *,
    ledger: AgentLedgerState | None = None,
    episode: dict[str, Any] | None = None,
    routing_summary: dict[str, Any] | None = None,
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
    if routing_summary is not None:
        patch["agent_policy_routing"] = routing_summary
    return patch
