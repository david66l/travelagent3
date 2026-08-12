"""LangGraph-facing integration for the bounded Agent Loop."""

from __future__ import annotations

import logging
from typing import Any

from agentic.action_executor import TravelActionExecutor
from agentic.loop import ActionExecutor, AgentPolicy, BoundedAgentLoop
from agentic.policy import ApiAgentPolicy
from agentic.state import AgentLedgerState
from agentic.trajectory import EpisodeRecorder
from evaluation.validator import VALIDATOR_VERSION
from vrp_solver_service.models import SolverResponse


logger = logging.getLogger(__name__)
AGENT_ENVIRONMENT_VERSION = "travel-agent-env.v1"


async def run_agent_branch(
    state: dict[str, Any],
    *,
    policy: AgentPolicy | None = None,
    executor: ActionExecutor | None = None,
) -> dict[str, Any]:
    """Run one bounded episode and return a legacy-compatible graph patch."""
    raw_ledger = state.get("agent_ledger")
    if not raw_ledger:
        return _fallback("AGENT_LEDGER_MISSING", "agent ledger was not initialized")

    ledger = AgentLedgerState(**raw_ledger)
    recorder = EpisodeRecorder(
        ledger,
        environment_version=AGENT_ENVIRONMENT_VERSION,
        validator_version=VALIDATOR_VERSION,
        policy_name="api-agent-policy",
        policy_version="v1",
    )
    try:
        result = await BoundedAgentLoop().run(
            ledger,
            policy=policy or ApiAgentPolicy(),
            executor=executor or TravelActionExecutor(),
            recorder=recorder,
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
        "stage": "agent_loop_done",
    }
    validation = _latest_artifact(result.ledger, "validation_report")
    if validation is not None:
        patch["validation_report"] = validation.payload

    solver = _latest_artifact(result.ledger, "solver_result")
    if solver is not None:
        try:
            from graph.node_impl import _vrp_response_to_itinerary

            response = SolverResponse(**solver.payload)
            patch["itinerary"] = _vrp_response_to_itinerary(response)
            patch["solve_status"] = response.status
            patch["solve_time_ms"] = response.solve_time_ms
        except Exception as exc:
            logger.warning("Agent solver artifact could not be projected: %s", exc)
            return _fallback("SOLVER_PROJECTION_FAILED", str(exc), ledger=result.ledger)

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

    return _fallback(
        result.termination_reason,
        "agent episode did not produce a validated draft",
        ledger=result.ledger,
    )


def _latest_artifact(ledger: AgentLedgerState, artifact_type: str):
    matches = [
        artifact
        for artifact in ledger.artifacts.values()
        if artifact.artifact_type == artifact_type
        and artifact.goal_version == ledger.goal.goal_version
        and artifact.plan_version == ledger.task_graph.plan_version
    ]
    return matches[-1] if matches else None


def _fallback(
    code: str,
    message: str,
    *,
    ledger: AgentLedgerState | None = None,
) -> dict[str, Any]:
    patch: dict[str, Any] = {
        "agent_status": "fallback",
        "termination_reason": code,
        "stage": "agent_fallback",
        "warnings": [f"Agent mode fallback: {code}: {message}"],
    }
    if ledger is not None:
        patch["agent_ledger"] = ledger.model_dump(mode="json")
    return patch
