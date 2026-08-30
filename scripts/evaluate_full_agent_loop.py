"""Run raw user requests through intent, ReAct tools, CP-SAT and Verifier."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from agentic.integration import run_agent_branch  # noqa: E402
from agentic.loop import ActionExecutor  # noqa: E402
from agentic.policy import NativeToolAgentPolicy  # noqa: E402
from agentic.runtime import initialize_agent_ledger, revise_agent_ledger  # noqa: E402
from agentic.sft_dataset import EpisodeCandidate  # noqa: E402
from agentic.trajectory import AgentEpisode  # noqa: E402
from core.city_names import canonical_city_name  # noqa: E402
from core.conversation_state import default_conversation_state  # noqa: E402
from core.conversation_turn import process_user_turn  # noqa: E402
from core.inference_metrics import percentile  # noqa: E402
from core.llm_client import LLMClient, llm  # noqa: E402
from core.redis_client import redis_client  # noqa: E402
from core.settings import settings  # noqa: E402
from evaluation.full_agent_loop_benchmark import (  # noqa: E402
    SCHEMA_VERSION,
    FullAgentLoopCase,
    benchmark_hash,
    build_frozen_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--episode-output",
        type=Path,
        help=(
            "Optional EpisodeCandidate JSONL sidecar containing complete, replayable "
            "AgentEpisode records for later SFT auditing."
        ),
    )
    parser.add_argument(
        "--rollout-id",
        default="default",
        help="Stable rollout label used to keep repeated case trajectories distinct.",
    )
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument(
        "--suite",
        choices=("core", "expanded"),
        help="Run one frozen case suite without repeating the other suite.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--policy-model",
        help="Model name used only for Agent policy tool decisions.",
    )
    parser.add_argument(
        "--policy-base-url",
        help="Optional OpenAI-compatible policy endpoint, for example a remote vLLM /v1 URL.",
    )
    parser.add_argument(
        "--policy-api-key",
        default="not-needed",
        help="API key for --policy-base-url. The default is suitable for local vLLM.",
    )
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Agent policy decisions.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="With --resume, replace only previously failed records.",
    )
    return parser.parse_args()


def _episode_candidate(
    case: FullAgentLoopCase,
    episode: dict[str, Any],
    *,
    rollout_id: str,
    phase: str,
) -> EpisodeCandidate:
    """Wrap one complete ReAct episode in the common audited-corpus contract."""
    parsed = AgentEpisode(**episode)
    initial_goal = parsed.initial_state.get("goal") or {}
    hard_constraints = initial_goal.get("hard_constraints") or {}
    destination = hard_constraints.get("destination") or case.expected_slots.get("destination")
    city = canonical_city_name(str(destination or "")) or "unspecified"
    return EpisodeCandidate(
        scenario_id=f"{case.case_id}:{rollout_id}:{phase}",
        source="teacher",
        template_family=f"native-react:{case.slice}",
        city=city,
        episode=parsed,
    )


def _collect_episode(
    collector: list[EpisodeCandidate] | None,
    case: FullAgentLoopCase,
    result: dict[str, Any],
    *,
    rollout_id: str,
    phase: str,
) -> None:
    if collector is None:
        return
    episode = result.get("agent_episode")
    if not episode:
        return
    collector.append(
        _episode_candidate(
            case,
            episode,
            rollout_id=rollout_id,
            phase=phase,
        )
    )


def _write_episode_candidates(path: Path, candidates: list[EpisodeCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        candidate.model_dump_json()
        for candidate in sorted(candidates, key=lambda item: item.scenario_id)
    )
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def _current_artifacts(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    goal_version = int((ledger.get("goal") or {}).get("goal_version") or 1)
    plan_version = int((ledger.get("task_graph") or {}).get("plan_version") or 1)
    return [
        artifact
        for artifact in (ledger.get("artifacts") or {}).values()
        if int(artifact.get("goal_version") or 0) == goal_version
        and int(artifact.get("plan_version") or 0) == plan_version
    ]


def _action_rows(episode: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "step": int(step.get("step_index") or 0),
            "task": step.get("task_id"),
            "action": (step.get("action") or {}).get("action"),
            "arguments": (step.get("action") or {}).get("arguments") or {},
            "source": (step.get("action") or {}).get("decision_source"),
            "tokens": int((step.get("action") or {}).get("token_usage") or 0),
            "policy_latency_ms": int(step.get("policy_latency_ms") or 0),
            "action_latency_ms": int(step.get("action_latency_ms") or 0),
            "repair_attempts": int((step.get("action") or {}).get("repair_attempts") or 0),
            "route_trace": (step.get("action") or {}).get("route_trace"),
            "shadow_trace": (step.get("action") or {}).get("shadow_trace"),
            "verification": step.get("verification") or {},
            "observation_errors": [
                observation.get("error")
                for observation in (step.get("observations") or [])
                if observation.get("error")
            ],
            "context_failures": (step.get("context") or {}).get("failure_summary") or [],
            "context_artifacts": (step.get("context") or {}).get("relevant_artifacts") or [],
        }
        for step in episode.get("steps") or []
    ]


def _agent_runtime_error(result: dict[str, Any]) -> str | None:
    if result.get("agent_error") or result.get("error"):
        return str(result.get("agent_error") or result.get("error"))
    events = list((result.get("agent_episode") or {}).get("events") or [])
    return next(
        (
            str(event.get("payload", {}).get("error"))
            for event in reversed(events)
            if event.get("event_type") == "episode_terminated"
            and event.get("payload", {}).get("error")
        ),
        None,
    )


def _normalized_text(value: Any) -> str:
    return "".join(str(value).strip().lower().split())


def _value_matches(actual: Any, expected: Any) -> bool:
    """Match frozen semantic expectations without overfitting model wording."""
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        return all(
            any(_value_matches(actual_item, expected_item) for actual_item in actual)
            for expected_item in expected
        )
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return abs(float(actual) - float(expected)) < 1e-6
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str):
        actual_text = _normalized_text(actual)
        expected_text = _normalized_text(expected)
        return bool(actual_text and expected_text) and (
            actual_text == expected_text
            or actual_text in expected_text
            or expected_text in actual_text
        )
    return actual == expected


def _score_expected_mapping(
    actual: dict[str, Any],
    expected: dict[str, object],
    *,
    prefix: str,
) -> list[str]:
    failures: list[str] = []
    for field, expected_value in expected.items():
        if field not in actual:
            failures.append(f"{prefix}_MISSING:{field}")
        elif not _value_matches(actual[field], expected_value):
            failures.append(f"{prefix}_MISMATCH:{field}")
    return failures


def _score_intent(case: FullAgentLoopCase, intent: Any) -> list[str]:
    return _score_expected_mapping(
        dict(intent.slots or {}),
        case.expected_slots,
        prefix="INTENT_SLOT",
    )


def _score_draft(
    case: FullAgentLoopCase,
    result: dict[str, Any],
    actions: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
) -> list[str]:
    failures: list[str] = []
    if result.get("agent_status") != "awaiting_confirmation":
        failures.append(f"STATUS:{result.get('agent_status')}")
    if not (result.get("validation_report") or {}).get("hard_pass"):
        failures.append("VERIFIER_HARD_FAIL")
    if not result.get("itinerary"):
        failures.append("ITINERARY_MISSING")
    if int((result.get("agent_ledger") or {}).get("budget", {}).get("used_solver_calls") or 0) != 1:
        failures.append("SOLVER_CALL_COUNT")
    action_names = [str(item["action"]) for item in actions]
    for action in case.required_actions:
        if action not in action_names:
            failures.append(f"MISSING_ACTION:{action}")
    artifact_types = {str(item.get("artifact_type")) for item in artifacts}
    for artifact in case.required_artifacts:
        if artifact not in artifact_types:
            failures.append(f"MISSING_ARTIFACT:{artifact}")
    if sum(item["source"] == "policy" for item in actions) < 2:
        failures.append("NOT_A_MULTI_TURN_POLICY_LOOP")
    exact = Counter(
        (
            item["action"],
            json.dumps(item["arguments"], ensure_ascii=False, sort_keys=True),
        )
        for item in actions
        if item["source"] == "policy"
    )
    if any(count > 1 for count in exact.values()):
        failures.append("EXACT_POLICY_ACTION_REPEAT")
    return failures


def _score_safe_termination(
    case: FullAgentLoopCase,
    result: dict[str, Any],
    actions: list[dict[str, Any]],
) -> list[str]:
    """Accept a grounded tradeoff when a live provider cannot satisfy hard gates."""
    failures: list[str] = []
    if result.get("agent_status") != "awaiting_information":
        failures.append(f"STATUS:{result.get('agent_status')}")
    if result.get("itinerary"):
        failures.append("UNVERIFIED_ITINERARY_EMITTED")
    ledger = result.get("agent_ledger") or {}
    verifier_grounded = any(
        artifact.get("artifact_type") == "validation_report"
        and artifact.get("hard_pass") is False
        and bool(artifact.get("violation_codes"))
        for action in actions
        for artifact in action.get("context_artifacts") or []
    )
    if (
        int((ledger.get("budget") or {}).get("used_solver_calls") or 0) != 0
        and not verifier_grounded
    ):
        failures.append("UNSAFE_SOLVER_CALL")
    action_names = [str(item["action"]) for item in actions]
    for action in case.safe_required_actions:
        if action not in action_names:
            failures.append(f"MISSING_SAFE_ACTION:{action}")
    recorded = ledger.get("failures") or []
    capability = (ledger.get("goal") or {}).get("capability") or {}
    capability_grounded = capability.get("status") in {"infeasible", "unsafe"} and bool(
        capability.get("evidence")
    )
    if (
        not capability_grounded
        and not verifier_grounded
        and not any(
            item.get("code") in {"RESEARCH_EVIDENCE_INSUFFICIENT", "TOOL_UNAVAILABLE"}
            for item in recorded
        )
    ):
        failures.append("GROUNDED_TRADEOFF_REASON_MISSING")
    minimum_policy_calls = 1 if capability_grounded else 2
    if sum(item["source"] == "policy" for item in actions) < minimum_policy_calls:
        failures.append("NOT_A_MULTI_TURN_POLICY_LOOP")
    return failures


async def _evaluate_revision_result(
    case: FullAgentLoopCase,
    *,
    intent: Any,
    intent_tokens: int,
    initial_result: dict[str, Any],
    initial_actions: list[dict[str, Any]],
    initial_artifacts: list[dict[str, Any]],
    policy: NativeToolAgentPolicy,
    executor: ActionExecutor | None,
    started: float,
    intent_failures: list[str],
    episode_collector: list[EpisodeCandidate] | None,
    rollout_id: str,
) -> dict[str, Any]:
    """Reject the first draft, version the goal, then require a second hard pass."""
    failures = [
        *intent_failures,
        *[
            f"INITIAL:{code}"
            for code in _score_draft(case, initial_result, initial_actions, initial_artifacts)
        ],
    ]
    first_ledger = initial_result.get("agent_ledger") or {}
    first_agent_tokens = int((first_ledger.get("budget") or {}).get("used_tokens") or 0)
    revised = await revise_agent_ledger(
        first_ledger,
        revision_reason=str(case.revision_input or ""),
    )
    revision_tokens = int(llm.last_token_usage or 0)
    revised_result = await run_agent_branch(
        {"agent_ledger": revised.model_dump(mode="json")},
        policy=policy,
        executor=executor,
        execution_mode="react",
    )
    _collect_episode(
        episode_collector,
        case,
        revised_result,
        rollout_id=rollout_id,
        phase="revised",
    )
    second_ledger = revised_result.get("agent_ledger") or {}
    second_episode = revised_result.get("agent_episode") or {}
    second_actions = _action_rows(second_episode)
    second_artifacts = _current_artifacts(second_ledger)
    failures.extend(
        f"REVISED:{code}"
        for code in _score_draft(case, revised_result, second_actions, second_artifacts)
    )
    first_goal_version = int((first_ledger.get("goal") or {}).get("goal_version") or 0)
    second_goal = second_ledger.get("goal") or {}
    second_goal_version = int(second_goal.get("goal_version") or 0)
    first_plan_version = int((first_ledger.get("task_graph") or {}).get("plan_version") or 0)
    second_plan_version = int((second_ledger.get("task_graph") or {}).get("plan_version") or 0)
    if second_goal_version != first_goal_version + 1:
        failures.append("GOAL_VERSION_NOT_INCREMENTED")
    if second_plan_version != first_plan_version + 1:
        failures.append("PLAN_VERSION_NOT_INCREMENTED")
    second_hard = dict(second_goal.get("hard_constraints") or {})
    second_soft = dict(second_goal.get("soft_preferences") or {})
    if case.expected_revision_exclusions:
        actual_exclusions = [
            *list(second_hard.get("must_not_visit") or []),
            *list(second_soft.get("avoid_pois") or []),
        ]
        if not _value_matches(
            actual_exclusions,
            case.expected_revision_exclusions,
        ):
            failures.append("REVISION_EXCLUSIONS_NOT_APPLIED")
    failures.extend(
        _score_expected_mapping(
            second_hard,
            case.expected_revision_hard,
            prefix="REVISION_HARD",
        )
    )
    failures.extend(
        _score_expected_mapping(
            second_soft,
            case.expected_revision_soft,
            prefix="REVISION_SOFT",
        )
    )
    expected_days = case.expected_revision_hard.get("travel_days")
    if expected_days is not None and not _value_matches(
        len(revised_result.get("itinerary") or []), expected_days
    ):
        failures.append("REVISED_ITINERARY_DAY_COUNT")

    second_agent_tokens = int((second_ledger.get("budget") or {}).get("used_tokens") or 0)
    all_actions = [
        *({**item, "phase": "initial"} for item in initial_actions),
        *({**item, "phase": "revised"} for item in second_actions),
    ]
    return {
        "case_id": case.case_id,
        "slice": case.slice,
        "expected_outcome": case.expected_outcome,
        "passed": not failures,
        "failures": failures,
        "intent": intent.model_dump(mode="json"),
        "intent_tokens": intent_tokens,
        "revision_tokens": revision_tokens,
        "agent_tokens": first_agent_tokens + second_agent_tokens,
        "total_tokens": (
            intent_tokens + revision_tokens + first_agent_tokens + second_agent_tokens
        ),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        "agent_status": revised_result.get("agent_status"),
        "termination_reason": revised_result.get("termination_reason"),
        "solver_status": revised_result.get("solve_status"),
        "solver_metadata": next(
            (
                artifact.get("payload", {}).get("metadata", {})
                for artifact in second_artifacts
                if artifact.get("artifact_type") == "solver_result"
            ),
            {},
        ),
        "validation_hard_pass": (revised_result.get("validation_report") or {}).get("hard_pass"),
        "itinerary_days": len(revised_result.get("itinerary") or []),
        "policy_calls": sum(item["source"] == "policy" for item in all_actions),
        "tool_calls": int((first_ledger.get("budget") or {}).get("used_tool_calls") or 0)
        + int((second_ledger.get("budget") or {}).get("used_tool_calls") or 0),
        "episode_steps": int((first_ledger.get("budget") or {}).get("used_episode_steps") or 0)
        + int((second_ledger.get("budget") or {}).get("used_episode_steps") or 0),
        "actions": all_actions,
        "artifact_types": sorted({str(item.get("artifact_type")) for item in second_artifacts}),
        "failures_recorded": [
            *(first_ledger.get("failures") or []),
            *(second_ledger.get("failures") or []),
        ],
        "revision": {
            "feedback": case.revision_input,
            "first_goal_version": first_goal_version,
            "second_goal_version": second_goal_version,
            "first_plan_version": first_plan_version,
            "second_plan_version": second_plan_version,
            "first_itinerary_days": len(initial_result.get("itinerary") or []),
            "second_itinerary_days": len(revised_result.get("itinerary") or []),
        },
        "runtime_errors": {
            "initial": _agent_runtime_error(initial_result),
            "revised": _agent_runtime_error(revised_result),
        },
        "error": None,
    }


async def evaluate_case(
    case: FullAgentLoopCase,
    *,
    policy: NativeToolAgentPolicy,
    executor: ActionExecutor | None = None,
    rollout_id: str | None = None,
    episode_collector: list[EpisodeCandidate] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    state = default_conversation_state()
    identity = f"travelagent-benchmark:{case.case_id}:{rollout_id or 'default'}"
    state["user_id"] = str(uuid5(NAMESPACE_URL, identity))
    try:
        intent = await process_user_turn(state, case.user_input)
        intent_tokens = int(intent.token_usage or 0)
        intent_failures = _score_intent(case, intent)
        state["user_input"] = case.user_input
        state["missing_slots"] = list(intent.missing_required)
        if case.expected_outcome == "clarification":
            missing = set(intent.missing_required)
            failures = [
                *intent_failures,
                *[
                    f"MISSING_CLARIFICATION_FIELD:{field}"
                    for field in case.expected_missing
                    if field not in missing
                ],
            ]
            if not intent.clarification_questions:
                failures.append("CLARIFICATION_QUESTION_MISSING")
            return {
                "case_id": case.case_id,
                "slice": case.slice,
                "expected_outcome": case.expected_outcome,
                "passed": not failures,
                "failures": failures,
                "intent": intent.model_dump(mode="json"),
                "intent_tokens": intent_tokens,
                "agent_tokens": 0,
                "total_tokens": intent_tokens,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "agent_status": "not_started_missing_information",
                "solver_status": None,
                "validation_hard_pass": None,
                "actions": [],
                "artifact_types": [],
                "error": None,
            }
        if intent.missing_required:
            raise RuntimeError("unexpected missing fields: " + ",".join(intent.missing_required))
        initialized = initialize_agent_ledger(
            state,
            mode="agent",
            task_graph_mode="react",
        )
        result = await run_agent_branch(
            {**state, **initialized},
            policy=policy,
            executor=executor,
            execution_mode="react",
        )
        effective_rollout_id = rollout_id or "default"
        ledger = result.get("agent_ledger") or {}
        episode = result.get("agent_episode") or {}
        actions = _action_rows(episode)
        artifacts = _current_artifacts(ledger)
        if case.expected_outcome == "revision":
            return await _evaluate_revision_result(
                case,
                intent=intent,
                intent_tokens=intent_tokens,
                initial_result=result,
                initial_actions=actions,
                initial_artifacts=artifacts,
                policy=policy,
                executor=executor,
                started=started,
                intent_failures=intent_failures,
                episode_collector=episode_collector,
                rollout_id=effective_rollout_id,
            )
        _collect_episode(
            episode_collector,
            case,
            result,
            rollout_id=effective_rollout_id,
            phase="initial",
        )
        if (
            case.expected_outcome == "draft_or_safe_termination"
            and result.get("agent_status") != "awaiting_confirmation"
        ):
            failures = [
                *intent_failures,
                *_score_safe_termination(case, result, actions),
            ]
        else:
            failures = [
                *intent_failures,
                *_score_draft(case, result, actions, artifacts),
            ]
        agent_tokens = int((ledger.get("budget") or {}).get("used_tokens") or 0)
        return {
            "case_id": case.case_id,
            "slice": case.slice,
            "expected_outcome": case.expected_outcome,
            "passed": not failures,
            "failures": failures,
            "intent": intent.model_dump(mode="json"),
            "intent_tokens": intent_tokens,
            "agent_tokens": agent_tokens,
            "total_tokens": intent_tokens + agent_tokens,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "agent_status": result.get("agent_status"),
            "termination_reason": result.get("termination_reason"),
            "runtime_error": _agent_runtime_error(result),
            "solver_status": result.get("solve_status"),
            "solver_metadata": next(
                (
                    artifact.get("payload", {}).get("metadata", {})
                    for artifact in artifacts
                    if artifact.get("artifact_type") == "solver_result"
                ),
                {},
            ),
            "validation_hard_pass": (result.get("validation_report") or {}).get("hard_pass"),
            "itinerary_days": len(result.get("itinerary") or []),
            "policy_calls": sum(item["source"] == "policy" for item in actions),
            "tool_calls": int((ledger.get("budget") or {}).get("used_tool_calls") or 0),
            "episode_steps": int((ledger.get("budget") or {}).get("used_episode_steps") or 0),
            "actions": actions,
            "artifact_types": sorted({str(item.get("artifact_type")) for item in artifacts}),
            "failures_recorded": ledger.get("failures") or [],
            "error": None,
        }
    except Exception as exc:
        return {
            "case_id": case.case_id,
            "slice": case.slice,
            "expected_outcome": case.expected_outcome,
            "passed": False,
            "failures": [f"EXCEPTION:{type(exc).__name__}"],
            "intent_tokens": int(llm.last_token_usage or 0),
            "agent_tokens": 0,
            "total_tokens": int(llm.last_token_usage or 0),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "agent_status": "exception",
            "solver_status": None,
            "validation_hard_pass": None,
            "actions": [],
            "artifact_types": [],
            "error": str(exc),
        }


def build_report(
    cases: list[FullAgentLoopCase],
    records: list[dict[str, Any]],
    *,
    policy_model: str | None = None,
    policy_backend: str = "configured_default",
) -> dict[str, Any]:
    latencies = [float(item["latency_ms"]) for item in records]
    draft_rows = [item for item in records if item["expected_outcome"] == "draft"]
    planned_rows = [item for item in records if item["expected_outcome"] in {"draft", "revision"}]
    passed = sum(bool(item["passed"]) for item in records)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_hash": benchmark_hash(),
        "selected_case_ids": [case.case_id for case in cases],
        "model": policy_model or settings.llm_model,
        "policy_model": policy_model or settings.llm_model,
        "policy_backend": policy_backend,
        "intent_model": settings.llm_model,
        "execution_mode": "react",
        "policy_protocol": "native_tool",
        "summary": {
            "total": len(records),
            "passed": passed,
            "failed": len(records) - passed,
            "pass_rate": round(passed / len(records), 4) if records else 0.0,
            "draft_hard_pass_rate": round(
                sum(bool(item.get("validation_hard_pass")) for item in draft_rows)
                / len(draft_rows),
                4,
            )
            if draft_rows
            else 0.0,
            "planned_count": len(planned_rows),
            "planned_hard_pass_rate": round(
                sum(bool(item.get("validation_hard_pass")) for item in planned_rows)
                / len(planned_rows),
                4,
            )
            if planned_rows
            else 0.0,
            "cpsat_success_rate": round(
                sum(item.get("solver_status") == "optimal" for item in draft_rows)
                / len(draft_rows),
                4,
            )
            if draft_rows
            else 0.0,
            "total_tokens": sum(int(item["total_tokens"]) for item in records),
            "mean_tokens": round(statistics.fmean(int(item["total_tokens"]) for item in records), 3)
            if records
            else 0.0,
            "mean_latency_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p95_latency_ms": round(percentile(latencies, 0.95), 3),
            "mean_policy_calls": round(
                statistics.fmean(int(item.get("policy_calls") or 0) for item in draft_rows),
                3,
            )
            if draft_rows
            else 0.0,
            "mean_tool_calls": round(
                statistics.fmean(int(item.get("tool_calls") or 0) for item in draft_rows),
                3,
            )
            if draft_rows
            else 0.0,
            "solver_status_counts": dict(
                Counter(str(item.get("solver_status")) for item in draft_rows)
            ),
            "solver_status_counts_scope": "expected_outcome=draft",
            "planned_solver_status_counts": dict(
                Counter(str(item.get("solver_status")) for item in planned_rows)
            ),
            "actual_outcome_counts": dict(
                Counter(str(item.get("agent_status")) for item in records)
            ),
            "failure_counts": dict(
                Counter(code for item in records for code in item.get("failures") or [])
            ),
        },
        "records": records,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = build_frozen_cases()
    if args.suite:
        cases = [case for case in cases if case.suite == args.suite]
    if args.case_ids:
        requested = set(args.case_ids)
        unknown = requested - {case.case_id for case in cases}
        if unknown:
            raise ValueError(f"unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case.case_id in requested]
    if args.limit is not None:
        cases = cases[: args.limit]

    existing: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8")).get("records", [])
    by_id = {item["case_id"]: item for item in existing}
    episode_candidates_by_id: dict[str, EpisodeCandidate] = {}
    if args.episode_output and args.resume and args.episode_output.exists():
        for line_number, line in enumerate(
            args.episode_output.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                candidate = EpisodeCandidate(**json.loads(line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{args.episode_output}:{line_number}: {exc}") from exc
            episode_candidates_by_id[candidate.scenario_id] = candidate
    policy_model = args.policy_model or settings.llm_model
    if args.policy_base_url:
        client = LLMClient(
            base_url=args.policy_base_url,
            api_key=args.policy_api_key,
            using_vllm=True,
        )
        policy_backend = args.policy_base_url
    else:
        client = LLMClient()
        policy_backend = "configured_default"
    policy = NativeToolAgentPolicy(
        client,
        model=policy_model,
        temperature=args.policy_temperature,
        max_tokens=256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    await redis_client.connect()
    try:
        for case in cases:
            if case.case_id in by_id and not (
                args.rerun_failed and not bool(by_id[case.case_id].get("passed"))
            ):
                continue
            episode_collector: list[EpisodeCandidate] | None = (
                [] if args.episode_output else None
            )
            prefix = f"{case.case_id}:{args.rollout_id}:"
            for scenario_id in [
                item for item in episode_candidates_by_id if item.startswith(prefix)
            ]:
                episode_candidates_by_id.pop(scenario_id)
            record = await evaluate_case(
                case,
                policy=policy,
                rollout_id=args.rollout_id,
                episode_collector=episode_collector,
            )
            by_id[case.case_id] = record
            for candidate in episode_collector or []:
                episode_candidates_by_id[candidate.scenario_id] = candidate
            if args.episode_output:
                _write_episode_candidates(
                    args.episode_output,
                    list(episode_candidates_by_id.values()),
                )
            selected_records = [by_id[item.case_id] for item in cases if item.case_id in by_id]
            args.output.write_text(
                json.dumps(
                    build_report(
                        cases,
                        selected_records,
                        policy_model=policy_model,
                        policy_backend=policy_backend,
                    ),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "progress": f"{len(selected_records)}/{len(cases)}",
                        "case_id": case.case_id,
                        "passed": record["passed"],
                        "failures": record["failures"],
                        "solver_status": record.get("solver_status"),
                        "tokens": record["total_tokens"],
                        "latency_ms": record["latency_ms"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        await redis_client.disconnect()
    records = [by_id[case.case_id] for case in cases if case.case_id in by_id]
    report = build_report(
        cases,
        records,
        policy_model=policy_model,
        policy_backend=policy_backend,
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("sqlalchemy.engine").disabled = True
    final = asyncio.run(run(parse_args()))
    print(json.dumps(final["summary"], ensure_ascii=False, indent=2))
