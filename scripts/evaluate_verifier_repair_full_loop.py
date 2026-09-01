"""Evaluate verifier-repair checkpoints through a real retry/solve/validate loop."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import (  # noqa: E402
    GRPOCorpusRow,
    load_grpo_corpus,
    to_trl_environment_rows,
)
from agentic.local_policy import LocalCheckpointAgentPolicy  # noqa: E402
from agentic.loop import PolicyContext  # noqa: E402
from agentic.policy import VerifierRepairSpecialistRoutedAgentPolicy  # noqa: E402
from agentic.trl_environment import build_trl_environment_factories  # noqa: E402
from scripts.audit_model_curriculum import (  # noqa: E402
    paired_rollout_seed,
    select_verifier_repair_stratified,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_sha256(checkpoint: str | None) -> str | None:
    if not checkpoint:
        return None
    adapter = Path(checkpoint) / "adapter_model.safetensors"
    return _sha256(adapter) if adapter.is_file() else None


def _full_loop_row(row: GRPOCorpusRow) -> tuple[GRPOCorpusRow, dict[str, Any]]:
    full = row.model_copy(deep=True)
    contract = full.snapshot.hidden_test_facts.pop("grpo_decision_state", None)
    if not isinstance(contract, dict):
        raise ValueError(f"task has no verifier-repair contract: {row.task.task_id}")
    return full, contract


def score_downstream_rollout(
    *,
    target_action: str,
    episode_actions: list[str],
    gate_status: str,
    audit_metrics: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    """Require downstream execution, not only a well-formed review decision."""
    first_validation = (
        episode_actions.index("validate_itinerary")
        if "validate_itinerary" in episode_actions
        else -1
    )
    first_repair_action = (
        episode_actions[first_validation + 1]
        if 0 <= first_validation < len(episode_actions) - 1
        else None
    )
    checks = {
        "failed_validation_reached": first_validation >= 0,
        "first_repair_action_matches_target": first_repair_action == target_action,
        "gate_passed": gate_status == "passed",
    }
    if target_action == "retry_solve":
        retry_index = first_validation + 1 if first_repair_action == "retry_solve" else -1
        later = episode_actions[retry_index + 1 :]
        solve_index = later.index("solve_itinerary") if "solve_itinerary" in later else -1
        validate_index = (
            later.index("validate_itinerary") if "validate_itinerary" in later else -1
        )
        checks.update(
            {
                "fresh_solve_after_retry": solve_index >= 0,
                "fresh_validation_after_retry": validate_index > solve_index >= 0,
                "final_hard_pass": audit_metrics.get("hard_pass") is True,
            }
        )
    elif target_action == "propose_tradeoff":
        checks["awaiting_user"] = audit_metrics.get("needs_user_action_mismatch") is False
    elif target_action == "abort":
        checks["safe_termination"] = (
            audit_metrics.get("termination_action_mismatch") is False
            and audit_metrics.get("capability_termination_mismatch") is False
        )
    else:
        checks["supported_target"] = False
    return all(checks.values()), checks


async def _rollout(
    row: GRPOCorpusRow,
    policy: Any,
    *,
    max_tool_calling_iterations: int,
) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]]]:
    route = to_trl_environment_rows([row])[0]["environment"]
    environment = build_trl_environment_factories("react")[route]()
    rendered = environment.reset(
        task=row.task.model_dump(mode="json"),
        snapshot=row.snapshot.model_dump(mode="json"),
    )
    sampled_actions: list[dict[str, Any]] = []
    policy_errors: list[dict[str, Any]] = []
    try:
        for _ in range(max_tool_calling_iterations):
            payload = json.loads(rendered)
            if payload.get("done") is True:
                break
            state = payload.get("policy_state") or {}
            if not state.get("allowed_actions"):
                break
            try:
                action = await policy.propose(PolicyContext.model_validate(state))
            except Exception as exc:
                policy_errors.append(
                    {
                        "code": str(getattr(exc, "code", type(exc).__name__)),
                        "message": str(exc),
                    }
                )
                break
            sampled_actions.append(
                {
                    "action": action.action,
                    "arguments": action.arguments,
                    "token_usage": action.token_usage,
                    "route_trace": (
                        action.route_trace.model_dump(mode="json")
                        if action.route_trace is not None
                        else None
                    ),
                    "inference_metrics": (
                        action.inference_metrics.model_dump(mode="json")
                        if action.inference_metrics is not None
                        else None
                    ),
                }
            )
            rendered = environment._act(action.action, action.arguments)
    finally:
        environment.get_reward()
    rollout = environment.rollout_record
    if rollout is None:
        raise RuntimeError("full-loop verifier-repair rollout was not finalized")
    return rollout, sampled_actions, policy_errors


async def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_grpo_corpus(args.corpus_file)
    selected = select_verifier_repair_stratified(
        rows,
        args.tasks_per_target,
        offset_per_target=args.target_offset,
    )
    if not selected:
        raise ValueError("verifier-repair selection is empty")
    generalist = LocalCheckpointAgentPolicy(
        args.generalist,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
    )
    specialist = None
    policy: Any = generalist
    topology = "sft-generalist"
    if args.specialist:
        specialist = LocalCheckpointAgentPolicy(
            args.specialist,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            load_in_4bit=args.load_in_4bit,
        )
        policy = VerifierRepairSpecialistRoutedAgentPolicy(generalist, specialist)
        topology = "sft-generalist-plus-verifier-repair-specialist"

    rollout_rows = []
    try:
        for original in selected:
            full_row, contract = _full_loop_row(original)
            for sample_index in range(args.group_size):
                seed = paired_rollout_seed(
                    args.seed,
                    task_id=original.task.task_id,
                    sample_index=sample_index,
                )
                policy.set_rollout_seed(seed)
                started = time.perf_counter()
                rollout, sampled_actions, policy_errors = await _rollout(
                    full_row,
                    policy,
                    max_tool_calling_iterations=args.max_tool_calling_iterations,
                )
                episode_actions = [step.action.action for step in rollout.episode.steps]
                target = str(contract.get("target_action") or "")
                first_validation = (
                    episode_actions.index("validate_itinerary")
                    if "validate_itinerary" in episode_actions
                    else -1
                )
                repair_review_reached = (
                    first_validation >= 0
                    and len(episode_actions[first_validation + 1 :]) > 0
                )
                success, downstream_checks = score_downstream_rollout(
                    target_action=target,
                    episode_actions=episode_actions,
                    gate_status=rollout.reward.gate_status,
                    audit_metrics=rollout.reward.audit_metrics,
                )
                route_traces = [
                    action["route_trace"]
                    for action in sampled_actions
                    if action.get("route_trace")
                ]
                rollout_rows.append(
                    {
                        "task_id": original.task.task_id,
                        "source_task_id": contract.get("source_task_id"),
                        "source_snapshot_version": contract.get(
                            "source_snapshot_version"
                        ),
                        "target_action": target,
                        "sample_index": sample_index,
                        "rollout_seed": seed,
                        "success": success,
                        "downstream_checks": downstream_checks,
                        "gate_status": rollout.reward.gate_status,
                        "audit_metrics": rollout.reward.audit_metrics,
                        "episode_status": rollout.episode.status,
                        "termination_reason": rollout.episode.termination_reason,
                        "episode_actions": episode_actions,
                        "repair_review_reached": repair_review_reached,
                        "sampled_actions": sampled_actions,
                        "policy_errors": policy_errors,
                        "specialist_requested": any(
                            trace.get("requested_target") == "student"
                            for trace in route_traces
                        ),
                        "specialist_executed": any(
                            trace.get("executed_target") == "student"
                            for trace in route_traces
                        ),
                        "specialist_fallback": any(
                            trace.get("fallback_used") is True for trace in route_traces
                        ),
                        "scope_violation": any(
                            trace.get("fallback_error_code")
                            == "SPECIALIST_SCOPE_VIOLATION"
                            for trace in route_traces
                        ),
                        "latency_ms": round(
                            (time.perf_counter() - started) * 1000,
                            3,
                        ),
                        "completion_tokens": sum(
                            int(action.get("token_usage") or 0)
                            for action in sampled_actions
                        ),
                    }
                )
    finally:
        generalist.close()
        if specialist is not None:
            specialist.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "rollouts.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rollout_rows) + "\n",
        encoding="utf-8",
    )
    successes = sum(bool(row["success"]) for row in rollout_rows)
    report = {
        "schema_version": "verifier-repair-full-loop-audit.v1",
        "scope": "real production-style review to downstream termination audit",
        "topology": topology,
        "execution_mode": "react",
        "structured_decoding_mode": "native",
        "generalist": args.generalist,
        "generalist_adapter_sha256": _adapter_sha256(args.generalist),
        "specialist": args.specialist,
        "specialist_adapter_sha256": _adapter_sha256(args.specialist),
        "corpus_file": str(args.corpus_file),
        "corpus_sha256": _sha256(args.corpus_file),
        "seed": args.seed,
        "temperature": args.temperature,
        "group_size": args.group_size,
        "max_new_tokens": args.max_new_tokens,
        "max_tool_calling_iterations": args.max_tool_calling_iterations,
        "load_in_4bit": args.load_in_4bit,
        "tasks_per_target": args.tasks_per_target,
        "target_offset": args.target_offset,
        "tasks": len(selected),
        "independent_source_clusters": len(
            {str(row.get("source_task_id")) for row in rollout_rows}
        ),
        "rollouts": len(rollout_rows),
        "successes": successes,
        "success_rate": successes / len(rollout_rows),
        "by_target": {
            target: {
                "rollouts": len(target_rows),
                "successes": sum(bool(row["success"]) for row in target_rows),
                "success_rate": sum(bool(row["success"]) for row in target_rows)
                / len(target_rows),
            }
            for target in ("retry_solve", "propose_tradeoff", "abort")
            if (target_rows := [
                row for row in rollout_rows if row["target_action"] == target
            ])
        },
        "specialist_routing": {
            "requested": sum(bool(row["specialist_requested"]) for row in rollout_rows),
            "executed": sum(bool(row["specialist_executed"]) for row in rollout_rows),
            "fallbacks": sum(bool(row["specialist_fallback"]) for row in rollout_rows),
            "scope_violations": sum(bool(row["scope_violation"]) for row in rollout_rows),
        },
        "policy_errors": dict(
            Counter(
                error["code"]
                for row in rollout_rows
                for error in row.get("policy_errors") or []
            )
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generalist", required=True)
    parser.add_argument("--specialist")
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks-per-target", type=int, default=4)
    parser.add_argument("--target-offset", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    if args.tasks_per_target < 1 or args.group_size < 1:
        parser.error("tasks-per-target and group-size must be positive")
    if args.max_tool_calling_iterations < 8:
        parser.error("full-loop audit requires at least 8 tool-calling iterations")
    return args


if __name__ == "__main__":
    asyncio.run(evaluate(parse_args()))
