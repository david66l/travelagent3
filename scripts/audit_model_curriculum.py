"""Route snapshot tasks using stochastic rollouts from the current policy."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.environment import EnvironmentRollout  # noqa: E402
from agentic.grpo import GRPOGroupAuditor, model_aware_curriculum  # noqa: E402
from agentic.grpo_training import (  # noqa: E402
    DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS,
    GRPOCorpusRow,
    MIN_POLICY_DRIVEN_TOOL_ITERATIONS,
    load_grpo_corpus,
    to_trl_environment_rows,
)
from agentic.local_policy import (  # noqa: E402
    LocalCheckpointAgentPolicy,
    parse_local_tool_call,
)
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT, PolicyOutputError  # noqa: E402
from agentic.policy_actions import policy_action_schemas  # noqa: E402
from agentic.trl_environment import (  # noqa: E402
    GRPOExecutionMode,
    build_trl_environment_factories,
)
from core.inference_metrics import summarize_inference_metrics  # noqa: E402


def task_family(row: GRPOCorpusRow) -> str:
    """Return the decision family that determines curriculum routing."""
    if row.task.missing_slots:
        return "clarification"
    if row.task.feasibility_report.get("feasible", True) is False:
        return "tradeoff"
    search = row.snapshot.tool_responses.get("search_pois") or []
    if search and (search[0].error_code or search[0].data_source == "unavailable"):
        return "recovery"
    return "search"


def boundary_stratum(row: GRPOCorpusRow) -> tuple[str, str] | None:
    """Return the synthetic decision-boundary kind and variant, when present."""
    metadata = row.snapshot.hidden_test_facts.get("decision_boundary_training")
    if not isinstance(metadata, dict):
        return None
    kind = metadata.get("boundary_kind")
    variant = metadata.get("variant")
    if not isinstance(kind, str) or not isinstance(variant, str):
        return None
    return kind, variant


def decision_loop_metadata(row: GRPOCorpusRow) -> dict[str, Any]:
    """Return the controlled Stage-3 decision-loop factors, when present."""
    metadata = row.snapshot.hidden_test_facts.get("decision_loop_curriculum")
    return metadata if isinstance(metadata, dict) else {}


def select_stratified(
    rows: list[GRPOCorpusRow],
    per_family: int,
    *,
    offset_per_family: int = 0,
) -> list[GRPOCorpusRow]:
    selected: dict[str, list[GRPOCorpusRow]] = defaultdict(list)
    seen: Counter[str] = Counter()
    for row in rows:
        family = task_family(row)
        if seen[family] < offset_per_family:
            seen[family] += 1
            continue
        if len(selected[family]) < per_family:
            selected[family].append(row)
    return [row for family in sorted(selected) for row in selected[family]]


def select_boundary_stratified(
    rows: list[GRPOCorpusRow],
    per_cell: int,
) -> list[GRPOCorpusRow]:
    """Select an equal number from each boundary-kind/variant cell."""
    selected: dict[tuple[str, str], list[GRPOCorpusRow]] = defaultdict(list)
    for row in rows:
        stratum = boundary_stratum(row)
        if stratum is not None and len(selected[stratum]) < per_cell:
            selected[stratum].append(row)
    return [row for stratum in sorted(selected) for row in selected[stratum]]


async def audit(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_grpo_corpus(args.corpus_file)
    if args.families:
        requested_families = set(args.families)
        rows = [row for row in rows if task_family(row) in requested_families]
    if args.boundary_kinds:
        requested_boundary_kinds = set(args.boundary_kinds)
        rows = [
            row
            for row in rows
            if (stratum := boundary_stratum(row)) is not None
            and stratum[0] in requested_boundary_kinds
        ]
    if args.boundary_variants:
        requested_boundary_variants = set(args.boundary_variants)
        rows = [
            row
            for row in rows
            if (stratum := boundary_stratum(row)) is not None
            and stratum[1] in requested_boundary_variants
        ]
    if args.decision_loop_scenarios:
        requested_scenarios = set(args.decision_loop_scenarios)
        rows = [
            row
            for row in rows
            if decision_loop_metadata(row).get("scenario") in requested_scenarios
        ]
    if args.decision_loop_evidence_styles:
        requested_styles = set(args.decision_loop_evidence_styles)
        rows = [
            row
            for row in rows
            if decision_loop_metadata(row).get("evidence_style") in requested_styles
        ]
    if args.decision_loop_target_positions:
        requested_positions = set(args.decision_loop_target_positions)
        rows = [
            row
            for row in rows
            if decision_loop_metadata(row).get("target_position")
            in requested_positions
        ]
    if args.tasks_per_boundary_cell is not None:
        selected = select_boundary_stratified(rows, args.tasks_per_boundary_cell)
    else:
        selected = select_stratified(
            rows,
            args.tasks_per_family,
            offset_per_family=args.family_offset,
        )
    policy = LocalCheckpointAgentPolicy(
        args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        do_sample=True,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
        structured_decoding=args.structured_decoding_mode,
    )
    auditor = GRPOGroupAuditor()
    environment_factories = build_trl_environment_factories(args.execution_mode)
    decisions = []
    rollout_rows: list[dict[str, Any]] = []
    try:
        for task_index, row in enumerate(selected, start=1):
            rollouts = []
            for sample_index in range(args.group_size):
                policy_errors: list[dict[str, Any]] = []
                policy_inference_metrics: list[dict[str, Any]] = []
                rollout_seed = paired_rollout_seed(
                    args.seed,
                    task_id=row.task.task_id,
                    sample_index=sample_index,
                )
                policy.set_rollout_seed(rollout_seed)
                rollout_started = time.perf_counter()
                rollout = await rollout_trl_history(
                    row,
                    policy,
                    execution_mode=args.execution_mode,
                    environment_factories=environment_factories,
                    max_tool_calling_iterations=args.max_tool_calling_iterations,
                    policy_errors=policy_errors,
                    policy_inference_metrics=policy_inference_metrics,
                )
                rollout_latency_ms = (time.perf_counter() - rollout_started) * 1000
                rollouts.append(rollout)
                rollout_rows.append(
                    {
                        "task_id": row.task.task_id,
                        "family": task_family(row),
                        "city": row.task.slots.get("destination"),
                        "decision_loop": decision_loop_metadata(row),
                        "sample_index": sample_index,
                        "rollout_seed": rollout_seed,
                        "trajectory_id": rollout.episode.trajectory_id,
                        "status": rollout.episode.status,
                        "termination_reason": rollout.episode.termination_reason,
                        "gate_status": rollout.reward.gate_status,
                        "reward": rollout.reward.episode_reward,
                        "latency_ms": round(rollout_latency_ms, 3),
                        "reward_config_version": rollout.reward.reward_config_version,
                        "reward_components": rollout.reward.components.model_dump(
                            mode="json"
                        ),
                        "turn_rewards": [
                            item.model_dump(mode="json")
                            for item in rollout.reward.turn_rewards
                        ],
                        "audit_metrics": rollout.reward.audit_metrics,
                        "policy_errors": policy_errors,
                        "actions": rollout_action_rows(
                            rollout,
                            policy_inference_metrics=policy_inference_metrics,
                        ),
                    }
                )
            decisions.append(
                auditor.evaluate(f"audit:{row.task.task_id}", rollouts)
            )
            print(
                f"[{task_index}/{len(selected)}] {row.task.task_id} "
                f"family={task_family(row)} rewards="
                f"{[item.reward.episode_reward for item in rollouts]}",
                flush=True,
            )
    finally:
        policy.close()

    priorities = model_aware_curriculum(decisions)
    route_by_task = {item.task_id: item.route for item in decisions}
    selected_by_id = {row.task.task_id: row for row in selected}
    routed = defaultdict(list)
    for task_id, route in route_by_task.items():
        routed[route].append(selected_by_id[task_id].model_dump(mode="json"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for route in ("grpo_update", "sft_repair", "evaluation", "reject"):
        _write_jsonl(args.output_dir / f"{route}.jsonl", routed[route])
    _write_jsonl(args.output_dir / "rollouts.jsonl", rollout_rows)
    _write_jsonl(
        args.output_dir / "group_decisions.jsonl",
        [item.model_dump(mode="json") for item in decisions],
    )
    report = {
        "schema_version": "model-aware-curriculum-audit.v1",
        "scope": "stochastic policy audit; not a training efficacy claim",
        "checkpoint": args.checkpoint,
        "execution_mode": args.execution_mode,
        "policy_decision_scope": (
            "all_dag_actions"
            if args.execution_mode == "policy_driven"
            else "delegated_choice_actions_only"
        ),
        "corpus_file": str(args.corpus_file),
        "seed": args.seed,
        "seed_protocol": "sha256-task-sample-v1",
        "temperature": args.temperature,
        "decoding_mode": {
            "native": "native-unconstrained",
            "json_schema": "outlines-json-schema-v1",
            "qwen_tool_envelope": "outlines-qwen-tool-envelope-v1",
        }[args.structured_decoding_mode],
        "quantization": "nf4-double-quant" if args.load_in_4bit else "none",
        "group_size": args.group_size,
        "tasks": len(selected),
        "requested_families": list(args.families or []),
        "family_offset": args.family_offset,
        "families": dict(Counter(task_family(row) for row in selected)),
        "boundary_cells": dict(
            Counter(
                "/".join(stratum)
                for row in selected
                if (stratum := boundary_stratum(row)) is not None
            )
        ),
        "requested_decision_loop_factors": {
            "scenarios": list(args.decision_loop_scenarios or []),
            "evidence_styles": list(args.decision_loop_evidence_styles or []),
            "target_positions": list(args.decision_loop_target_positions or []),
        },
        "routes": dict(Counter(item.route for item in decisions)),
        "decisions": [item.model_dump(mode="json") for item in decisions],
        "priorities": [item.model_dump(mode="json") for item in priorities],
        "behavior_gate": behavior_gate_metrics(rollout_rows),
        "decision_loop_breakdown": decision_loop_behavior_metrics(rollout_rows),
        "rollout_latency": rollout_latency_metrics(rollout_rows),
        "inference_metrics": summarize_inference_metrics(
            [
                action["inference_metrics"]
                for row in rollout_rows
                for action in row.get("actions") or []
                if action.get("inference_metrics")
            ]
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def paired_rollout_seed(base_seed: int, *, task_id: str, sample_index: int) -> int:
    """Derive a checkpoint-independent seed for one task/sample pair."""
    payload = f"{base_seed}:{task_id}:{sample_index}".encode()
    # Keep the value within torch's broadly portable signed 32-bit seed range.
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


async def rollout_trl_history(
    row: GRPOCorpusRow,
    policy: LocalCheckpointAgentPolicy,
    *,
    execution_mode: GRPOExecutionMode = "policy_driven",
    environment_factories: dict[str, Any] | None = None,
    max_tool_calling_iterations: int = DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS,
    policy_errors: list[dict[str, Any]] | None = None,
    policy_inference_metrics: list[dict[str, Any]] | None = None,
) -> EnvironmentRollout:
    """Drive the production TRL environment with the exact multi-turn history."""
    route, _initial_route_actions = _route_and_actions(row)
    factories = environment_factories or build_trl_environment_factories(execution_mode)
    environment = factories[route]()
    trl_row = to_trl_environment_rows([row])[0]
    reset_prompt = trl_row["prompt"]
    initial = environment.reset(
        task=row.task.model_dump(mode="json"),
        snapshot=row.snapshot.model_dump(mode="json"),
        prompt=reset_prompt,
    )
    if trl_row["rollout_contract"] == "verified_decision_state_replay.v1":
        messages = [dict(message) for message in reset_prompt]
        rendered_transition = str(messages[-1]["content"])
    else:
        messages = [
            {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
            {"role": "user", "content": initial},
        ]
        rendered_transition = initial
    try:
        for _ in range(max_tool_calling_iterations):
            transition_payload = json.loads(rendered_transition)
            policy_state = transition_payload.get("policy_state") or {}
            allowed_actions = list(policy_state.get("allowed_actions") or [])
            if not allowed_actions:
                break
            # The production Scheduler changes this subset after every verified
            # transition. Rebuilding the schema here is essential: keeping the
            # first route's actions would turn a nine-decision rollout into a
            # one-step audit even though the environment itself is policy-driven.
            tools = policy_action_schemas(allowed_actions)
            try:
                action = await policy.propose_from_history(
                    messages,
                    tools=tools,
                    allowed_actions=allowed_actions,
                )
            except PolicyOutputError as exc:
                # Match TRL: a completion without a valid tool call ends this
                # rollout. get_reward() below finalizes it as truncated/failed
                # instead of aborting the whole curriculum audit.
                if policy_errors is not None:
                    policy_errors.append(
                        {
                            "code": exc.code,
                            "message": str(exc),
                            "raw_output": exc.raw_output,
                        }
                    )
                break
            if action.inference_metrics is not None and policy_inference_metrics is not None:
                policy_inference_metrics.append(
                    action.inference_metrics.model_dump(mode="json")
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": action.action,
                                "arguments": action.arguments,
                            },
                        }
                    ],
                }
            )
            # The offline auditor rebuilds the exact state-scoped schema above.
            # Submit the already validated action directly so the audit can cover
            # recovery/termination actions that are intentionally absent from
            # TRL's static public method schema for the initial route.
            result = environment._act(action.action, action.arguments)
            messages.append(
                {"role": "tool", "name": action.action, "content": result}
            )
            rendered_transition = result
            if json.loads(rendered_transition).get("done") is True:
                break
    finally:
        environment.get_reward()
    rollout = environment.rollout_record
    if rollout is None:
        raise RuntimeError("TRL history audit did not produce a scored rollout")
    return rollout


def rollout_action_rows(
    rollout: EnvironmentRollout,
    *,
    policy_inference_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reattach out-of-band inference evidence to TRL's tool-only episode."""
    policy_steps = [
        step
        for step in rollout.episode.steps
        if step.action.decision_source != "controller"
    ]
    return [
        {
            "step_index": step.step_index,
            "task_id": step.task_id,
            "action": step.action.action,
            "arguments": step.action.arguments,
            "decision_source": step.action.decision_source,
            "allowed_actions": step.context.allowed_actions,
            "decision_cardinality": len(step.context.allowed_actions),
            "token_usage": metrics.get("completion_tokens", step.action.token_usage),
            "inference_metrics": metrics,
            "error_code": step.verification.get("error_code"),
            "verification": step.verification,
            "observations": [
                {
                    "tool": item.tool,
                    "ok": item.ok,
                    "error_code": item.error.code if item.error else None,
                    "is_fallback": item.is_fallback,
                }
                for item in step.observations
            ],
        }
        for step, metrics in zip(
            policy_steps,
            policy_inference_metrics,
            strict=False,
        )
    ]


def _route_and_actions(row: GRPOCorpusRow) -> tuple[str, list[str]]:
    route = to_trl_environment_rows([row])[0]["environment"]
    initial_actions = {
        "clarification": ["ask_user"],
        "tradeoff": ["propose_tradeoff", "abort"],
        "search": ["search_pois"],
        "search_current": ["search_pois", "search_current_info"],
        "search_transport": ["search_pois", "search_transport"],
        "decision_get_poi_detail": ["get_poi_detail"],
    }
    return route, initial_actions[route]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def behavior_gate_metrics(rollout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize model-visible action validity before checkpoint promotion."""
    total = len(rollout_rows)
    empty_actions = sum(not row.get("actions") for row in rollout_rows)
    successful = sum(
        row.get("gate_status") == "passed" and float(row.get("reward") or 0) > 0
        for row in rollout_rows
    )
    invalid_error_codes = {
        "ACTION_NOT_ALLOWED",
        "ARGUMENT_GROUNDING_FAILED",
        "POLICY_ARGUMENT_INVALID",
        "SNAPSHOT_ARGUMENT_MISMATCH",
    }
    invalid_actions = sum(
        action.get("error_code") in invalid_error_codes
        for row in rollout_rows
        for action in row.get("actions") or []
    )
    policy_errors = [
        error
        for row in rollout_rows
        for error in row.get("policy_errors") or []
    ]
    argument_errors = [
        error for error in policy_errors if error.get("code") == "POLICY_ARGUMENT_INVALID"
    ]
    protected_argument_names = {
        "city",
        "trusted_city",
        "max_results",
        "constraints",
        "facts",
        "matrices",
        "itineraries",
    }
    protected_argument_errors = 0
    unknown_argument_errors = 0
    for error in argument_errors:
        raw_output = error.get("raw_output")
        if not raw_output:
            continue
        try:
            action, arguments = parse_local_tool_call(str(raw_output))
            schema = policy_action_schemas([action])[0]
        except (PolicyOutputError, ValueError):
            continue
        allowed = set(
            schema["function"]["parameters"].get("properties") or {}
        )
        supplied = set(arguments)
        if supplied & protected_argument_names:
            protected_argument_errors += 1
        if supplied - allowed:
            unknown_argument_errors += 1
    return {
        "rollouts": total,
        "successful_rollouts": successful,
        "success_rate": successful / total if total else 0.0,
        "empty_action_rollouts": empty_actions,
        "empty_action_rate": empty_actions / total if total else 0.0,
        "invalid_actions": invalid_actions,
        "invalid_action_rate": invalid_actions / total if total else 0.0,
        "policy_output_errors": len(policy_errors),
        "policy_output_error_rate": len(policy_errors) / total if total else 0.0,
        "policy_argument_errors": len(argument_errors),
        "policy_argument_error_rate": len(argument_errors) / total if total else 0.0,
        "unknown_argument_errors": unknown_argument_errors,
        "unknown_argument_error_rate": unknown_argument_errors / total if total else 0.0,
        "protected_argument_errors": protected_argument_errors,
        "protected_argument_error_rate": (
            protected_argument_errors / total if total else 0.0
        ),
    }


def decision_loop_behavior_metrics(
    rollout_rows: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Expose success by each orthogonal factor instead of only a pooled score."""
    factor_names = (
        "scenario",
        "evidence_style",
        "target_position",
        "city",
        "scenario_evidence_position",
    )
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        factor: defaultdict(list) for factor in factor_names
    }
    for row in rollout_rows:
        metadata = row.get("decision_loop") or {}
        values = {
            "scenario": metadata.get("scenario"),
            "evidence_style": metadata.get("evidence_style"),
            "target_position": metadata.get("target_position"),
            "city": row.get("city"),
            "scenario_evidence_position": (
                "|".join(
                    (
                        str(metadata["scenario"]),
                        str(metadata["evidence_style"]),
                        str(metadata["target_position"]),
                    )
                )
                if all(
                    metadata.get(name) is not None
                    for name in ("scenario", "evidence_style", "target_position")
                )
                else None
            ),
        }
        for factor, value in values.items():
            if value is not None:
                grouped[factor][str(value)].append(row)

    report: dict[str, dict[str, dict[str, Any]]] = {}
    for factor, cells in grouped.items():
        report[factor] = {}
        for value, rows in sorted(cells.items()):
            successes = sum(
                row.get("gate_status") == "passed"
                and float(row.get("reward") or 0) > 0
                for row in rows
            )
            action_counts = [len(row.get("actions") or []) for row in rows]
            policy_errors = sum(bool(row.get("policy_errors")) for row in rows)
            report[factor][value] = {
                "rollouts": len(rows),
                "successful_rollouts": successes,
                "success_rate": successes / len(rows),
                "mean_policy_actions": sum(action_counts) / len(action_counts),
                "policy_error_rate": policy_errors / len(rows),
            }
    return report


def rollout_latency_metrics(rollout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize end-to-end rollout latency without mixing in model load time."""
    values = sorted(
        float(row["latency_ms"])
        for row in rollout_rows
        if row.get("latency_ms") is not None
    )
    if not values:
        return {"rollouts": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}

    def percentile(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    return {
        "rollouts": len(values),
        "mean_ms": round(sum(values) / len(values), 3),
        "p50_ms": round(percentile(0.50), 3),
        "p95_ms": round(percentile(0.95), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks-per-family", type=int, default=4)
    parser.add_argument(
        "--tasks-per-boundary-cell",
        type=int,
        help="Select this many tasks from each boundary-kind/variant cell.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=("clarification", "recovery", "search", "tradeoff"),
        help="Optionally audit only the named task families.",
    )
    parser.add_argument(
        "--family-offset",
        type=int,
        default=0,
        help="Skip this many tasks in each family before selecting the audit set.",
    )
    parser.add_argument(
        "--boundary-kinds",
        nargs="+",
        choices=("infeasible", "unsafe", "missing_tool"),
        help="Optionally restrict synthetic decision-boundary status kinds.",
    )
    parser.add_argument(
        "--boundary-variants",
        nargs="+",
        choices=("actionable_tradeoff", "necessary_abort"),
        help="Optionally restrict synthetic decision-boundary variants.",
    )
    parser.add_argument(
        "--decision-loop-scenarios",
        nargs="+",
        choices=("change_arguments", "retry_same_arguments"),
        help="Optionally restrict Stage-3 decision-loop recovery scenarios.",
    )
    parser.add_argument(
        "--decision-loop-evidence-styles",
        nargs="+",
        choices=("explicit_instruction", "diagnostic_evidence"),
        help="Optionally restrict Stage-3 decision-loop evidence styles.",
    )
    parser.add_argument(
        "--decision-loop-target-positions",
        nargs="+",
        type=int,
        choices=(0, 1),
        help="Optionally restrict which initial keyword position is correct.",
    )
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--execution-mode",
        choices=("policy_driven", "controller_first", "react"),
        default="policy_driven",
    )
    parser.add_argument(
        "--max-tool-calling-iterations",
        type=int,
        default=DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS,
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Match the NF4 QLoRA policy used by the GRPO trainer.",
    )
    parser.add_argument(
        "--structured-decoding",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--structured-decoding-mode",
        choices=("native", "json_schema", "qwen_tool_envelope"),
        default="native",
        help=(
            "Choose native generation, plain JSON Schema masking, or the same "
            "schema inside Qwen's native tool-call envelope. Constrained modes "
            "measure system-level legality, not native model legality."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.structured_decoding:
        if args.structured_decoding_mode != "native":
            parser.error(
                "--structured-decoding cannot be combined with "
                "--structured-decoding-mode"
            )
        args.structured_decoding_mode = "json_schema"
    if args.tasks_per_family < 1:
        parser.error("tasks-per-family must be positive")
    if args.tasks_per_boundary_cell is not None and args.tasks_per_boundary_cell < 1:
        parser.error("tasks-per-boundary-cell must be positive")
    if args.family_offset < 0:
        parser.error("family-offset must not be negative")
    if args.group_size < 4:
        parser.error("group-size must be at least 4")
    if args.max_tool_calling_iterations < 1:
        parser.error("max-tool-calling-iterations must be positive")
    if (
        args.execution_mode == "policy_driven"
        and args.max_tool_calling_iterations < MIN_POLICY_DRIVEN_TOOL_ITERATIONS
    ):
        parser.error(
            "policy_driven audit requires at least "
            f"{MIN_POLICY_DRIVEN_TOOL_ITERATIONS} tool-calling iterations"
        )
    report = asyncio.run(audit(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
