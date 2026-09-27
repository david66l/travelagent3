"""Generate auditable H-006 donor trajectories from a local PEFT checkpoint.

The command runs the same transported ReAct environment used by the training
and internal-evaluation stack.  It stores every sampled episode, including
failures, and marks selection eligibility without silently converting system
assembly into model credit.
"""

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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from agentic.distillation import build_teacher_candidate  # noqa: E402
from agentic.grpo_training import load_grpo_corpus  # noqa: E402
from agentic.local_policy import LocalCheckpointAgentPolicy  # noqa: E402
from agentic.trajectory import EpisodeReplayVerifier  # noqa: E402
from agentic.trl_environment import build_trl_environment_factories  # noqa: E402
from evaluation.posttraining_promotion_protocol import canonical_hash  # noqa: E402
from scripts.audit_model_curriculum import (  # noqa: E402
    paired_rollout_seed,
    rollout_action_rows,
    rollout_trl_history,
    transport_trl_environment_rows,
    verifier_repair_metadata,
)
from scripts.evaluate_tradeoff_grounding_sft import (  # noqa: E402
    effective_tokenizer_manifest,
    evaluation_code_snapshot,
    model_file_manifest,
)


SCHEMA_VERSION = "h006-local-donor-rollouts.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decision_selection_gate(
    *,
    audit_metrics: dict[str, Any],
    policy_errors: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, bool]:
    """Credit only the model-owned semantics of one verifier-repair decision."""
    action = actions[-1] if len(actions) == 1 else {}
    raw_semantic = bool(audit_metrics.get("decision_model_semantic_valid"))
    assembled = bool(audit_metrics.get("decision_system_step_valid"))
    deterministic = bool(audit_metrics.get("decision_execution_valid"))
    model_contract = bool(action.get("model_contract_compliant"))
    no_override = not bool(action.get("controller_override_attempt"))
    no_rejection = not policy_errors and bool(
        audit_metrics.get("decision_no_policy_call_rejections", True)
    )
    exactly_one = len(actions) == 1
    eligible = all(
        (
            raw_semantic,
            assembled,
            deterministic,
            model_contract,
            no_override,
            no_rejection,
            exactly_one,
        )
    )
    return {
        "raw_model_semantic_success": raw_semantic,
        "assembled_system_success": assembled,
        "deterministic_verifier_success": deterministic,
        "model_contract_compliant": model_contract,
        "no_controller_override": no_override,
        "no_policy_rejection": no_rejection,
        "exactly_one_sampled_decision": exactly_one,
        "selection_eligible": eligible,
    }


def normal_selection_gate(
    *,
    successful: bool,
    hard_pass: bool,
    replay_errors: list[str],
    policy_errors: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, bool]:
    contract = bool(actions) and all(
        bool(action.get("model_contract_compliant")) for action in actions
    )
    no_override = all(
        not bool(action.get("controller_override_attempt")) for action in actions
    )
    eligible = all(
        (
            successful,
            hard_pass,
            not replay_errors,
            not policy_errors,
            contract,
            no_override,
        )
    )
    return {
        "raw_model_semantic_success": bool(successful),
        "assembled_system_success": bool(successful),
        "deterministic_verifier_success": bool(hard_pass),
        "model_contract_compliant": contract,
        "no_controller_override": no_override,
        "no_policy_rejection": not policy_errors,
        "episode_replay_valid": not replay_errors,
        "selection_eligible": eligible,
    }


def _family(row: Any) -> str:
    decision = verifier_repair_metadata(row)
    if decision:
        return str(decision.get("scenario_family") or decision.get("target_action"))
    return str(row.task.template_family)


def _source_id(row: Any) -> str:
    decision = verifier_repair_metadata(row)
    if decision:
        return str(decision.get("source_task_id") or row.task.task_id)
    lineage = row.snapshot.hidden_test_facts.get("training_lineage")
    if isinstance(lineage, dict) and lineage.get("source_state_id"):
        return str(lineage["source_state_id"])
    return str(row.task.task_id)


def _template_lineage(row: Any) -> str:
    lineage = row.snapshot.hidden_test_facts.get("training_lineage")
    if isinstance(lineage, dict):
        return str(lineage.get("template_lineage") or "unknown")
    return "unknown"


async def generate(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_dir}")
    rows = load_grpo_corpus(args.corpus_file)
    if args.task_offset:
        rows = rows[args.task_offset :]
    if args.max_tasks is not None:
        rows = rows[: args.max_tasks]
    if not rows:
        raise ValueError("no donor rows selected")
    if any(
        not isinstance(row.snapshot.hidden_test_facts.get("training_lineage"), dict)
        for row in rows
    ):
        raise ValueError("all H-006 donor rows require embedded training_lineage")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    corpus_sha256 = _sha256(args.corpus_file)
    checkpoint_files = model_file_manifest(args.checkpoint)
    code_snapshot = evaluation_code_snapshot()
    policy = LocalCheckpointAgentPolicy(
        args.checkpoint,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        do_sample=True,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
        structured_decoding="native",
    )
    tokenizer_manifest = effective_tokenizer_manifest(policy.tokenizer)
    transported_rows, transport_evidence = transport_trl_environment_rows(rows)
    factories = build_trl_environment_factories("react")
    replay = EpisodeReplayVerifier()
    output_path = args.output_dir / "rollouts.jsonl"
    records: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        with output_path.open("x", encoding="utf-8") as output:
            for task_index, (row, transported) in enumerate(
                zip(rows, transported_rows, strict=True),
                start=1,
            ):
                decision = verifier_repair_metadata(row)
                for sample_index in range(args.samples_per_task):
                    rollout_seed = paired_rollout_seed(
                        args.seed,
                        task_id=row.task.task_id,
                        sample_index=sample_index,
                    )
                    policy.set_rollout_seed(rollout_seed)
                    policy_errors: list[dict[str, Any]] = []
                    inference_metrics: list[dict[str, Any]] = []
                    rollout = await rollout_trl_history(
                        row,
                        policy,
                        execution_mode="react",
                        environment_factories=factories,
                        max_tool_calling_iterations=args.max_tool_calling_iterations,
                        policy_errors=policy_errors,
                        policy_inference_metrics=inference_metrics,
                        transported_row=transported,
                    )
                    candidate = build_teacher_candidate(
                        rollout,
                        family=_family(row),
                        sample_index=sample_index,
                    )
                    replay_errors = replay.verify(rollout.episode)
                    actions = rollout_action_rows(
                        rollout,
                        policy_inference_metrics=inference_metrics,
                    )
                    if decision:
                        gates = decision_selection_gate(
                            audit_metrics=rollout.reward.audit_metrics,
                            policy_errors=policy_errors,
                            actions=actions,
                        )
                    else:
                        gates = normal_selection_gate(
                            successful=candidate.score.successful,
                            hard_pass=candidate.score.hard_pass,
                            replay_errors=replay_errors,
                            policy_errors=policy_errors,
                            actions=actions,
                        )
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "task_id": row.task.task_id,
                        "source_state_id": _source_id(row),
                        "template_lineage": _template_lineage(row),
                        "target_action": decision.get("target_action") if decision else None,
                        "family": _family(row),
                        "sample_index": sample_index,
                        "rollout_seed": rollout_seed,
                        "initial_state_fingerprint": rollout.initial_state_fingerprint,
                        "gates": gates,
                        "policy_errors": policy_errors,
                        "replay_error_hashes": [canonical_hash(item) for item in replay_errors],
                        "actions": actions,
                        "candidate": candidate.model_dump(mode="json"),
                    }
                    output.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
                    output.flush()
                    records.append(record)
                eligible = sum(
                    bool(record["gates"]["selection_eligible"])
                    for record in records[-args.samples_per_task :]
                )
                print(
                    f"[{task_index}/{len(rows)}] {row.task.task_id} "
                    f"eligible={eligible}/{args.samples_per_task}",
                    flush=True,
                )
    finally:
        policy.close()

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["source_state_id"]].append(record)
    successful_sources = sum(
        any(item["gates"]["selection_eligible"] for item in group)
        for group in by_source.values()
    )
    eligible_records = [item for item in records if item["gates"]["selection_eligible"]]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "training_authorized": False,
        "checkpoint": args.checkpoint,
        "effective_model_manifest": {
            "model_files": checkpoint_files,
            "tokenizer": tokenizer_manifest,
        },
        "evaluation_code_snapshot": code_snapshot,
        "corpus_file": str(args.corpus_file),
        "corpus_sha256": corpus_sha256,
        "transport_evidence": transport_evidence,
        "seed": args.seed,
        "seed_protocol": "sha256-task-sample-v1",
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "samples_per_task": args.samples_per_task,
        "load_in_4bit": args.load_in_4bit,
        "tasks": len(rows),
        "independent_sources": len(by_source),
        "rollouts": len(records),
        "eligible_rollouts": len(eligible_records),
        "successful_sources": successful_sources,
        "eligible_rollouts_per_target": dict(
            sorted(Counter(str(item.get("target_action")) for item in eligible_records).items())
        ),
        "eligible_rollouts_per_family": dict(
            sorted(Counter(item["family"] for item in eligible_records).items())
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "rollouts_sha256": _sha256(output_path),
        "next_gate": (
            "source-equal success selection, causal visibility, leakage and 40/40/20 mix audit"
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=14)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()
    if args.task_offset < 0 or min(args.samples_per_task, args.max_new_tokens) < 1:
        parser.error("offset must be non-negative and sample/token counts must be positive")
    if args.max_tasks is not None and args.max_tasks < 1:
        parser.error("max-tasks must be positive")
    if args.temperature <= 0:
        parser.error("temperature must be positive")
    report = asyncio.run(generate(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
