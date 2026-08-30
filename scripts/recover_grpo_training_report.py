"""Recover GRPO metadata when optimization and model save completed before reporting failed."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import (  # noqa: E402
    estimate_stateful_completion_budget,
    load_grpo_corpus,
    preflight_grpo_corpus,
    to_trl_environment_rows,
)
from agentic.reward import RewardConfig  # noqa: E402
from agentic.trl_environment import (  # noqa: E402
    VERIFIED_DECISION_STATE_REPLAY_CONTRACT,
    build_trl_environment_factories,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_train_metrics(log_path: Path) -> dict[str, Any]:
    for line in reversed(log_path.read_text(encoding="utf-8", errors="replace").splitlines()):
        if line.startswith("{'train_runtime':"):
            parsed = ast.literal_eval(line)
            if isinstance(parsed, dict):
                return parsed
    raise ValueError(f"completed train metrics not found in {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--execution-mode", default="policy_driven")
    parser.add_argument("--credit-mode", default="trajectory_b0")
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-tool-calling-iterations", type=int, required=True)
    parser.add_argument("--max-completion-length", type=int, required=True)
    parser.add_argument("--max-eval-tasks", type=int, default=0)
    parser.add_argument("--failure-log", type=Path, required=True)
    args = parser.parse_args()

    checkpoints = sorted(
        args.output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    if not checkpoints:
        raise FileNotFoundError("no completed checkpoint found")
    checkpoint = checkpoints[-1]
    trainer_state_path = checkpoint / "trainer_state.json"
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step") or 0) != int(
        trainer_state.get("max_steps") or -1
    ):
        raise ValueError("trainer state did not reach max_steps")

    adapter_path = args.output_dir / "adapter_model.safetensors"
    if not adapter_path.is_file():
        raise FileNotFoundError("saved adapter is missing")
    preflight = preflight_grpo_corpus(
        args.corpus_dir,
        minimum_train_tasks=1000,
        require_dependencies=False,
    )
    if preflight.errors:
        raise ValueError(f"corpus preflight failed: {preflight.errors}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.source_model, trust_remote_code=False)
    train_corpus = load_grpo_corpus(args.corpus_dir / "train.jsonl")
    validation_corpus = load_grpo_corpus(args.corpus_dir / "validation.jsonl")
    rollout_contracts = sorted(
        {str(row["rollout_contract"]) for row in to_trl_environment_rows(train_corpus)}
    )
    if args.max_eval_tasks > 0:
        validation_corpus = validation_corpus[: args.max_eval_tasks]
    completion_budget = estimate_stateful_completion_budget(
        [*train_corpus, *validation_corpus],
        tokenizer,
        build_trl_environment_factories(args.execution_mode),
    )
    report = {
        "status": "trained",
        "run_scope": "smoke",
        "method": "trajectory-level-agentic-grpo-b0",
        "credit_mode": args.credit_mode,
        "execution_mode": args.execution_mode,
        "policy_decision_scope": "all_dag_actions",
        "rollout_initialization_contract": (
            rollout_contracts[0] if len(rollout_contracts) == 1 else "mixed"
        ),
        "rollout_initialization_contracts": rollout_contracts,
        "teacher_trajectory_prefix": (
            VERIFIED_DECISION_STATE_REPLAY_CONTRACT in rollout_contracts
        ),
        "teacher_prefix_optimization_targets": False,
        "verified_replay_prefix_in_prompt": (
            VERIFIED_DECISION_STATE_REPLAY_CONTRACT in rollout_contracts
        ),
        "credit_assignment_claim": "trajectory-level only",
        "turn_credit_gamma": None,
        "turn_credit_blend": None,
        "turn_credit_totals": None,
        "turn_credit_gate_errors": [],
        "source_model": str(args.source_model),
        "continued_from_sft_adapter": (args.source_model / "adapter_config.json").is_file(),
        "git_commit": "unknown",
        "seed": args.seed,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "beta": args.beta,
        "max_tool_calling_iterations": args.max_tool_calling_iterations,
        "max_completion_length": args.max_completion_length,
        "completion_budget": completion_budget.model_dump(mode="json"),
        "reward": RewardConfig().config_version,
        "environment_versions": preflight.environment_versions,
        "snapshot_versions": preflight.snapshot_versions,
        "train_metrics": _final_train_metrics(args.output_dir / "run.log"),
        "optimization_history": trainer_state.get("log_history", []),
        "eval_metrics": {},
        "eval_status": "skipped_for_smoke_external_agent_loop_audit",
        "rollout_audit_path": None,
        "artifact": {
            "checkpoint": str(checkpoint),
            "global_step": trainer_state["global_step"],
            "adapter_sha256": _sha256(adapter_path),
        },
        "report_recovery": {
            "recovered": True,
            "reason": "metadata_exception_after_completed_optimization_and_model_save",
            "failure_log": str(args.failure_log),
            "trainer_state": str(trainer_state_path),
        },
    }
    destination = args.output_dir / "training_report.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
