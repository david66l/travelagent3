"""Real TRL Agentic GRPO-B0 entrypoint for snapshot travel environments."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import (  # noqa: E402
    DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS,
    MIN_POLICY_DRIVEN_TOOL_ITERATIONS,
    MIN_STATEFUL_COMPLETION_LENGTH,
    estimate_stateful_completion_budget,
    load_grpo_corpus,
    preflight_grpo_corpus,
    tool_result_suffix_ids,
    to_trl_environment_rows,
)
from agentic.reward import RewardConfig  # noqa: E402
from agentic.trl_environment import (  # noqa: E402
    VERIFIED_DECISION_STATE_REPLAY_CONTRACT,
    build_trl_environment_factories,
)


def _disable_unused_vllm_import() -> None:
    """Keep TRL's optional vLLM backend out of non-vLLM training imports.

    TRL imports its vLLM generation module whenever any vLLM distribution is
    installed, even when ``use_vllm=False``. Serving environments commonly pin
    an older vLLM release, so that eager optional import can prevent the native
    Transformers GRPO backend from starting. Patching TRL's availability probe
    is process-local and leaves the installed serving runtime untouched.
    """
    trl_import_utils = importlib.import_module("trl.import_utils")
    trl_import_utils.is_vllm_available = lambda min_version=None: False


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def validate_turn_credit_totals(totals: dict | None) -> list[str]:
    """Return hard R1-v2 evidence failures; an empty list is publishable."""
    if not totals:
        return ["TURN_CREDIT_TOTALS_MISSING"]
    errors: list[str] = []
    if int(totals.get("train_effective_nonzero_credited_turns") or 0) <= 0:
        errors.append("NO_EFFECTIVE_NONZERO_TRAIN_TURN_CREDIT")
    compared = int(totals.get("train_compared_turn_buckets") or 0)
    zero_variance = int(totals.get("train_zero_variance_turn_buckets") or 0)
    if compared <= 0:
        errors.append("NO_COMPARABLE_TURN_BUCKETS")
    elif zero_variance >= compared:
        errors.append("ALL_COMPARABLE_TURN_BUCKETS_ZERO_VARIANCE")
    if int(totals.get("train_invalid_action_positive_credit_count") or 0) > 0:
        errors.append("INVALID_ACTION_RECEIVED_POSITIVE_CREDIT")
    if int(totals.get("alignment_rejected_trajectories") or 0) > 0:
        errors.append("TURN_TO_TOKEN_ALIGNMENT_NOT_PROVEN")
    if int(totals.get("extra_unmatched_model_turns") or 0) > 0:
        errors.append("EXTRA_UNMATCHED_MODEL_TURNS")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model", required=True, help="SFT checkpoint or merged policy model"
    )
    parser.add_argument(
        "--tokenizer",
        help=(
            "Optional tokenizer path/name. Use the base tokenizer when an archived "
            "adapter contains tokenizer metadata from an older Transformers release."
        ),
    )
    parser.add_argument("--minimum-train-tasks", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument(
        "--execution-mode",
        choices=("policy_driven", "controller_first", "react"),
        default="react",
        help=(
            "react matches production: the model owns research/recovery choices while "
            "the controller advances deterministic gates. controller_first is the old "
            "narrow baseline; policy_driven remains a full-DAG research stress mode."
        ),
    )
    parser.add_argument(
        "--max-tool-calling-iterations",
        type=int,
        default=DEFAULT_POLICY_DRIVEN_TOOL_ITERATIONS,
        help=(
            "Maximum policy tool rounds. Controller-first recovery usually needs "
            "two decisions; full-DAG policy_driven audits need the larger default."
        ),
    )
    parser.add_argument("--max-completion-length", type=int, default=16384)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("linear", "cosine", "constant", "constant_with_warmup"),
        default="linear",
        help=(
            "Optimizer learning-rate schedule. Short bounded GRPO runs should set "
            "this explicitly; the linear default otherwise decays almost to zero "
            "before a small curriculum has been covered."
        ),
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.0,
        help="Fraction of optimizer steps used for LR warmup.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="On-policy rollout sampling temperature; must match the exploration audit.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL penalty against the frozen reference policy.",
    )
    parser.add_argument(
        "--credit-mode",
        choices=("trajectory_b0", "turn_r1"),
        default="trajectory_b0",
        help="B0 uses one group-relative trajectory advantage; R1 blends verified turn credit.",
    )
    parser.add_argument("--turn-credit-gamma", type=float, default=0.95)
    parser.add_argument("--turn-credit-blend", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="Checkpoint interval; staged 20/40/60 audits should set this to 20.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=3,
        help="Number of checkpoints to retain; keep at least 3 for 20/40/60 audits.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values bound smoke/debug runs; formal training keeps -1.",
    )
    parser.add_argument(
        "--max-train-tasks",
        type=int,
        default=0,
        help="Optional deterministic prefix used only to keep smoke runs small.",
    )
    parser.add_argument(
        "--max-eval-tasks",
        type=int,
        default=0,
        help="Optional deterministic validation prefix used only for smoke runs.",
    )
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-small-corpus",
        action="store_true",
        help="Smoke test only; never use this flag for a reported training run.",
    )
    args = parser.parse_args()

    if args.num_generations < 4:
        raise ValueError(
            "GRPO num_generations must be at least 4 for meaningful group variance"
        )
    if args.beta < 0:
        raise ValueError("GRPO beta must be non-negative")
    if args.temperature <= 0:
        raise ValueError("GRPO temperature must be positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("GRPO warmup-ratio must be in [0, 1)")
    if args.save_steps <= 0:
        raise ValueError("GRPO save-steps must be positive")
    if args.save_total_limit <= 0:
        raise ValueError("GRPO save-total-limit must be positive")
    if not 0 < args.turn_credit_gamma <= 1:
        raise ValueError("turn-credit-gamma must be in (0, 1]")
    if not 0 <= args.turn_credit_blend <= 1:
        raise ValueError("turn-credit-blend must be in [0, 1]")
    if args.credit_mode == "turn_r1" and args.max_tool_calling_iterations < 2:
        raise ValueError("turn_r1 requires at least two tool-calling iterations")
    if (
        args.execution_mode == "policy_driven"
        and args.max_tool_calling_iterations < MIN_POLICY_DRIVEN_TOOL_ITERATIONS
    ):
        raise ValueError(
            "policy_driven GRPO requires at least "
            f"{MIN_POLICY_DRIVEN_TOOL_ITERATIONS} tool-calling iterations for the "
            "nominal production DAG"
        )
    if args.max_completion_length < MIN_STATEFUL_COMPLETION_LENGTH:
        raise ValueError(
            "stateful Agentic GRPO max-completion-length is too small: "
            f"{args.max_completion_length}<{MIN_STATEFUL_COMPLETION_LENGTH}. "
            "TRL counts tool-result state against this budget, so a smaller value "
            "can silently remove retry/follow-up generations."
        )
    effective_batch = args.batch_size * args.gradient_accumulation
    if effective_batch % args.num_generations:
        raise ValueError(
            "batch-size * gradient-accumulation must be divisible by num-generations "
            "on a single-GPU run"
        )
    environment_factories = build_trl_environment_factories(args.execution_mode)
    minimum = 1 if args.allow_small_corpus else args.minimum_train_tasks
    report = preflight_grpo_corpus(
        args.corpus_dir,
        minimum_train_tasks=minimum,
        require_dependencies=not args.preflight_only,
    )
    print(report.model_dump_json(indent=2))
    if args.preflight_only:
        return (
            0
            if not [error for error in report.errors if "DEPENDENCIES" not in error]
            else 2
        )
    if not report.ready:
        return 2

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    if not args.use_vllm:
        _disable_unused_vllm_import()
    from trl import GRPOConfig, GRPOTrainer

    class StableToolSuffixGRPOTrainer(GRPOTrainer):
        """Use EOS-boundary alignment for Qwen's conditional thinking template."""

        def _get_tool_suffix_ids(self, tool_messages):
            return tool_result_suffix_ids(
                self.processing_class,
                tool_messages=tool_messages,
                chat_template=self.chat_template,
                chat_template_kwargs=self.chat_template_kwargs,
            )

    if not torch.cuda.is_available():
        raise RuntimeError("Agentic GRPO training requires a CUDA GPU")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer_source = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise RuntimeError(
            "policy checkpoint must provide a native tool-capable chat_template"
        )

    train_corpus = load_grpo_corpus(args.corpus_dir / "train.jsonl")
    validation_corpus = load_grpo_corpus(args.corpus_dir / "validation.jsonl")
    if args.max_train_tasks > 0:
        train_corpus = train_corpus[: args.max_train_tasks]
    if args.max_eval_tasks > 0:
        validation_corpus = validation_corpus[: args.max_eval_tasks]
    if len(validation_corpus) < args.num_generations:
        raise ValueError(
            "validation corpus must contain at least num-generations tasks "
            f"({len(validation_corpus)}<{args.num_generations})"
        )
    train_rows = to_trl_environment_rows(train_corpus)
    validation_rows = to_trl_environment_rows(validation_corpus)
    rollout_contracts = sorted({str(row["rollout_contract"]) for row in train_rows})
    completion_budget = estimate_stateful_completion_budget(
        [*train_corpus, *validation_corpus],
        tokenizer,
        environment_factories,
    )
    print(completion_budget.model_dump_json(indent=2))
    if args.max_completion_length < completion_budget.minimum_completion_length:
        raise ValueError(
            "max-completion-length cannot hold the measured production tool result "
            "and a follow-up action: "
            f"{args.max_completion_length}<{completion_budget.minimum_completion_length} "
            f"(limiting task: {completion_budget.limiting_task_id})"
        )
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    adapter_config = Path(args.model) / "adapter_config.json"
    if adapter_config.is_file():
        # Continue optimizing the existing SFT adapter. Passing an adapter path
        # as a plain model id makes Transformers look for full-model weights and
        # either fail or silently start from the wrong policy.
        peft_config = PeftConfig.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=quantization,
            device_map="auto",
            dtype=compute_dtype,
            trust_remote_code=False,
        )
        base_model = prepare_model_for_kbit_training(
            base_model,
            use_gradient_checkpointing=True,
        )
        model = PeftModel.from_pretrained(base_model, args.model, is_trainable=True)
        trainer_quantization = None
        trainer_peft = None
        model_init_kwargs = None
        continued_from_adapter = True
    else:
        model = args.model
        trainer_quantization = quantization
        trainer_peft = lora
        model_init_kwargs = {"dtype": compute_dtype, "trust_remote_code": False}
        continued_from_adapter = False
    report_to = ["mlflow"] if os.environ.get("MLFLOW_TRACKING_URI") else []
    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        # TRL scores evaluation rewards in generation groups as well. A batch
        # smaller than the group cannot be reshaped into preference groups.
        per_device_eval_batch_size=args.num_generations,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        temperature=args.temperature,
        beta=args.beta,
        # Current TRL recommends DAPO-style token aggregation. The historical
        # `grpo` loss is length-biased and rewards short positive trajectories.
        loss_type="dapo",
        scale_rewards="group",
        mask_truncated_completions=True,
        chat_template_kwargs={"enable_thinking": False},
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        eval_strategy="no" if args.max_steps > 0 else "steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=5,
        seed=args.seed,
        report_to=report_to,
        run_name=(
            f"agent-policy-grpo-b0-{args.execution_mode}-{Path(args.model).name}"
        ),
        model_init_kwargs=model_init_kwargs,
    )
    trainer_class = StableToolSuffixGRPOTrainer
    trainer_extra: dict[str, float] = {}
    if args.credit_mode == "turn_r1":
        from ml.agentic.training.turn_credit_trainer import (
            create_turn_credit_trainer_class,
        )

        trainer_class = create_turn_credit_trainer_class(
            base_trainer_class=StableToolSuffixGRPOTrainer
        )
        trainer_extra = {
            "turn_credit_gamma": args.turn_credit_gamma,
            "turn_credit_blend": args.turn_credit_blend,
        }
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows),
        processing_class=tokenizer,
        quantization_config=trainer_quantization,
        peft_config=trainer_peft,
        environment_factory=environment_factories,
        **trainer_extra,
    )
    train_result = trainer.train()
    # Preserve a completed optimization run before optional evaluation.
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    # Stateful grouped evaluation has a much higher peak-memory footprint than
    # a one-step smoke update.  Smoke checkpoints are evaluated afterwards by
    # the sequential native Agent Loop audit; formal runs retain Trainer eval.
    if args.max_steps > 0:
        eval_metrics = {}
        eval_status = "skipped_for_smoke_external_agent_loop_audit"
    else:
        eval_metrics = trainer.evaluate()
        eval_status = "completed"
    turn_credit_totals = getattr(trainer, "turn_credit_totals", None)
    turn_credit_gate_errors = (
        validate_turn_credit_totals(turn_credit_totals)
        if args.credit_mode == "turn_r1"
        else []
    )
    audit_path = os.environ.get("AGENTIC_GRPO_AUDIT_PATH")
    metadata = {
        "status": "rejected" if turn_credit_gate_errors else "trained",
        "run_scope": "smoke" if args.max_steps > 0 or args.allow_small_corpus else "formal",
        "method": (
            "group-relative-turn-credit-grpo-r1"
            if args.credit_mode == "turn_r1"
            else "trajectory-level-agentic-grpo-b0"
        ),
        "credit_mode": args.credit_mode,
        "execution_mode": args.execution_mode,
        "policy_decision_scope": {
            "policy_driven": "all_dag_actions",
            "controller_first": "legacy_narrow_delegated_actions",
            "react": "production_research_recovery_clarification_tradeoff_actions",
        }[args.execution_mode],
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
        "credit_assignment_claim": (
            "programmatic turn-relative research baseline"
            if args.credit_mode == "turn_r1"
            else "trajectory-level only"
        ),
        "turn_credit_gamma": (
            args.turn_credit_gamma if args.credit_mode == "turn_r1" else None
        ),
        "turn_credit_blend": (
            args.turn_credit_blend if args.credit_mode == "turn_r1" else None
        ),
        "turn_credit_totals": turn_credit_totals,
        "turn_credit_gate_errors": turn_credit_gate_errors,
        "source_model": args.model,
        "tokenizer": tokenizer_source,
        "continued_from_sft_adapter": continued_from_adapter,
        "git_commit": _git_commit(),
        "seed": args.seed,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "beta": args.beta,
        "optimization": {
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "warmup_ratio": args.warmup_ratio,
        },
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "max_tool_calling_iterations": args.max_tool_calling_iterations,
        "max_completion_length": args.max_completion_length,
        "completion_budget": completion_budget.model_dump(mode="json"),
        "reward": RewardConfig().config_version,
        "environment_versions": report.environment_versions,
        "snapshot_versions": report.snapshot_versions,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
        "eval_status": eval_status,
        "rollout_audit_path": audit_path,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if turn_credit_gate_errors:
        raise RuntimeError(
            "turn_r1 training failed its evidence gate: "
            + ", ".join(turn_credit_gate_errors)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
