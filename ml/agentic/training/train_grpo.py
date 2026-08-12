"""Real TRL Agentic GRPO-B0 entrypoint for snapshot travel environments."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import (  # noqa: E402
    load_grpo_corpus,
    preflight_grpo_corpus,
    to_trl_environment_rows,
)
from agentic.trl_environment import TRLTravelEnvironment  # noqa: E402


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model", required=True, help="SFT checkpoint or merged policy model"
    )
    parser.add_argument("--minimum-train-tasks", type=int, default=1000)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-tool-calling-iterations", type=int, default=16)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
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
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("Agentic GRPO training requires a CUDA GPU")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.chat_template:
        raise RuntimeError(
            "policy checkpoint must provide a native tool-capable chat_template"
        )

    train_rows = to_trl_environment_rows(
        load_grpo_corpus(args.corpus_dir / "train.jsonl")
    )
    validation_rows = to_trl_environment_rows(
        load_grpo_corpus(args.corpus_dir / "validation.jsonl")
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
    report_to = ["mlflow"] if os.environ.get("MLFLOW_TRACKING_URI") else []
    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        num_generations=args.num_generations,
        num_generations_eval=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=args.max_tool_calling_iterations,
        temperature=0.8,
        beta=0.04,
        loss_type="grpo",
        scale_rewards="group",
        mask_truncated_completions=True,
        chat_template_kwargs={"enable_thinking": False},
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_steps=5,
        seed=args.seed,
        report_to=report_to,
        run_name=f"agent-policy-grpo-b0-{Path(args.model).name}",
        model_init_kwargs={"dtype": compute_dtype, "trust_remote_code": False},
    )
    trainer = GRPOTrainer(
        model=args.model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(validation_rows),
        processing_class=tokenizer,
        quantization_config=quantization,
        peft_config=lora,
        environment_factory=TRLTravelEnvironment,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {
        "status": "trained",
        "method": "trajectory-level-agentic-grpo-b0",
        "credit_assignment_claim": "trajectory-level only",
        "source_model": args.model,
        "git_commit": _git_commit(),
        "seed": args.seed,
        "num_generations": args.num_generations,
        "reward": "hierarchical-b0.v1",
        "environment_versions": report.environment_versions,
        "snapshot_versions": report.snapshot_versions,
        "train_metrics": train_result.metrics,
        "eval_metrics": eval_metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_report.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
