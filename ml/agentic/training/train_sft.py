"""Real TRL + PEFT QLoRA entrypoint for the Agent Policy model."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.training import (  # noqa: E402
    load_jsonl,
    preflight_sft_model,
    preflight_sft_dataset,
    preflight_sft_termination_boundaries,
    select_sft_smoke_rows,
    to_conversational_prompt_completion,
)
from agentic.policy_actions import POLICY_ACTION_MODELS  # noqa: E402


def apply_action_sequence_weights(
    labels,
    weights,
    action_token_sequences: tuple[tuple[int, ...], ...],
    action_token_weight: float,
) -> None:
    """Upweight completion action-name tokens without touching prompt labels."""
    if action_token_weight <= 1.0:
        return
    import torch

    for sequence in action_token_sequences:
        if not sequence or len(sequence) > labels.shape[-1]:
            continue
        target = torch.tensor(sequence, dtype=labels.dtype, device=labels.device)
        matches = labels.unfold(-1, len(sequence), 1).eq(target).all(dim=-1)
        for batch_index, start in matches.nonzero(as_tuple=False).tolist():
            stop = start + len(sequence)
            weights[batch_index, start:stop] = torch.maximum(
                weights[batch_index, start:stop],
                torch.full_like(
                    weights[batch_index, start:stop],
                    action_token_weight,
                ),
            )


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
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--tokenizer",
        help=(
            "Optional tokenizer path/name. Use the base model tokenizer when "
            "continuing an adapter whose archived tokenizer metadata is stale."
        ),
    )
    parser.add_argument("--minimum-train-examples", type=int, default=3000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--eval-during-smoke",
        action="store_true",
        help="Evaluate loss during bounded runs; checkpoint promotion still requires Agent Loop eval.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values bound smoke/debug runs; formal training keeps -1.",
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=0,
        help="Optional deterministic prefix used only to keep smoke runs small.",
    )
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=0,
        help="Optional deterministic validation prefix used only for smoke runs.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-small-dataset",
        action="store_true",
        help="Smoke test only; never use this flag for a reported training run.",
    )
    parser.add_argument(
        "--termination-token-weight",
        type=float,
        default=1.0,
        help=(
            "Upweight the single EOS immediately after </tool_call>. Values above 1 "
            "enable boundary-weighted SFT."
        ),
    )
    parser.add_argument(
        "--action-token-weight",
        type=float,
        default=1.0,
        help=(
            "Upweight completion tokens that spell a policy action name. This "
            "prevents long reason/options arguments from dominating boundary SFT."
        ),
    )
    args = parser.parse_args()
    if args.termination_token_weight < 1.0:
        parser.error("--termination-token-weight must be at least 1.0")
    if args.action_token_weight < 1.0:
        parser.error("--action-token-weight must be at least 1.0")
    for name in ("logging_steps", "eval_steps", "save_steps", "save_total_limit"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must be in [0, 1)")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")

    minimum = 1 if args.allow_small_dataset else args.minimum_train_examples
    report = preflight_sft_dataset(
        args.dataset_dir,
        minimum_train_examples=minimum,
        require_dependencies=not args.preflight_only,
    )
    print(report.model_dump_json(indent=2))
    from transformers import AutoTokenizer

    tokenizer_source = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_preflight = preflight_sft_model(
        args.dataset_dir,
        tokenizer,
        max_length=args.max_length,
    )
    print(model_preflight.model_dump_json(indent=2))
    boundary_preflight = preflight_sft_termination_boundaries(args.dataset_dir, tokenizer)
    print(boundary_preflight.model_dump_json(indent=2))
    data_errors = [error for error in report.errors if "DEPENDENCIES" not in error]
    if args.preflight_only:
        return (
            0
            if not data_errors and model_preflight.ready and boundary_preflight.ready
            else 2
        )
    if not report.ready or not model_preflight.ready or not boundary_preflight.ready:
        return 2

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA training requires a CUDA GPU")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    @dataclass
    class BoundaryWeightedDataCollator(DataCollatorForLanguageModeling):
        """Attach loss weights without changing TRL's completion-only labels."""

        termination_token_id: int = -1
        termination_token_weight: float = 1.0
        action_token_sequences: tuple[tuple[int, ...], ...] = ()
        action_token_weight: float = 1.0

        def torch_call(self, examples):
            batch = super().torch_call(examples)
            labels = batch["labels"]
            weights = torch.ones_like(labels, dtype=torch.float32)
            weights.masked_fill_(labels == -100, 0.0)
            weights.masked_fill_(
                labels == self.termination_token_id,
                self.termination_token_weight,
            )
            apply_action_sequence_weights(
                labels,
                weights,
                self.action_token_sequences,
                self.action_token_weight,
            )
            batch["loss_weights"] = weights
            return batch

    class BoundaryWeightedSFTTrainer(SFTTrainer):
        """Token-normalized causal CE with extra credit on the tool-call boundary."""

        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            del num_items_in_batch
            labels = inputs.pop("labels")
            weights = inputs.pop("loss_weights")
            inputs["use_cache"] = False
            outputs = model(**inputs)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = weights[..., 1:].to(shift_logits.device)
            per_token_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
                ignore_index=-100,
            ).view_as(shift_labels)
            denominator = shift_weights.sum().clamp_min(1.0)
            loss = (per_token_loss * shift_weights).sum() / denominator
            return (loss, outputs) if return_outputs else loss

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    adapter_config = Path(args.model) / "adapter_config.json"
    if adapter_config.is_file():
        # Load the base and adapter explicitly. AutoPeftModel also auto-loads the
        # tokenizer bundled with an adapter; archived checkpoints may contain
        # tokenizer metadata from an older Transformers release even when the
        # caller supplied a compatible base tokenizer via --tokenizer.
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
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        lora = None
        continued_from_adapter = True
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization,
            device_map="auto",
            dtype=compute_dtype,
            trust_remote_code=False,
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        continued_from_adapter = False

    raw_train_rows = select_sft_smoke_rows(
        load_jsonl(args.dataset_dir / "train.jsonl"), args.max_train_examples
    )
    raw_validation_rows = select_sft_smoke_rows(
        load_jsonl(args.dataset_dir / "validation.jsonl"), args.max_eval_examples
    )
    train_rows = to_conversational_prompt_completion(raw_train_rows)
    validation_rows = to_conversational_prompt_completion(raw_validation_rows)
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(validation_rows)
    report_to = ["mlflow"] if os.environ.get("MLFLOW_TRACKING_URI") else []
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if args.eval_during_smoke or args.max_steps <= 0 else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        seed=args.seed,
        report_to=report_to,
        run_name=f"agent-policy-sft-{report.dataset_version}",
    )
    weighted_sft = (
        args.termination_token_weight > 1.0 or args.action_token_weight > 1.0
    )
    trainer_class = BoundaryWeightedSFTTrainer if weighted_sft else SFTTrainer
    data_collator = None
    if weighted_sft:
        action_token_sequences = tuple(
            tuple(tokenizer.encode(action, add_special_tokens=False))
            for action in sorted(POLICY_ACTION_MODELS)
        )
        data_collator = BoundaryWeightedDataCollator(
            pad_token_id=tokenizer.pad_token_id,
            termination_token_id=tokenizer.eos_token_id,
            termination_token_weight=args.termination_token_weight,
            action_token_sequences=action_token_sequences,
            action_token_weight=args.action_token_weight,
        )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora,
        data_collator=data_collator,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {
        "status": "trained",
        "run_scope": "smoke" if args.max_steps > 0 or args.allow_small_dataset else "formal",
        "base_model": args.model,
        "tokenizer": tokenizer_source,
        "continued_from_adapter": continued_from_adapter,
        "dataset_version": report.dataset_version,
        "git_commit": _git_commit(),
        "seed": args.seed,
        "quantization": "nf4-double-quant",
        "model_preflight": model_preflight.model_dump(mode="json"),
        "termination_boundary_preflight": boundary_preflight.model_dump(mode="json"),
        "termination_token_weight": args.termination_token_weight,
        "action_token_weight": args.action_token_weight,
        "checkpoint_cadence": {
            "logging_steps": args.logging_steps,
            "eval_steps": args.eval_steps,
            "save_steps": args.save_steps,
            "save_total_limit": args.save_total_limit,
            "eval_during_smoke": args.eval_during_smoke,
        },
        "optimization_safety": {
            "warmup_ratio": args.warmup_ratio,
            "max_grad_norm": args.max_grad_norm,
        },
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
