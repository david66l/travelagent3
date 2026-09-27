"""Real TRL + PEFT QLoRA entrypoint for the Agent Policy model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.training import (  # noqa: E402
    load_jsonl,
    preflight_sft_dataset,
    preflight_sft_model,
    preflight_sft_termination_boundaries,
    require_git_commit,
    select_sft_smoke_rows,
    to_rendered_prompt_completion,
)
from agentic.chat_template_contract import (  # noqa: E402
    AGENT_CHAT_TEMPLATE_KWARGS,
    AGENT_CHAT_TEMPLATE_SHA256,
    AGENT_RENDER_PROTOCOL_VERSION,
    install_agent_chat_template,
    validate_agent_chat_template,
)
from agentic.policy_actions import POLICY_ACTION_MODELS  # noqa: E402
from agentic.reason_quality import repair_reason_rationale_prefixes  # noqa: E402
from agentic.source_equal_sft import (  # noqa: E402
    SOURCE_EQUAL_LOSS_CONTRACT,
    completion_weighted_loss,
    source_equal_weights,
)


LOSS_NORMALIZATION_CONTRACT = "mean-of-microbatch-token-weighted-means.v1"


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


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


def stratify_sft_rows_by_action(rows: list[dict]) -> list[dict]:
    """Round-robin audited SFT rows by policy action.

    With equal action counts, sequential micro-batches and a gradient
    accumulation multiple of the number of actions, every optimizer step sees
    the same action mix.  Fail closed on malformed rows instead of silently
    assigning them to an ``unknown`` bucket.
    """
    groups: dict[str, list[dict]] = {}
    for index, row in enumerate(rows):
        try:
            calls = row["messages"][-1]["tool_calls"]
            action = str(calls[0]["function"]["name"])
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"row {index} has no final policy action") from exc
        if len(calls) != 1 or action not in POLICY_ACTION_MODELS:
            raise ValueError(f"row {index} has an invalid final policy action")
        groups.setdefault(action, []).append(row)
    ordered: list[dict] = []
    actions = sorted(groups)
    cursor = 0
    while len(ordered) < len(rows):
        progressed = False
        for action in actions:
            if cursor < len(groups[action]):
                ordered.append(groups[action][cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    if len(ordered) != len(rows):
        raise ValueError("action stratification dropped SFT rows")
    return ordered


def sequence_match_count(
    token_ids: list[int], sequences: tuple[tuple[int, ...], ...]
) -> int:
    """Count exact token-sequence matches without importing torch."""
    matches = 0
    for sequence in sequences:
        width = len(sequence)
        if not width or width > len(token_ids):
            continue
        matches += sum(
            tuple(token_ids[start : start + width]) == sequence
            for start in range(len(token_ids) - width + 1)
        )
    return matches


def preflight_rationale_sequence_coverage(
    rendered_rows: list[dict[str, str]],
    tokenizer,
    sequences: tuple[tuple[int, ...], ...],
) -> dict[str, int | bool]:
    """Require exactly one audited rationale prefix in every completion."""
    missing = 0
    multiple = 0
    for row in rendered_rows:
        token_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
        count = sequence_match_count(token_ids, sequences)
        if count == 0:
            missing += 1
        elif count > 1:
            multiple += 1
    return {
        "ready": missing == 0 and multiple == 0,
        "rows_checked": len(rendered_rows),
        "matched_once": len(rendered_rows) - missing - multiple,
        "missing": missing,
        "multiple": multiple,
    }


def declare_microbatch_mean_loss(trainer):
    """Tell Transformers that custom CE ignores batch-level loss kwargs.

    Transformers 5 skips its gradient-accumulation division when this flag is
    true.  Our custom loss is already a token-weighted mean *within* each
    microbatch, so the intended effective-batch objective is the arithmetic
    mean of those microbatch means.
    """
    trainer.model_accepts_loss_kwargs = False
    trainer.loss_normalization_contract = LOSS_NORMALIZATION_CONTRACT
    return trainer


def configure_agent_sft_tokenizer(tokenizer):
    """Pin SFT rendering to the same prefix-preserving train/rollout contract."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    install_agent_chat_template(tokenizer)
    validate_agent_chat_template(tokenizer.chat_template)
    return tokenizer


def audit_trainer_completion_masks(trainer, tokenizer, rendered_splits) -> dict:
    """Inspect actual TRL-preprocessed examples and its actual loss collator.

    Fail before training if prompt/tool observations or padding receive labels,
    or if a tool-call target is truncated. This audits both train and dev loss.
    """
    from agentic.local_policy import parse_local_tool_call
    report = {"collator": type(trainer.data_collator).__name__, "splits": {}}
    for split, dataset in (("train", trainer.train_dataset), ("validation", trainer.eval_dataset)):
        counts = {"rows": 0, "masked_prompt_tokens": 0, "supervised_tokens": 0, "max_length": 0}
        if len(dataset) != len(rendered_splits[split]):
            raise RuntimeError("TRL preprocessing changed example coverage")
        for row, rendered in zip(dataset, rendered_splits[split], strict=True):
            prompt_ids = tokenizer.encode(rendered["prompt"], add_special_tokens=False)
            full_ids = tokenizer.encode(rendered["prompt"] + rendered["completion"], add_special_tokens=False)
            if full_ids[:len(prompt_ids)] != prompt_ids or row["input_ids"] != full_ids:
                raise RuntimeError("TRL changed rendered input tokens or completion boundary")
            # TRL may retain completion_mask or materialize labels directly.
            # Derive the expected boundary independently from the source prompt.
            mask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
            batch = trainer.data_collator([row])
            ids = batch["input_ids"][0].tolist()
            labels = batch["labels"][0].tolist()
            attention = batch["attention_mask"][0].tolist()
            for index, (token, label, active) in enumerate(zip(ids, labels, attention, strict=True)):
                expected = token if active and index < len(mask) and mask[index] else -100
                if label != expected:
                    raise RuntimeError(f"{split}: prompt/completion loss-mask mismatch")
            targets = [label for label in labels if label != -100]
            if not targets or targets.count(tokenizer.eos_token_id) != 1:
                raise RuntimeError(f"{split}: EOS target is missing, duplicated, or truncated")
            if tokenizer.decode(targets[targets.index(tokenizer.eos_token_id) + 1:], skip_special_tokens=False).strip():
                raise RuntimeError(f"{split}: substantive target text after EOS")
            parse_local_tool_call(tokenizer.decode(targets, skip_special_tokens=False))
            counts["rows"] += 1
            counts["masked_prompt_tokens"] += len(ids) - len(targets)
            counts["supervised_tokens"] += len(targets)
            counts["max_length"] = max(counts["max_length"], len(ids))
        report["splits"][split] = counts
    report["passed"] = True
    return report


def _adapter_sha256(path: str | Path) -> str | None:
    adapter = Path(path) / "adapter_model.safetensors"
    if not adapter.is_file():
        return None
    digest = hashlib.sha256()
    with adapter.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_snapshot_manifest(path: Path, repo_root: Path) -> dict:
    """Verify a hash-locked execution copy when the remote has no Git metadata."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version")
        not in {"h006-source-snapshot.v1", "internal-sft-source-snapshot.v1"}
        or payload.get("status") != "frozen_internal_run_input"
        or not re.fullmatch(
            r"[0-9a-f]{40}", str(payload.get("origin_git_commit") or "")
        )
        or payload.get("promotion_eligible") is not False
        or payload.get("online_replacement_authorized") is not False
    ):
        raise ValueError("invalid source snapshot contract")
    root = repo_root.resolve()
    code_files = payload.get("code_files")
    if not isinstance(code_files, dict) or not code_files:
        raise ValueError("source snapshot has no code files")
    for relative, expected in code_files.items():
        target = (root / relative).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise ValueError("source snapshot path escapes or is missing")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"source snapshot hash mismatch: {relative}")
    dataset = Path(str(payload.get("dataset_dir") or "")).resolve()
    dataset_files = payload.get("dataset_files")
    if not isinstance(dataset_files, dict) or not dataset_files:
        raise ValueError("source snapshot has no dataset files")
    for relative, expected in dataset_files.items():
        target = (dataset / relative).resolve()
        if not target.is_relative_to(dataset) or not target.is_file():
            raise ValueError("dataset snapshot path escapes or is missing")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"dataset snapshot hash mismatch: {relative}")
    return {
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "origin_git_commit": payload["origin_git_commit"],
        "code_files": len(code_files),
        "dataset_files": len(dataset_files),
    }


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
    parser.add_argument(
        "--optim",
        default="adamw_torch",
        choices=("adamw_torch", "paged_adamw_8bit"),
        help="Explicit optimizer implementation recorded in the run manifest.",
    )
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
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("linear", "cosine"),
        default="linear",
    )
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
        "--stop-after-steps",
        type=int,
        default=0,
        help=(
            "Pause a pre-registered max-steps run after this many optimizer steps. "
            "Unlike lowering --max-steps, this preserves the full-run LR schedule "
            "and produces a resumable intermediate checkpoint."
        ),
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
        "--external-test-evaluation",
        action="store_true",
        help=(
            "Allow an empty local test.jsonl for an internal-only run whose frozen "
            "holdout is evaluated outside Trainer. Never grants promotion eligibility."
        ),
    )
    parser.add_argument("--source-equal-loss", action="store_true")
    parser.add_argument("--source-lineage", type=Path)
    parser.add_argument("--source-snapshot-manifest", type=Path)
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="Mark the run as ineligible for promotion pending Agent Loop evaluation.",
    )
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
    parser.add_argument(
        "--rationale-token-weight",
        type=float,
        default=1.0,
        help=(
            "Upweight only the audited action-rationale prefix inside reason. "
            "Evidence, action names, tool envelope, and EOS remain unchanged."
        ),
    )
    parser.add_argument(
        "--stratified-action-batches",
        action="store_true",
        help=(
            "Use a deterministic round-robin action order and sequential sampler. "
            "For a balanced three-action dataset and gradient accumulation 12, "
            "each optimizer step sees four examples per action."
        ),
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Resume optimizer/scheduler/RNG state from a Trainer checkpoint.",
    )
    parser.add_argument(
        "--allow-unknown-git-commit",
        action="store_true",
        help="Escape hatch for non-repo smoke sandboxes only; formal training runs "
        "must stay reproducible from a git commit.",
    )
    args = parser.parse_args()
    if args.source_equal_loss and (
        args.batch_size != 1
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
        or args.max_train_examples
        or args.max_eval_examples
        or not args.source_lineage
        or not args.source_lineage.is_file()
    ):
        parser.error(
            "source-equal v1 requires batch size 1, one process, complete splits and --source-lineage"
        )
    if args.source_snapshot_manifest:
        try:
            source_snapshot = validate_source_snapshot_manifest(
                args.source_snapshot_manifest, REPO_ROOT
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        git_commit = f"snapshot:{source_snapshot['manifest_sha256']}"
    else:
        source_snapshot = None
        git_commit = require_git_commit(
            parser, REPO_ROOT, allow_unknown=args.allow_unknown_git_commit
        )
    if args.termination_token_weight < 1.0:
        parser.error("--termination-token-weight must be at least 1.0")
    if args.action_token_weight < 1.0:
        parser.error("--action-token-weight must be at least 1.0")
    if not 1.0 <= args.rationale_token_weight <= 4.0:
        parser.error("--rationale-token-weight must be in [1.0, 4.0]")
    for name in ("logging_steps", "eval_steps", "save_steps", "save_total_limit"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if not 0.0 <= args.warmup_ratio < 1.0:
        parser.error("--warmup-ratio must be in [0, 1)")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm must be positive")
    if args.weight_decay < 0:
        parser.error("--weight-decay must be non-negative")
    if args.resume_from_checkpoint and not args.resume_from_checkpoint.is_dir():
        parser.error("--resume-from-checkpoint must be an existing directory")
    if args.stop_after_steps < 0:
        parser.error("--stop-after-steps must be non-negative")
    if args.stop_after_steps and (
        args.max_steps <= 0 or args.stop_after_steps >= args.max_steps
    ):
        parser.error("--stop-after-steps must be smaller than positive --max-steps")

    minimum = 1 if args.allow_small_dataset else args.minimum_train_examples
    report = preflight_sft_dataset(
        args.dataset_dir,
        minimum_train_examples=minimum,
        require_dependencies=not args.preflight_only,
        require_test_split=not args.external_test_evaluation,
    )
    print(report.model_dump_json(indent=2))
    from transformers import AutoTokenizer

    tokenizer_source = args.tokenizer or args.model
    source_adapter_sha256_before = _adapter_sha256(args.model)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=False)
    configure_agent_sft_tokenizer(tokenizer)
    model_preflight = preflight_sft_model(
        args.dataset_dir,
        tokenizer,
        max_length=args.max_length,
    )
    print(model_preflight.model_dump_json(indent=2))
    boundary_preflight = preflight_sft_termination_boundaries(
        args.dataset_dir, tokenizer
    )
    print(boundary_preflight.model_dump_json(indent=2))
    data_errors = [error for error in report.errors if "DEPENDENCIES" not in error]
    source_weight_reports = {}
    if args.source_equal_loss:
        source_lineage = load_jsonl(args.source_lineage)
        for name, split in (("train", "train"), ("validation", "train-shadow")):
            _, source_weight_reports[name] = source_equal_weights(
                load_jsonl(args.dataset_dir / f"{name}.jsonl"),
                source_lineage,
                split=split,
            )
        source_weight_reports["lineage_sha256"] = hashlib.sha256(
            args.source_lineage.read_bytes()
        ).hexdigest()
        print(json.dumps({"source_weighting": source_weight_reports}, indent=2))
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
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
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
        rationale_token_sequences: tuple[tuple[int, ...], ...] = ()
        rationale_token_weight: float = 1.0
        source_equal_loss: bool = False

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
            apply_action_sequence_weights(
                labels,
                weights,
                self.rationale_token_sequences,
                self.rationale_token_weight,
            )
            batch["loss_weights"] = weights
            if self.source_equal_loss:
                batch["source_weights"] = torch.tensor(
                    [item["source_weight"] for item in examples],
                    dtype=torch.float32,
                )
            return batch

    class BoundaryWeightedSFTTrainer(SFTTrainer):
        """Token-normalized causal CE with extra credit on the tool-call boundary."""

        def __init__(self, *trainer_args, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            # This trainer normalizes by its own token-weight denominator and
            # intentionally ignores ``num_items_in_batch``.  Transformers 5
            # otherwise skips gradient-accumulation normalization, inflating
            # gradients and reported train loss by exactly grad_acc steps.
            declare_microbatch_mean_loss(self)

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
            source_weights = inputs.pop("source_weights", None)
            inputs["use_cache"] = False
            outputs = model(**inputs)
            loss = completion_weighted_loss(
                outputs.logits, labels, weights, source_weights
            )
            return (loss, outputs) if return_outputs else loss

    class StratifiedSFTTrainer(SFTTrainer):
        """Keep the pre-stratified row order deterministic across training."""

        def _get_train_sampler(self, train_dataset=None):
            from torch.utils.data import SequentialSampler

            return SequentialSampler(
                train_dataset if train_dataset is not None else self.train_dataset
            )

    class StratifiedBoundaryWeightedSFTTrainer(BoundaryWeightedSFTTrainer):
        """Combine token weights with the deterministic action sampler."""

        def _get_train_sampler(self, train_dataset=None):
            from torch.utils.data import SequentialSampler

            return SequentialSampler(
                train_dataset if train_dataset is not None else self.train_dataset
            )

    class StopAfterStepsCallback(TrainerCallback):
        """Pause without shortening the scheduler's pre-registered horizon."""

        def __init__(self, stop_after_steps: int):
            self.stop_after_steps = stop_after_steps

        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            if state.global_step >= self.stop_after_steps:
                control.should_save = True
                control.should_training_stop = True
            return control

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
    if args.stratified_action_batches:
        raw_train_rows = stratify_sft_rows_by_action(raw_train_rows)
    raw_validation_rows = select_sft_smoke_rows(
        load_jsonl(args.dataset_dir / "validation.jsonl"), args.max_eval_examples
    )
    train_rows = to_rendered_prompt_completion(
        raw_train_rows,
        tokenizer,
        chat_template_kwargs=AGENT_CHAT_TEMPLATE_KWARGS,
    )
    rationale_token_sequences = tuple(
        tuple(tokenizer.encode(prefix, add_special_tokens=False))
        for action in ("abort", "propose_tradeoff")
        for prefix in repair_reason_rationale_prefixes(action)
    )
    rationale_weight_preflight = preflight_rationale_sequence_coverage(
        train_rows,
        tokenizer,
        rationale_token_sequences,
    )
    if args.rationale_token_weight > 1.0 and not rationale_weight_preflight["ready"]:
        raise RuntimeError(
            "rationale token weighting requires exactly one prefix match per train row"
        )
    validation_rows = to_rendered_prompt_completion(
        raw_validation_rows,
        tokenizer,
        chat_template_kwargs=AGENT_CHAT_TEMPLATE_KWARGS,
    )
    # The canonical chat template permits whitespace after EOS. TRL's string
    # path appends EOS unless the string ends with it, which otherwise creates
    # a second supervised EOS. Local generation stops at the first EOS.
    for rendered in [*train_rows, *validation_rows]:
        rendered["completion"] = rendered["completion"].rstrip()
        if not rendered["completion"].endswith(tokenizer.eos_token):
            raise RuntimeError("Rendered tool completion does not end with EOS")
    if args.source_equal_loss:
        for raw, rendered, split in (
            (raw_train_rows, train_rows, "train"),
            (raw_validation_rows, validation_rows, "train-shadow"),
        ):
            weights, _ = source_equal_weights(raw, source_lineage, split=split)
            for original, row, weight in zip(raw, rendered, weights, strict=True):
                row["source_weight"] = weight
                row["source_record_id"] = original["example_id"]
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(validation_rows)
    report_to = ["mlflow"] if os.environ.get("MLFLOW_TRACKING_URI") else []
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        optim=args.optim,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        remove_unused_columns=not args.source_equal_loss,
        logging_steps=args.logging_steps,
        eval_strategy="steps"
        if args.eval_during_smoke or args.max_steps <= 0
        else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        seed=args.seed,
        report_to=report_to,
        run_name=f"agent-policy-sft-{report.dataset_version}",
    )
    weighted_sft = (
        args.source_equal_loss
        or args.termination_token_weight > 1.0
        or args.action_token_weight > 1.0
        or args.rationale_token_weight > 1.0
    )
    if args.stratified_action_batches:
        trainer_class = (
            StratifiedBoundaryWeightedSFTTrainer
            if weighted_sft
            else StratifiedSFTTrainer
        )
    else:
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
            rationale_token_sequences=rationale_token_sequences,
            rationale_token_weight=args.rationale_token_weight,
            source_equal_loss=args.source_equal_loss,
        )
    callbacks = (
        [StopAfterStepsCallback(args.stop_after_steps)]
        if args.stop_after_steps
        else None
    )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    if args.source_equal_loss:
        for dataset, expected in (
            (trainer.train_dataset, train_rows),
            (trainer.eval_dataset, validation_rows),
        ):
            expected_weights = {
                row["source_record_id"]: row["source_weight"] for row in expected
            }
            actual_weights = dict(
                zip(dataset["source_record_id"], dataset["source_weight"], strict=True)
            )
            if len(dataset) != len(expected) or actual_weights != expected_weights:
                raise RuntimeError(
                    "TRL preprocessing changed source-equal dataset coverage"
                )
        trainer.loss_normalization_contract = SOURCE_EQUAL_LOSS_CONTRACT
    trainable_parameters = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    # PEFT expands packed Params4bit storage to logical parameter counts.
    # Summing .numel() directly undercounts the quantized base model.
    trainable_parameter_count, total_parameter_count = trainer.model.get_nb_trainable_parameters()
    mask_audit = audit_trainer_completion_masks(trainer, tokenizer, {"train": train_rows, "validation": validation_rows})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "completion-mask-audit.json").write_text(json.dumps(mask_audit, indent=2), encoding="utf-8")
    probe_name, probe = next((name, p) for name, p in trainer.model.named_parameters() if p.requires_grad and "lora_B" in name)
    probe_before = probe.detach().float().cpu().clone()
    train_result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        )
    )
    eval_metrics = trainer.evaluate()
    probe_after = probe.detach().float().cpu()
    update_audit = {"parameter": probe_name, "l2_delta": float((probe_after - probe_before).norm()), "finite": bool(torch.isfinite(probe_after).all())}
    if not update_audit["finite"] or update_audit["l2_delta"] <= 0:
        raise RuntimeError("SFT did not produce a finite LoRA parameter update")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    source_adapter_sha256_after = _adapter_sha256(args.model)
    if source_adapter_sha256_before != source_adapter_sha256_after:
        raise RuntimeError("source adapter changed during SFT")
    metadata = {
        "status": "trained",
        "candidate_status": "quarantine_pending_agent_loop_eval",
        "promotion_eligible": False,
        "external_test_evaluation": args.external_test_evaluation,
        "quarantine_requested": args.quarantine,
        "run_scope": "smoke"
        if args.max_steps > 0 or args.allow_small_dataset
        else "formal",
        "base_model": args.model,
        "tokenizer": tokenizer_source,
        "continued_from_adapter": continued_from_adapter,
        "adapter_config": (
            peft_config.to_dict() if continued_from_adapter else lora.to_dict()
        ),
        "trainable_parameters": {
            "names": trainable_parameters,
            "trainable_count": trainable_parameter_count,
            "total_count": total_parameter_count,
            "trainable_fraction": trainable_parameter_count / total_parameter_count,
        },
        "dataset_version": report.dataset_version,
        "git_commit": git_commit,
        "source_snapshot": source_snapshot,
        "seed": args.seed,
        "quantization": "nf4-double-quant",
        "model_preflight": model_preflight.model_dump(mode="json"),
        "completion_mask_audit": mask_audit,
        "parameter_update_audit": update_audit,
        "termination_boundary_preflight": boundary_preflight.model_dump(mode="json"),
        "termination_token_weight": args.termination_token_weight,
        "action_token_weight": args.action_token_weight,
        "rationale_token_weight": args.rationale_token_weight,
        "rationale_weight_preflight": rationale_weight_preflight,
        "loss_normalization_contract": (
            SOURCE_EQUAL_LOSS_CONTRACT
            if args.source_equal_loss
            else LOSS_NORMALIZATION_CONTRACT
        ),
        "source_weighting": source_weight_reports,
        "stratified_action_batches": args.stratified_action_batches,
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
        ),
        "stop_after_steps": args.stop_after_steps,
        "dataset_transport": "rendered-scalar-prompt-completion.v1",
        "source_adapter_sha256_before": source_adapter_sha256_before,
        "source_adapter_sha256_after": source_adapter_sha256_after,
        "output_adapter_sha256": _adapter_sha256(args.output_dir),
        "training_contract": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "optimizer": args.optim,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "seed": args.seed,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": args.lr_scheduler_type,
            "max_steps": args.max_steps,
            "stop_after_steps": args.stop_after_steps,
            "maximum_train_examples": args.max_train_examples,
            "maximum_eval_examples": args.max_eval_examples,
            "completion_only_loss": True,
            "packing": False,
        },
        "runtime_versions": {
            name: _package_version(name)
            for name in ("transformers", "trl", "peft", "torch", "bitsandbytes")
        },
        "render_protocol": AGENT_RENDER_PROTOCOL_VERSION,
        "chat_template_sha256": AGENT_CHAT_TEMPLATE_SHA256,
        "chat_template_kwargs": AGENT_CHAT_TEMPLATE_KWARGS,
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
