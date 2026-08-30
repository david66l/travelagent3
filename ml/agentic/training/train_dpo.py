"""Train a QLoRA travel-policy adapter with verified chosen/rejected pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

CONTRACT_EVIDENCE_POLICY = "verifier_success_or_deterministic_single_action_contract"
DECISION_BOUNDARY_EVIDENCE_POLICY = (
    "verifier_success_or_deterministic_decision_boundary_contract"
)


def _is_deterministic_single_action_pair(row: dict[str, Any]) -> bool:
    if "SINGLE_ACTION_CONTRACT_OVER_DUPLICATE_CALL" not in row.get("reason_codes", []):
        return False
    chosen_calls = (row.get("chosen") or {}).get("tool_calls") or []
    rejected_calls = (row.get("rejected") or {}).get("tool_calls") or []
    return (
        len(chosen_calls) == 1
        and len(rejected_calls) == 2
        and rejected_calls[0] == chosen_calls[0]
        and rejected_calls[1] == chosen_calls[0]
    )


def _call_name(response: dict[str, Any]) -> str | None:
    calls = response.get("tool_calls") or []
    if len(calls) != 1:
        return None
    return str((calls[0].get("function") or {}).get("name") or "") or None


def _is_deterministic_decision_boundary_pair(row: dict[str, Any]) -> bool:
    if "DECISION_BOUNDARY_CONTRACT_OVER_OPPOSITE_ACTION" not in row.get(
        "reason_codes", []
    ):
        return False
    user_messages = [
        message
        for message in row.get("messages") or []
        if message.get("role") == "user" and message.get("content")
    ]
    if len(user_messages) != 1:
        return False
    try:
        context = json.loads(user_messages[0]["content"])
    except (TypeError, json.JSONDecodeError):
        return False
    actionable = (context.get("capability") or {}).get("actionable_alternatives")
    expected = "propose_tradeoff" if actionable is True else (
        "abort" if actionable is False else None
    )
    opposite = "abort" if expected == "propose_tradeoff" else (
        "propose_tradeoff" if expected == "abort" else None
    )
    return bool(
        expected
        and _call_name(row.get("chosen") or {}) == expected
        and _call_name(row.get("rejected") or {}) == opposite
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"preference split is empty: {path}")
    return rows


def validate_preference_dataset(
    dataset_dir: Path, minimum_train_examples: int
) -> dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("preference manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("status") != "passed":
        errors.append("MANIFEST_NOT_PASSED")
    evidence_policy = manifest.get("preference_evidence_policy")
    accepts_contract_evidence = evidence_policy in {
        CONTRACT_EVIDENCE_POLICY,
        DECISION_BOUNDARY_EVIDENCE_POLICY,
    }
    if (
        not manifest.get("requires_verifier_success_over_failure")
        and not accepts_contract_evidence
    ):
        errors.append("VERIFIER_SUCCESS_CONTRACT_MISSING")

    splits = {
        split: load_jsonl(dataset_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    if len(splits["train"]) < minimum_train_examples:
        errors.append(
            f"TRAIN_TOO_SMALL:{len(splits['train'])}<{minimum_train_examples}"
        )
    seen_pairs: set[str] = set()
    split_contexts: dict[str, set[str]] = {}
    family_counts: Counter[str] = Counter()
    for split, rows in splits.items():
        split_contexts[split] = set()
        for row in rows:
            pair_id = str(row.get("pair_id") or "")
            if not pair_id or pair_id in seen_pairs:
                errors.append(f"INVALID_OR_DUPLICATE_PAIR:{pair_id}")
            seen_pairs.add(pair_id)
            context_hash = str(row.get("context_hash") or "")
            if not context_hash:
                errors.append(f"CONTEXT_HASH_MISSING:{pair_id}")
            split_contexts[split].add(context_hash)
            family_counts[str(row.get("family") or "unknown")] += 1
            has_verifier_evidence = "VERIFIER_SUCCESS_OVER_FAILURE" in row.get(
                "reason_codes", []
            )
            has_contract_evidence = accepts_contract_evidence and (
                _is_deterministic_single_action_pair(row)
                or _is_deterministic_decision_boundary_pair(row)
            )
            if not has_verifier_evidence and not has_contract_evidence:
                errors.append(f"UNVERIFIED_PAIR:{pair_id}")
            if not isinstance(row.get("messages"), list) or len(row["messages"]) < 2:
                errors.append(f"PROMPT_MESSAGES_INVALID:{pair_id}")
            if not isinstance(row.get("tools"), list) or not row["tools"]:
                errors.append(f"TOOLS_INVALID:{pair_id}")
            for key in ("chosen", "rejected"):
                response = row.get(key)
                if (
                    not isinstance(response, dict)
                    or response.get("role") != "assistant"
                ):
                    errors.append(f"{key.upper()}_INVALID:{pair_id}")
            if row.get("chosen") == row.get("rejected"):
                errors.append(f"IDENTICAL_RESPONSES:{pair_id}")
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if split_contexts[left] & split_contexts[right]:
            errors.append(f"CONTEXT_SPLIT_OVERLAP:{left}:{right}")
    version = (
        manifest.get("dataset_version")
        or "preference-" + hashlib.sha256(manifest_path.read_bytes()).hexdigest()[:16]
    )
    return {
        "ready": not errors,
        "dataset_version": version,
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "family_counts": dict(family_counts),
        "unique_pairs": len(seen_pairs),
        "errors": errors,
    }


def select_stratified_rows(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[str(row["family"])].append(row)
    for items in families.values():
        items.sort(
            key=lambda row: hashlib.sha256(str(row["pair_id"]).encode()).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    family_names = sorted(families)
    index = 0
    while len(selected) < limit:
        progressed = False
        for family in family_names:
            items = families[family]
            if index < len(items) and len(selected) < limit:
                selected.append(items[index])
                progressed = True
        if not progressed:
            break
        index += 1
    return selected


def to_dpo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "prompt": row["messages"],
            "chosen": [row["chosen"]],
            "rejected": [row["rejected"]],
            "tools": row["tools"],
            "pair_id": row["pair_id"],
            "family": row["family"],
        }
        for row in rows
    ]


def _token_ids(value: Any) -> list[int]:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def preflight_model(
    rows: list[dict[str, Any]], tokenizer: Any, max_length: int
) -> dict[str, Any]:
    lengths: list[int] = []
    prefix_errors: list[str] = []
    for row in rows:
        kwargs = {"tools": row["tools"], "enable_thinking": False}
        prompt_ids = _token_ids(
            tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True, **kwargs
            )
        )
        for key in ("chosen", "rejected"):
            full_ids = _token_ids(
                tokenizer.apply_chat_template(
                    [*row["messages"], row[key]],
                    tokenize=True,
                    add_generation_prompt=False,
                    **kwargs,
                )
            )
            if full_ids[: len(prompt_ids)] != prompt_ids:
                prefix_errors.append(f"{row['pair_id']}:{key}")
            lengths.append(len(full_ids))
    ordered = sorted(lengths)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "ready": not prefix_errors and max(ordered) <= max_length,
        "pairs_checked": len(rows),
        "sequences_checked": len(lengths),
        "max_sequence_tokens": max(ordered),
        "p95_sequence_tokens": p95,
        "over_max_length": sum(length > max_length for length in lengths),
        "prefix_errors": prefix_errors,
    }


def install_frozen_sft_reference_adapter(model: Any, source_adapter: Path) -> str:
    """Install an explicit frozen SFT reference for PEFT-aware DPO.

    TRL treats ``ref_model=None`` plus a PEFT policy without a ``ref`` adapter as
    the base model with all adapters disabled.  DPO must instead compare against
    the exact SFT policy that initialized training.
    """

    if "ref" in model.peft_config:
        raise ValueError("reserved DPO reference adapter name already exists: ref")
    active_adapter = model.active_adapter
    model.load_adapter(str(source_adapter), adapter_name="ref", is_trainable=False)
    if "ref" not in model.peft_config:
        raise RuntimeError("failed to install frozen SFT reference adapter")
    model.set_adapter(active_adapter)
    return "frozen-sft-adapter:ref"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Balanced SFT PEFT checkpoint")
    parser.add_argument("--minimum-train-examples", type=int, default=600)
    parser.add_argument("--max-length", type=int, default=1152)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-small-dataset", action="store_true")
    args = parser.parse_args()
    if args.beta <= 0:
        raise ValueError("DPO beta must be positive")
    if not (Path(args.model) / "adapter_config.json").is_file():
        raise ValueError("DPO must continue from an audited SFT PEFT adapter")

    minimum = 1 if args.allow_small_dataset else args.minimum_train_examples
    dataset_report = validate_preference_dataset(args.dataset_dir, minimum)
    print(json.dumps(dataset_report, ensure_ascii=False, indent=2))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = select_stratified_rows(
        load_jsonl(args.dataset_dir / "train.jsonl"), args.max_train_examples
    )
    validation_rows = select_stratified_rows(
        load_jsonl(args.dataset_dir / "validation.jsonl"), args.max_eval_examples
    )
    model_report = preflight_model(
        [*train_rows, *validation_rows], tokenizer, args.max_length
    )
    print(json.dumps(model_report, ensure_ascii=False, indent=2))
    if args.preflight_only:
        return 0 if dataset_report["ready"] and model_report["ready"] else 2
    if not dataset_report["ready"] or not model_report["ready"]:
        return 2

    import torch
    from datasets import Dataset
    from peft import AutoPeftModelForCausalLM
    from transformers import BitsAndBytesConfig
    from trl import DPOConfig, DPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("DPO QLoRA training requires a CUDA GPU")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.model,
        is_trainable=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=compute_dtype,
        trust_remote_code=False,
    )
    reference_policy = install_frozen_sft_reference_adapter(model, Path(args.model))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    report_to = ["mlflow"] if os.environ.get("MLFLOW_TRACKING_URI") else []
    config = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        beta=args.beta,
        loss_type=["sigmoid"],
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        max_length=args.max_length,
        truncation_mode="keep_start",
        logging_steps=5,
        eval_strategy="no" if args.max_steps > 0 else "steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        seed=args.seed,
        report_to=report_to,
        run_name=f"agent-policy-dpo-{dataset_report['dataset_version']}",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=Dataset.from_list(to_dpo_rows(train_rows)),
        eval_dataset=Dataset.from_list(to_dpo_rows(validation_rows)),
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    eval_metrics = trainer.evaluate()
    metadata = {
        "status": "trained",
        "run_scope": "smoke"
        if args.max_steps > 0 or args.allow_small_dataset
        else "formal",
        "method": "direct-preference-optimization",
        "source_model": args.model,
        "reference_policy": reference_policy,
        "dataset_version": dataset_report["dataset_version"],
        "git_commit": _git_commit(),
        "seed": args.seed,
        "beta": args.beta,
        "loss_type": "sigmoid",
        "quantization": "nf4-double-quant",
        "train_examples": len(train_rows),
        "eval_examples": len(validation_rows),
        "dataset_preflight": dataset_report,
        "model_preflight": model_report,
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
