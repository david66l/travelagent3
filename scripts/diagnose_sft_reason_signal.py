"""Measure where verifier-repair SFT loss and gradient signal are spent.

This is a zero-optimizer diagnostic: it may run forward/backward passes, but it
never constructs an optimizer and never updates model parameters.  It compares
the audited rationale-first completion with the historical evidence-only
shortcut on the same prompt and action.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.sft_dataset import SFTExample  # noqa: E402
from agentic.training import load_jsonl, to_rendered_prompt_completion  # noqa: E402
from ml.agentic.training.train_sft import (  # noqa: E402
    configure_agent_sft_tokenizer,
    stratify_sft_rows_by_action,
)


SEGMENTS = ("action", "rationale_prefix", "evidence", "eos", "other")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_sha256(path: Path) -> str | None:
    adapter = path / "adapter_model.safetensors"
    return _sha256(adapter) if adapter.is_file() else None


def _call(row: dict[str, Any]) -> tuple[str, str, str, str]:
    example = SFTExample(**row)
    call = example.messages[-1].tool_calls[0].function
    reason = str(call.arguments.get("reason") or "")
    if "：" not in reason:
        raise ValueError(f"rationale-first delimiter missing: {example.example_id}")
    rationale, evidence = reason.split("：", 1)
    if not rationale or not evidence:
        raise ValueError(f"empty rationale/evidence segment: {example.example_id}")
    return call.name, reason, rationale, evidence


def _evidence_only_row(row: dict[str, Any], evidence: str) -> dict[str, Any]:
    rejected = copy.deepcopy(row)
    rejected["messages"][-1]["tool_calls"][0]["function"]["arguments"][
        "reason"
    ] = evidence
    return rejected


def _tokenize_with_segments(
    tokenizer: Any,
    rendered: dict[str, str],
    *,
    action: str,
    rationale: str | None,
    evidence: str,
) -> dict[str, Any]:
    prompt = rendered["prompt"]
    completion = rendered["completion"]
    full = prompt + completion
    encoded = tokenizer(
        full,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    offsets = encoded["offset_mapping"][0].tolist()
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if input_ids[: len(prompt_ids)].tolist() != list(prompt_ids):
        raise ValueError("rendered prompt is not a token prefix of the full sample")
    completion_start = len(prompt)

    def span(fragment: str) -> tuple[int, int]:
        start = completion.find(fragment)
        if start < 0:
            raise ValueError(f"completion fragment not found: {fragment!r}")
        start += completion_start
        return start, start + len(fragment)

    char_spans: dict[str, tuple[int, int]] = {
        "action": span(action),
        "evidence": span(evidence),
    }
    if rationale is not None:
        char_spans["rationale_prefix"] = span(rationale)
    labels = input_ids.clone()
    labels[: len(prompt_ids)] = -100
    segment_ids: dict[str, list[int]] = {name: [] for name in SEGMENTS}
    eos_id = tokenizer.eos_token_id
    for token_index in range(len(prompt_ids), len(input_ids)):
        token_id = int(input_ids[token_index])
        start, stop = offsets[token_index]
        if token_id == eos_id:
            segment = "eos"
        else:
            matches = [
                name
                for name, (left, right) in char_spans.items()
                if stop > left and start < right
            ]
            segment = matches[0] if matches else "other"
        segment_ids[segment].append(token_index)
    required = ("action", "evidence", "eos")
    if rationale is not None:
        required = (*required, "rationale_prefix")
    if any(not segment_ids[name] for name in required):
        raise ValueError("one or more required token segments are empty")
    return {
        "input_ids": input_ids,
        "attention_mask": encoded["attention_mask"][0],
        "labels": labels,
        "segments": segment_ids,
        "completion_tokens": int((labels != -100).sum()),
    }


def _per_token_loss(model: Any, item: dict[str, Any], torch: Any):
    device = next(model.parameters()).device
    input_ids = item["input_ids"].unsqueeze(0).to(device)
    attention_mask = item["attention_mask"].unsqueeze(0).to(device)
    completion_start = next(
        index for index, label in enumerate(item["labels"].tolist()) if label != -100
    )
    # Qwen3 can project only the final hidden states to vocabulary logits.  The
    # completion is a suffix, so retaining one preceding position plus all
    # completion tokens is exactly sufficient for assistant-only CE and avoids
    # materializing ~5k prompt positions x the full vocabulary.
    logits_to_keep = len(input_ids[0]) - completion_start + 1
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=logits_to_keep,
    )
    logits = outputs.logits[..., :-1, :].contiguous()
    labels = item["labels"][completion_start:].to(device)
    losses = torch.nn.functional.cross_entropy(
        logits[0], labels, reduction="none", ignore_index=-100
    )
    return losses, completion_start


def _loss_summary(
    losses: Any, item: dict[str, Any], completion_start: int
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in SEGMENTS:
        positions = [index - completion_start for index in item["segments"][name]]
        values = losses[positions]
        report[name] = {
            "tokens": len(positions),
            "nll_sum": float(values.sum().detach().cpu()),
            "nll_mean": float(values.mean().detach().cpu()),
        }
    completion_positions = list(range(len(losses)))
    values = losses[completion_positions]
    report["completion"] = {
        "tokens": len(completion_positions),
        "nll_sum": float(values.sum().detach().cpu()),
        "nll_mean": float(values.mean().detach().cpu()),
    }
    return report


def _grad_norm(model: Any, losses: Any, positions: list[int], denominator: int, torch: Any):
    model.zero_grad(set_to_none=True)
    (losses[positions].sum() / denominator).backward()
    squared = torch.zeros((), device=losses.device, dtype=torch.float32)
    parameters = 0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += parameter.grad.detach().float().pow(2).sum()
            parameters += parameter.grad.numel()
    return float(squared.sqrt().cpu()), parameters


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("zero-optimizer gradient diagnostic requires CUDA")
    rows = stratify_sft_rows_by_action(load_jsonl(args.dataset_dir / "train.jsonl"))
    rows = rows[: args.max_examples]
    action_counts = Counter(_call(row)[0] for row in rows)
    if len(action_counts) != 3:
        raise ValueError("diagnostic sample must contain all three target actions")

    tokenizer = configure_agent_sft_tokenizer(
        AutoTokenizer.from_pretrained(args.tokenizer or args.checkpoint)
    )
    rendered = to_rendered_prompt_completion(rows, tokenizer)
    rejected_rows = [_evidence_only_row(row, _call(row)[3]) for row in rows]
    rejected_rendered = to_rendered_prompt_completion(rejected_rows, tokenizer)

    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    peft_config = PeftConfig.from_pretrained(args.checkpoint)
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=quantization,
        device_map="auto",
        dtype=compute_dtype,
        trust_remote_code=False,
    )
    base_model = prepare_model_for_kbit_training(
        base_model, use_gradient_checkpointing=True
    )
    model = PeftModel.from_pretrained(
        base_model, args.checkpoint, is_trainable=True
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.eval()

    examples: list[dict[str, Any]] = []
    aggregates: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row, chosen_text, rejected_text in zip(
        rows, rendered, rejected_rendered, strict=True
    ):
        action, _, rationale, evidence = _call(row)
        chosen = _tokenize_with_segments(
            tokenizer,
            chosen_text,
            action=action,
            rationale=rationale,
            evidence=evidence,
        )
        rejected = _tokenize_with_segments(
            tokenizer,
            rejected_text,
            action=action,
            rationale=None,
            evidence=evidence,
        )
        with torch.no_grad():
            chosen_losses, chosen_start = _per_token_loss(model, chosen, torch)
            rejected_losses, rejected_start = _per_token_loss(model, rejected, torch)
        chosen_summary = _loss_summary(chosen_losses, chosen, chosen_start)
        rejected_completion_positions = list(range(len(rejected_losses)))
        rejected_values = rejected_losses[rejected_completion_positions]
        rejected_sum = float(rejected_values.sum().cpu())
        rejected_mean = float(rejected_values.mean().cpu())
        chosen_total = chosen_summary["completion"]
        comparison = {
            "chosen_minus_evidence_only_logprob_sum": (
                -chosen_total["nll_sum"] + rejected_sum
            ),
            "chosen_minus_evidence_only_logprob_mean": (
                -chosen_total["nll_mean"] + rejected_mean
            ),
            "chosen_tokens": chosen_total["tokens"],
            "evidence_only_tokens": len(rejected_completion_positions),
        }
        for segment, metrics in chosen_summary.items():
            aggregates[segment]["tokens"].append(float(metrics["tokens"]))
            aggregates[segment]["nll_sum"].append(metrics["nll_sum"])
            aggregates[segment]["nll_mean"].append(metrics["nll_mean"])
        aggregates["preference"]["sum_margin"].append(
            comparison["chosen_minus_evidence_only_logprob_sum"]
        )
        aggregates["preference"]["mean_margin"].append(
            comparison["chosen_minus_evidence_only_logprob_mean"]
        )
        examples.append(
            {
                "example_id": row.get("example_id"),
                "action": action,
                "chosen": chosen_summary,
                "preference": comparison,
            }
        )
        del chosen_losses, rejected_losses, rejected_values

    gc.collect()
    torch.cuda.empty_cache()

    gradient_examples: list[dict[str, Any]] = []
    # Transformers activates gradient checkpointing only while the module is
    # in training mode.  No optimizer exists, and the fixed seed makes LoRA
    # dropout deterministic for this diagnostic.
    torch.manual_seed(20260923)
    torch.cuda.manual_seed_all(20260923)
    model.train()
    for row, chosen_text in zip(rows[:3], rendered[:3], strict=True):
        action, _, rationale, evidence = _call(row)
        item = _tokenize_with_segments(
            tokenizer,
            chosen_text,
            action=action,
            rationale=rationale,
            evidence=evidence,
        )
        gradients: dict[str, Any] = {}
        for segment in ("action", "rationale_prefix", "evidence", "eos"):
            losses, completion_start = _per_token_loss(model, item, torch)
            positions = [
                index - completion_start for index in item["segments"][segment]
            ]
            norm, parameters = _grad_norm(
                model, losses, positions, item["completion_tokens"], torch
            )
            gradients[segment] = {
                "tokens": len(positions),
                "global_ce_contribution_grad_norm": norm,
                "trainable_gradient_elements": parameters,
            }
            del losses
            gc.collect()
            torch.cuda.empty_cache()
        gradient_examples.append(
            {
                "example_id": row.get("example_id"),
                "action": action,
                "segments": gradients,
            }
        )
    model.zero_grad(set_to_none=True)

    aggregate_report = {
        group: {
            metric: sum(values) / len(values)
            for metric, values in metrics.items()
        }
        for group, metrics in aggregates.items()
    }
    return {
        "schema_version": "sft-reason-signal-diagnostic.v1",
        "status": "completed_without_optimizer_step",
        "optimizer_constructed": False,
        "parameters_updated": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_adapter_sha256_before": _adapter_sha256(args.checkpoint),
        "checkpoint_adapter_sha256_after": _adapter_sha256(args.checkpoint),
        "dataset_dir": str(args.dataset_dir),
        "train_sha256": _sha256(args.dataset_dir / "train.jsonl"),
        "sample_selection": "deterministic-action-round-robin-prefix",
        "examples": len(rows),
        "action_counts": dict(sorted(action_counts.items())),
        "aggregate": aggregate_report,
        "per_example": examples,
        "gradient_examples": gradient_examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()
    if args.max_examples < 3:
        parser.error("--max-examples must be at least 3")
    report = diagnose(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
