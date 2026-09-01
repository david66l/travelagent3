"""Train a QLoRA travel-policy adapter with verified chosen/rejected pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.training import require_git_commit  # noqa: E402
from agentic.chat_template_contract import (  # noqa: E402
    AGENT_CHAT_TEMPLATE_KWARGS,
    AGENT_CHAT_TEMPLATE_SHA256,
    install_agent_chat_template,
    validate_agent_chat_template,
)
from agentic.reason_quality import (  # noqa: E402
    build_grounded_repair_reason,
    normalize_reason_text,
    repair_reason_rationale_prefixes,
    verifier_reason_quality_checks,
)

CONTRACT_EVIDENCE_POLICY = "verifier_success_or_deterministic_single_action_contract"
DECISION_BOUNDARY_EVIDENCE_POLICY = (
    "verifier_success_or_deterministic_decision_boundary_contract"
)
SAME_ACTION_REASON_QUALITY_POLICY = "deterministic_same_action_reason_quality_contract.v1"
_DPO_ACTIONS = ("retry_solve", "propose_tradeoff", "abort")
_DPO_NEGATIVE_TYPES = ("A_evidence_only", "B_wrong_action_rationale")
_FORBIDDEN_REASON_ARGUMENT_FIELDS = {
    "options",
    "strategy",
    "controller",
    "controller_arguments",
}


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




def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"preference split is empty: {path}")
    return rows


def _has_forbidden_reason_fields(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in _FORBIDDEN_REASON_ARGUMENT_FIELDS
            or _has_forbidden_reason_fields(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_forbidden_reason_fields(item) for item in value)
    return False


def _reason_call(response: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    calls = response.get("tool_calls") or []
    if len(calls) != 1:
        return None
    function = calls[0].get("function") or {}
    action = str(function.get("name") or "")
    arguments = function.get("arguments")
    return (action, arguments) if action and isinstance(arguments, dict) else None


def _has_target_rationale_prefix(reason: Any, action: str) -> bool:
    rendered = normalize_reason_text(reason)
    return any(
        rendered.startswith(normalize_reason_text(prefix))
        for prefix in repair_reason_rationale_prefixes(action)
    )


def _same_action_reason_quality_errors(row: dict[str, Any]) -> list[str]:
    """Validate the quarantine corpus locally; every failure blocks training."""
    pair_id = str(row.get("pair_id") or "")
    errors: list[str] = []
    chosen = row.get("chosen") or {}
    rejected = row.get("rejected") or {}
    chosen_call = _reason_call(chosen)
    rejected_call = _reason_call(rejected)
    if chosen_call is None or rejected_call is None:
        return [f"SAME_ACTION_CALL_INVALID:{pair_id}"]
    action, chosen_args = chosen_call
    rejected_action, rejected_args = rejected_call
    if action not in _DPO_ACTIONS or action != rejected_action:
        errors.append(f"SAME_ACTION_CHANGED:{pair_id}")
    if {
        key: value for key, value in chosen_args.items() if key != "reason"
    } != {key: value for key, value in rejected_args.items() if key != "reason"}:
        errors.append(f"SAME_ACTION_ARGUMENTS_CHANGED:{pair_id}")
    if set(chosen_args) != {"reason"} or set(rejected_args) != {"reason"}:
        errors.append(f"REASON_ARGUMENT_SCHEMA_INVALID:{pair_id}")
    if _has_forbidden_reason_fields(chosen_args) or _has_forbidden_reason_fields(rejected_args):
        errors.append(f"FORBIDDEN_REASON_ARGUMENT_FIELD:{pair_id}")
    evidence = str(row.get("evidence") or "").strip()
    phrases = [str(item).strip() for item in row.get("grounding_phrases") or [] if str(item).strip()]
    if not evidence or not phrases:
        errors.append(f"GROUNDING_EVIDENCE_MISSING:{pair_id}")
        return errors
    chosen_checks = verifier_reason_quality_checks(
        reason=chosen_args.get("reason"), target_action=action,
        grounding_phrases=phrases, evidence=evidence,
    )
    rejected_checks = verifier_reason_quality_checks(
        reason=rejected_args.get("reason"), target_action=action,
        grounding_phrases=phrases, evidence=evidence,
    )
    if not all(chosen_checks.values()):
        errors.append(f"CHOSEN_REASON_QUALITY_FAIL:{pair_id}")
    negative_type = str(row.get("negative_type") or "")
    if negative_type == "A_evidence_only":
        if str(rejected_args.get("reason") or "").strip() != evidence:
            errors.append(f"A_REASON_NOT_EVIDENCE_EOS:{pair_id}")
    elif negative_type == "B_wrong_action_rationale":
        if not rejected_checks["grounding_match"]:
            errors.append(f"B_GROUNDING_FAIL:{pair_id}")
        if (
            rejected_checks["reason_action_rationale_match"]
            or _has_target_rationale_prefix(rejected_args.get("reason"), action)
        ):
            errors.append(f"B_TARGET_RATIONALE_NOT_FAILED:{pair_id}")
    else:
        errors.append(f"NEGATIVE_TYPE_INVALID:{pair_id}")
    return errors


def _validate_train_only_reason_quality_dataset(
    dataset_dir: Path, manifest: dict[str, Any], minimum_train_examples: int
) -> dict[str, Any]:
    forbidden_splits = [split for split in ("validation", "test") if (dataset_dir / f"{split}.jsonl").exists()]
    rows = load_jsonl(dataset_dir / "train.jsonl")
    errors: list[str] = []
    if manifest.get("status") != "passed":
        errors.append("MANIFEST_NOT_PASSED")
    if forbidden_splits:
        errors.append("TRAIN_ONLY_SPLIT_VIOLATION:" + ",".join(forbidden_splits))
    if manifest.get("split_policy") != "train-only":
        errors.append("TRAIN_ONLY_MANIFEST_POLICY_MISSING")
    audit = manifest.get("audit")
    if not isinstance(audit, dict) or not audit.get("ready"):
        errors.append("BUILDER_HARD_AUDIT_NOT_PASSED")
    elif (
        abs(float(audit.get("mean_chosen_minus_rejected_completion_tokens", float("inf")))) > 0.5
        or not 0.45 <= float(audit.get("length_only_auc", 1.0)) <= 0.55
    ):
        errors.append("BUILDER_LENGTH_AUDIT_INVALID")
    if len(rows) < minimum_train_examples:
        errors.append(f"TRAIN_TOO_SMALL:{len(rows)}<{minimum_train_examples}")
    pair_ids: set[str] = set()
    contexts: set[str] = set()
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_pairs: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    negative_types: Counter[str] = Counter()
    expected_order = [
        (action, negative)
        for action in _DPO_ACTIONS
        for negative in _DPO_NEGATIVE_TYPES
    ]
    for index, row in enumerate(rows):
        pair_id = str(row.get("pair_id") or "")
        if not pair_id or pair_id in pair_ids:
            errors.append(f"INVALID_OR_DUPLICATE_PAIR:{pair_id}")
        pair_ids.add(pair_id)
        context = str(row.get("context_hash") or "")
        if not context:
            errors.append(f"CONTEXT_HASH_MISSING:{pair_id}")
        contexts.add(context)
        by_context[context].append(row)
        source_pairs[str(row.get("source_task_id") or "")] += 1
        if not isinstance(row.get("messages"), list) or len(row["messages"]) < 2:
            errors.append(f"PROMPT_MESSAGES_INVALID:{pair_id}")
        if not isinstance(row.get("tools"), list) or not row["tools"]:
            errors.append(f"TOOLS_INVALID:{pair_id}")
        errors.extend(_same_action_reason_quality_errors(row))
        action = str(row.get("action") or "")
        negative = str(row.get("negative_type") or "")
        chosen_call = _reason_call(row.get("chosen") or {})
        if (
            row.get("schema_version") != SAME_ACTION_REASON_QUALITY_POLICY
            or row.get("reason_codes") != ["SAME_ACTION_REASON_QUALITY_CONTRACT"]
            or chosen_call is None
            or action != chosen_call[0]
            or row.get("family") != f"{action}:{negative}"
        ):
            errors.append(f"QUARANTINE_ROW_METADATA_INVALID:{pair_id}")
        if context != hashlib.sha256(json.dumps({"messages": row.get("messages"), "tools": row.get("tools")}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
            errors.append(f"CONTEXT_HASH_MISMATCH:{pair_id}")
        if chosen_call is not None and str(chosen_call[1].get("reason") or "") != build_grounded_repair_reason(str(row.get("evidence") or ""), chosen_call[0]):
            errors.append(f"CHOSEN_REASON_NOT_CANONICAL:{pair_id}")
        actions[action] += 1
        negative_types[negative] += 1
        if (action, negative) != expected_order[index % len(expected_order)]:
            errors.append(f"NON_SEQUENTIAL_SIX_GROUP_ORDER:{pair_id}")
    if len(rows) != 600 or len(contexts) != 300:
        errors.append(f"QUARANTINE_CARDINALITY_INVALID:{len(rows)}:{len(contexts)}")
    if any(actions[action] != 200 for action in _DPO_ACTIONS):
        errors.append("ACTION_BALANCE_INVALID")
    if any(negative_types[negative] != 300 for negative in _DPO_NEGATIVE_TYPES):
        errors.append("NEGATIVE_TYPE_BALANCE_INVALID")
    if len(source_pairs) != 40 or set(source_pairs.values()) - {14, 16}:
        errors.append("SOURCE_PAIR_CONTRIBUTION_INVALID")
    for context, members in by_context.items():
        if len(members) != 2 or {item.get("negative_type") for item in members} != set(_DPO_NEGATIVE_TYPES):
            errors.append(f"CONTEXT_PAIR_CARDINALITY_INVALID:{context}")
            continue
        frozen = [
            hashlib.sha256(json.dumps({key: item.get(key) for key in ("messages", "tools", "chosen", "action", "evidence", "grounding_phrases", "source_task_id")}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            for item in members
        ]
        if len(set(frozen)) != 1:
            errors.append(f"CONTEXT_PAIR_PAYLOAD_MISMATCH:{context}")
    return {
        "ready": not errors,
        "dataset_version": str(manifest.get("dataset_version") or "unknown"),
        "split_counts": {"train": len(rows)},
        "family_counts": dict(actions),
        "unique_pairs": len(pair_ids),
        "errors": errors,
    }


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
    if evidence_policy == SAME_ACTION_REASON_QUALITY_POLICY:
        return _validate_train_only_reason_quality_dataset(
            dataset_dir, manifest, minimum_train_examples
        )
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
            "chat_template_kwargs": dict(AGENT_CHAT_TEMPLATE_KWARGS),
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


def _checkpoint_provenance(path: Path) -> dict[str, Any]:
    """Small, stable audit record for the exact policy/ref checkpoint bytes."""
    candidates = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
    files = {
        name: hashlib.sha256((path / name).read_bytes()).hexdigest()
        for name in candidates
        if (path / name).is_file()
    }
    if "adapter_config.json" not in files:
        raise ValueError("audited policy checkpoint lacks adapter_config.json")
    return {"path": str(path.resolve()), "files_sha256": files}


def _adapter_payload_sha256(path: Path) -> str:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
    raise ValueError("audited policy checkpoint lacks adapter weights")


def _adapter_parameter_hash(model: Any, adapter_name: str) -> str:
    """Hash only PEFT adapter tensors, never the quantized base model."""
    import torch
    from peft import get_peft_model_state_dict

    state = get_peft_model_state_dict(model, adapter_name=adapter_name)
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _require_quarantine_source(path: Path) -> None:
    """V13 is explicitly prohibited; operator evidence records the V11 sync."""
    if "v13" in str(path).casefold():
        raise ValueError("quarantine DPO must not use a V13 checkpoint")


def _stop_after_steps_callback(max_steps: int) -> Any:
    from transformers import TrainerCallback

    class StopAfterSteps(TrainerCallback):
        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            if state.global_step >= max_steps:
                control.should_training_stop = True
            return control

    return StopAfterSteps()


def _validate_quarantine_gradient_isolation(model: Any, policy_adapter_name: str) -> None:
    for name, parameter in model.named_parameters():
        if ".ref." in name:
            if parameter.requires_grad or parameter.grad is not None:
                raise RuntimeError(f"frozen reference parameter is trainable or has a gradient: {name}")
        elif parameter.requires_grad and (
            "lora_" not in name or f".{policy_adapter_name}." not in name
        ):
            raise RuntimeError(f"non-policy LoRA parameter is trainable: {name}")


def _validate_quarantine_logs(log_history: list[dict[str, Any]]) -> None:
    first_step = next(
        (item for item in log_history if item.get("step") == 1),
        None,
    )
    if first_step is None or not {"loss", "grad_norm"} <= set(first_step):
        raise RuntimeError(
            "quarantine DPO did not emit step-1 loss and gradient records"
        )
    for item in log_history:
        for key in ("loss", "grad_norm"):
            if key in item and not math.isfinite(float(item[key])):
                raise RuntimeError(f"quarantine DPO emitted non-finite {key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Balanced SFT PEFT checkpoint")
    parser.add_argument("--minimum-train-examples", type=int, default=600)
    parser.add_argument("--max-length", type=int, default=5120)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--max-eval-examples", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--stop-after-steps", type=int, default=12)
    parser.add_argument("--expected-source-adapter-sha256")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument(
        "--allow-unknown-git-commit",
        action="store_true",
        help="Escape hatch for non-repo smoke sandboxes only; formal training runs "
        "must stay reproducible from a git commit.",
    )
    args = parser.parse_args()
    git_commit = require_git_commit(
        parser, REPO_ROOT, allow_unknown=args.allow_unknown_git_commit
    )
    if args.beta <= 0:
        raise ValueError("DPO beta must be positive")
    if not (Path(args.model) / "adapter_config.json").is_file():
        raise ValueError("DPO must continue from an audited SFT PEFT adapter")

    minimum = 1 if args.allow_small_dataset else args.minimum_train_examples
    dataset_report = validate_preference_dataset(args.dataset_dir, minimum)
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    is_train_only = manifest.get("preference_evidence_policy") == SAME_ACTION_REASON_QUALITY_POLICY
    if is_train_only:
        fixed = {
            "max_steps": 24,
            "learning_rate": 5e-6,
            "beta": 0.1,
            "warmup_steps": 1,
            "max_grad_norm": 1.0,
            "seed": 20260923,
            "max_train_examples": 0,
            "minimum_train_examples": 600,
        }
        if {key: getattr(args, key) for key in fixed} != fixed or args.allow_small_dataset:
            raise ValueError("quarantine DPO parameters are fixed; sweeps are prohibited")
        if args.max_length < 5120:
            raise ValueError("quarantine DPO requires --max-length >= 5120")
        if args.batch_size * args.gradient_accumulation != 12:
            raise ValueError("quarantine DPO requires effective batch size exactly 12")
        if not 0 < args.stop_after_steps <= args.max_steps:
            raise ValueError("--stop-after-steps must be in (0, --max-steps]")
        if not args.expected_source_adapter_sha256:
            raise ValueError("quarantine DPO requires --expected-source-adapter-sha256")
        _require_quarantine_source(Path(args.model))
        if _adapter_payload_sha256(Path(args.model)) != args.expected_source_adapter_sha256:
            raise ValueError("synced V11 adapter hash does not match --expected-source-adapter-sha256")
    print(json.dumps(dataset_report, ensure_ascii=False, indent=2))
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    install_agent_chat_template(tokenizer)
    validate_agent_chat_template(tokenizer.chat_template)
    train_rows = select_stratified_rows(
        load_jsonl(args.dataset_dir / "train.jsonl"), args.max_train_examples
    )
    validation_rows = [] if is_train_only else select_stratified_rows(
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
    policy_adapter_name = str(model.active_adapter)
    policy_parameter_hash_before = _adapter_parameter_hash(model, policy_adapter_name)
    reference_parameter_hash_before = _adapter_parameter_hash(model, "ref")
    if is_train_only:
        _validate_quarantine_gradient_isolation(model, policy_adapter_name)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    report_to = ["mlflow"] if os.environ.get("MLFLOW_TRACKING_URI") else []
    config = DPOConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        beta=0.1 if is_train_only else args.beta,
        loss_type=["sigmoid"],
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=True,
        max_length=args.max_length,
        truncation_mode="keep_start",
        logging_steps=1 if is_train_only else 5,
        eval_strategy="no" if is_train_only else "steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=12 if is_train_only else 50,
        save_total_limit=2,
        bf16=compute_dtype == torch.bfloat16,
        fp16=compute_dtype == torch.float16,
        seed=args.seed,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        train_sampling_strategy="sequential" if is_train_only else "random",
        lr_scheduler_type="linear",
        report_to=report_to,
        run_name=f"agent-policy-dpo-{dataset_report['dataset_version']}",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=Dataset.from_list(to_dpo_rows(train_rows)),
        eval_dataset=None if is_train_only else Dataset.from_list(to_dpo_rows(validation_rows)),
        processing_class=tokenizer,
        # TRL 0.x / Transformers 5 owns model_accepts_loss_kwargs and sets it
        # false in DPOTrainer.__init__; do not override that safety contract.
        callbacks=[_stop_after_steps_callback(args.stop_after_steps)] if is_train_only else None,
    )
    train_result = trainer.train(
        resume_from_checkpoint=(str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    )
    policy_parameter_hash_after = _adapter_parameter_hash(model, policy_adapter_name)
    reference_parameter_hash_after = _adapter_parameter_hash(model, "ref")
    if is_train_only:
        _validate_quarantine_gradient_isolation(model, policy_adapter_name)
        _validate_quarantine_logs(trainer.state.log_history)
        if reference_parameter_hash_after != reference_parameter_hash_before:
            raise RuntimeError("frozen DPO reference adapter changed during training")
        if policy_parameter_hash_after == policy_parameter_hash_before:
            raise RuntimeError("DPO policy adapter did not change during training")
        if trainer.state.global_step != args.stop_after_steps:
            raise RuntimeError("DPO trainer did not stop at the audited stop-after step")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    eval_metrics = {} if is_train_only else trainer.evaluate()
    metadata = {
        "status": "trained",
        "run_scope": "quarantine" if is_train_only else "formal",
        "method": "direct-preference-optimization",
        "source_model": args.model,
        "policy_checkpoint": _checkpoint_provenance(Path(args.model)),
        "reference_checkpoint": _checkpoint_provenance(Path(args.model)),
        "reference_policy": reference_policy,
        "policy_adapter_name": policy_adapter_name,
        "policy_adapter_parameter_sha256": {
            "before": policy_parameter_hash_before,
            "after": policy_parameter_hash_after,
        },
        "reference_adapter_parameter_sha256": {
            "before": reference_parameter_hash_before,
            "after": reference_parameter_hash_after,
        },
        "dataset_version": dataset_report["dataset_version"],
        "git_commit": git_commit,
        "seed": args.seed,
        "beta": args.beta,
        "loss_type": "sigmoid",
        "quantization": "nf4-double-quant",
        "train_examples": len(train_rows),
        "eval_examples": len(validation_rows),
        "effective_batch_size": args.batch_size * args.gradient_accumulation,
        "max_steps": args.max_steps,
        "stop_after_steps": args.stop_after_steps,
        "global_step": trainer.state.global_step,
        "warmup_steps": args.warmup_steps,
        "max_grad_norm": args.max_grad_norm,
        "train_sampling_strategy": "sequential",
        "chat_template_sha256": AGENT_CHAT_TEMPLATE_SHA256,
        "chat_template_kwargs": AGENT_CHAT_TEMPLATE_KWARGS,
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
