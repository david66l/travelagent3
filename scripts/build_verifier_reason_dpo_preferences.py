"""Build the one-shot, train-only same-action reason-quality DPO corpus.

This command deliberately accepts *files*, not corpus directories: it can only
read the rationale-first balanced SFT train split and the original GRPO train
split.  Each of the 300 SFT decisions produces an evidence-only negative (A)
and a longer, wrong-action-rationale negative (B), while preserving the exact
prompt, tools, tool name and arguments.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

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


SCHEMA_VERSION = "deterministic_same_action_reason_quality_contract.v1"
NEGATIVE_TYPES = ("A_evidence_only", "B_wrong_action_rationale")
ACTIONS = ("retry_solve", "propose_tradeoff", "abort")
FORBIDDEN_ARGUMENT_FIELDS = {"options", "strategy", "controller", "controller_arguments"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"required train-only input is missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"train-only input is empty: {path}")
    return rows


def _call(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    calls = response.get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError("completion must contain exactly one tool call")
    function = calls[0].get("function") or {}
    action = str(function.get("name") or "")
    arguments = function.get("arguments")
    if not action or not isinstance(arguments, dict):
        raise ValueError("completion tool call is malformed")
    return action, arguments


def _has_forbidden_fields(value: Any) -> bool:
    if isinstance(value, dict):
        return any(str(key).casefold() in FORBIDDEN_ARGUMENT_FIELDS or _has_forbidden_fields(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_has_forbidden_fields(item) for item in value)
    return False


def _visible_violation(contract: dict[str, Any]) -> tuple[str, list[str]]:
    messages = contract.get("prompt_messages") or []
    if not messages:
        raise ValueError("GRPO decision contract has no visible prompt")
    try:
        visible = json.loads(str(messages[-1].get("content") or "{}"))["policy_state"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("GRPO decision contract has no visible policy state") from exc
    reports = [item for item in visible.get("relevant_artifacts") or [] if isinstance(item, dict) and item.get("artifact_type") == "validation_report"]
    violation = ((reports[-1].get("violations") or [{}])[0] if reports else {})
    evidence = str(violation.get("message") or "").strip()
    phrases = [str(item).strip() for item in contract.get("grounding_phrases") or [] if str(item).strip()]
    if not evidence or not phrases:
        raise ValueError("GRPO decision contract lacks grounding evidence")
    return evidence, phrases


def _grpo_mapping(grpo_train: Path) -> dict[str, dict[str, Any]]:
    """Map SFT scenario task ids to their original train-only decision contract."""
    # Keep the lightweight audit helpers importable in minimal test runtimes;
    # the full GRPO model schema is needed only when an operator builds data.
    from agentic.grpo_training import load_grpo_corpus

    mapped: dict[str, dict[str, Any]] = {}
    for row in load_grpo_corpus(grpo_train):
        contract = row.snapshot.hidden_test_facts.get("grpo_decision_state")
        if not isinstance(contract, dict):
            continue
        if contract.get("schema_version") != "react-verifier-repair-decision.v5":
            continue
        task_id = str(row.task.task_id)
        if task_id in mapped:
            raise ValueError(f"duplicate GRPO task mapping: {task_id}")
        mapped[task_id] = contract
    if not mapped:
        raise ValueError("original GRPO train contains no supported decision contracts")
    return mapped


def _assistant(action: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": action, "arguments": arguments}}]}


def _wrong_actions(action: str) -> tuple[str, ...]:
    return tuple(other for other in ACTIONS if other != action)


def has_target_rationale_prefix(reason: Any, action: str) -> bool:
    rendered = normalize_reason_text(reason)
    return any(rendered.startswith(normalize_reason_text(prefix)) for prefix in repair_reason_rationale_prefixes(action))


def _wrong_reason_candidates(action: str, evidence: str) -> list[str]:
    """A bounded candidate pool; repetition is chosen by the length DP below."""
    candidates: list[str] = []
    tails = (
        "该判断直接来自当前条件",
        "现有信息已足以支持这一结论",
        "后续处理应遵循同一结论",
        "此结论与当前条件保持一致",
    )
    for wrong in _wrong_actions(action):
        for prefix in repair_reason_rationale_prefixes(wrong):
            base = f"{prefix}：{evidence.rstrip('。.!！')}"
            for copies in range(1, 13):
                tail = "；".join(tails[index % len(tails)] for index in range(copies))
                candidates.append(f"{base}；{tail}。")
    return candidates


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, dict):
        value = value["input_ids"]
    elif hasattr(value, "input_ids"):
        value = value.input_ids
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value[0] if value and isinstance(value[0], list) else value)


def _plain_token_length(tokenizer: Any, text: str) -> int:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(_token_ids(encoded))


def rendered_lengths(tokenizer: Any, prompt: list[dict[str, Any]], tools: list[dict[str, Any]], response: dict[str, Any], max_length: int) -> tuple[int, int]:
    """Return full and completion lengths while proving the template boundary."""
    kwargs = {"tools": tools, **AGENT_CHAT_TEMPLATE_KWARGS}
    prompt_ids = _token_ids(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, **kwargs))
    full_ids = _token_ids(tokenizer.apply_chat_template([*prompt, response], tokenize=True, add_generation_prompt=False, **kwargs))
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("PROMPT_PREFIX_BOUNDARY_MISMATCH")
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("COMPLETION_SUFFIX_BOUNDARY_MISMATCH")
    if len(full_ids) > max_length:
        raise ValueError(f"TOKENIZATION_WOULD_TRUNCATE:{len(full_ids)}>{max_length}")
    return len(full_ids), len(full_ids) - len(prompt_ids)


def _length_compensation_candidates(tokenizer: Any, prompt: list[dict[str, Any]], tools: list[dict[str, Any]], action: str, arguments: dict[str, Any], evidence: str, grounding_phrases: list[str], chosen_tokens: int, max_length: int) -> list[tuple[int, str]]:
    """Return valid B candidates for the per-action token-balance DP."""
    # Full tool-template rendering is intentionally reserved for the nearest
    # candidates.  Ranking with plain text tokens is safe only as a prefilter;
    # all final choices and the full 1,200-response audit use the real shared
    # template and exact prompt boundary.
    plain_target = (
        2 * _plain_token_length(tokenizer, str(arguments.get("reason") or ""))
        - _plain_token_length(tokenizer, evidence)
    )
    candidates_to_render = sorted(
        _wrong_reason_candidates(action, evidence),
        key=lambda reason: (
            abs(_plain_token_length(tokenizer, reason) - plain_target),
            reason,
        ),
    )[:16]
    choices: list[tuple[int, str]] = []
    for reason in candidates_to_render:
        candidate_args = {**arguments, "reason": reason}
        candidate = _assistant(action, candidate_args)
        _, completion = rendered_lengths(tokenizer, prompt, tools, candidate, max_length)
        checks = verifier_reason_quality_checks(reason=reason, target_action=action, grounding_phrases=grounding_phrases, evidence=evidence)
        if (
            checks["grounding_match"]
            and not checks["reason_action_rationale_match"]
            and not has_target_rationale_prefix(reason, action)
            and completion > chosen_tokens
        ):
            choices.append((completion, reason))
    if not choices:
        raise ValueError("NO_VALID_LONG_WRONG_RATIONALE_CANDIDATE")
    return sorted(set(choices), key=lambda item: (item[0], item[1]))


def select_length_compensation_dp(items: list[dict[str, Any]]) -> list[str]:
    """Choose one B per context, minimizing the action-level signed delta.

    ``items`` has fixed A deltas and a finite B candidate pool.  The DP state
    is the sum of chosen-minus-rejected suffix tokens, so it cannot silently
    trade one action's length shortcut against another action's distribution.
    """
    states: dict[int, tuple[int, ...]] = {0: ()}
    for item in items:
        candidates = item["b_candidates"]
        next_states: dict[int, tuple[int, ...]] = {}
        for total, picks in states.items():
            for index, (b_tokens, _reason) in enumerate(candidates):
                delta = item["a_delta"] + item["chosen_tokens"] - b_tokens
                candidate_total = total + delta
                proposal = (*picks, index)
                previous = next_states.get(candidate_total)
                if previous is None or proposal < previous:
                    next_states[candidate_total] = proposal
        states = next_states
    _total, picks = min(states.items(), key=lambda item: (abs(item[0]), item[0], item[1]))
    return [items[index]["b_candidates"][pick][1] for index, pick in enumerate(picks)]


def _auc(scores: list[float], labels: list[int]) -> float:
    """Tie-aware ROC AUC without a sklearn dependency."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUC requires both classes")
    wins = 0.0
    for score, label in zip(scores, labels):
        if label != 1:
            continue
        for other, other_label in zip(scores, labels):
            if other_label != 0:
                continue
            wins += 1.0 if score > other else 0.5 if score == other else 0.0
    return wins / (positives * negatives)


def _round_robin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["action"]), str(row["negative_type"]))].append(row)
    expected = [(action, negative) for action in ACTIONS for negative in NEGATIVE_TYPES]
    if set(groups) != set(expected) or any(len(groups[key]) != 100 for key in expected):
        raise ValueError("six DPO groups must each contain exactly 100 pairs")
    for key in expected:
        groups[key].sort(key=lambda row: str(row["source_task_id"]))
    return [groups[key][index] for index in range(100) for key in expected]


def audit_pairs(rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> dict[str, Any]:
    errors: list[str] = []
    contexts: set[str] = set()
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_pairs: Counter[str] = Counter()
    pair_ids: set[str] = set()
    action_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    signed_deltas: list[int] = []
    completion_lengths: list[int] = []
    full_lengths: list[int] = []
    labels: list[int] = []
    for row in rows:
        pair_id = str(row.get("pair_id") or "")
        if not pair_id or pair_id in pair_ids:
            errors.append(f"NON_UNIQUE_PAIR:{pair_id}")
        pair_ids.add(pair_id)
        context = str(row.get("context_hash") or "")
        if not context:
            errors.append(f"CONTEXT_MISSING:{pair_id}")
        contexts.add(context)
        by_context[context].append(row)
        source_pairs[str(row.get("source_task_id") or "")] += 1
        prompt, tools = row.get("messages"), row.get("tools")
        if not isinstance(prompt, list) or not isinstance(tools, list) or not tools:
            errors.append(f"PROMPT_OR_TOOLS_INVALID:{pair_id}")
            continue
        try:
            chosen_action, chosen_args = _call(row["chosen"])
            rejected_action, rejected_args = _call(row["rejected"])
            if chosen_action != rejected_action or {key: value for key, value in chosen_args.items() if key != "reason"} != {key: value for key, value in rejected_args.items() if key != "reason"}:
                errors.append(f"SAME_ACTION_OR_ARGUMENTS_CHANGED:{pair_id}")
            if row.get("schema_version") != SCHEMA_VERSION or row.get("reason_codes") != ["SAME_ACTION_REASON_QUALITY_CONTRACT"]:
                errors.append(f"SCHEMA_OR_REASON_CODE_INVALID:{pair_id}")
            if row.get("action") != chosen_action or row.get("family") != f"{chosen_action}:{row.get('negative_type')}":
                errors.append(f"ACTION_OR_FAMILY_METADATA_INVALID:{pair_id}")
            if context != _digest({"messages": prompt, "tools": tools}):
                errors.append(f"CONTEXT_HASH_MISMATCH:{pair_id}")
            if _has_forbidden_fields(chosen_args) or _has_forbidden_fields(rejected_args):
                errors.append(f"FORBIDDEN_ARGUMENT_FIELD:{pair_id}")
            evidence = str(row.get("evidence") or "")
            phrases = [str(item) for item in row.get("grounding_phrases") or []]
            chosen_checks = verifier_reason_quality_checks(reason=chosen_args.get("reason"), target_action=chosen_action, grounding_phrases=phrases, evidence=evidence)
            rejected_checks = verifier_reason_quality_checks(reason=rejected_args.get("reason"), target_action=chosen_action, grounding_phrases=phrases, evidence=evidence)
            if not all(chosen_checks.values()):
                errors.append(f"CHOSEN_REASON_QUALITY_FAIL:{pair_id}")
            if str(chosen_args.get("reason") or "") != build_grounded_repair_reason(evidence, chosen_action):
                errors.append(f"CHOSEN_REASON_NOT_CANONICAL:{pair_id}")
            negative_type = row.get("negative_type")
            if negative_type == "A_evidence_only":
                if str(rejected_args.get("reason") or "").strip() != evidence:
                    errors.append(f"A_NOT_EVIDENCE_THEN_EOS:{pair_id}")
            elif negative_type == "B_wrong_action_rationale":
                if (
                    not rejected_checks["grounding_match"]
                    or rejected_checks["reason_action_rationale_match"]
                    or has_target_rationale_prefix(rejected_args.get("reason"), chosen_action)
                ):
                    errors.append(f"B_REASON_CONTRACT_FAIL:{pair_id}")
            else:
                errors.append(f"UNKNOWN_NEGATIVE_TYPE:{pair_id}")
            chosen_full, chosen_tokens = rendered_lengths(tokenizer, prompt, tools, row["chosen"], max_length)
            rejected_full, rejected_tokens = rendered_lengths(tokenizer, prompt, tools, row["rejected"], max_length)
            signed_deltas.append(chosen_tokens - rejected_tokens)
            # The length-only shortcut audit deliberately measures completion
            # tokens.  Prompt length is identical within a preference pair and
            # must not camouflage a response-length correlation.
            completion_lengths.extend((chosen_tokens, rejected_tokens))
            full_lengths.extend((chosen_full, rejected_full))
            labels.extend((1, 0))
            action_counts[chosen_action] += 1
            type_counts[str(negative_type)] += 1
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"PAIR_AUDIT:{pair_id}:{exc}")
    mean_delta = sum(signed_deltas) / len(signed_deltas) if signed_deltas else float("inf")
    auc = _auc(completion_lengths, labels) if completion_lengths else 1.0
    if len(rows) != 600 or len(contexts) != 300:
        errors.append("PAIR_OR_CONTEXT_COUNT_MISMATCH")
    if set(action_counts) != set(ACTIONS) or any(action_counts[action] != 200 for action in ACTIONS):
        errors.append("ACTION_BALANCE_MISMATCH")
    if any(type_counts[negative] != 300 for negative in NEGATIVE_TYPES):
        errors.append("NEGATIVE_TYPE_BALANCE_MISMATCH")
    if len(source_pairs) != 40 or set(source_pairs.values()) - {14, 16}:
        errors.append("SOURCE_PAIR_CONTRIBUTION_MISMATCH")
    for context, members in by_context.items():
        if len(members) != 2 or {item.get("negative_type") for item in members} != set(NEGATIVE_TYPES):
            errors.append(f"CONTEXT_DOES_NOT_HAVE_EXACT_A_B:{context}")
            continue
        frozen = [
            _digest({key: item.get(key) for key in ("messages", "tools", "chosen", "action", "evidence", "grounding_phrases", "source_task_id")})
            for item in members
        ]
        if len(set(frozen)) != 1:
            errors.append(f"CONTEXT_PAIR_PAYLOAD_MISMATCH:{context}")
    if abs(mean_delta) > 0.5:
        errors.append(f"OVERALL_TOKEN_DELTA_UNBALANCED:{mean_delta:.4f}")
    for action in ACTIONS:
        values = [delta for row, delta in zip(rows, signed_deltas) if row.get("action") == action]
        if abs(sum(values) / len(values)) > 1.0:
            errors.append(f"ACTION_TOKEN_DELTA_UNBALANCED:{action}")
    if not 0.45 <= auc <= 0.55:
        errors.append(f"LENGTH_ONLY_AUC_OUT_OF_RANGE:{auc:.6f}")
    return {"ready": not errors, "pair_count": len(rows), "unique_contexts": len(contexts), "action_counts": dict(action_counts), "negative_type_counts": dict(type_counts), "mean_chosen_minus_rejected_completion_tokens": mean_delta, "length_only_auc": auc, "max_full_sequence_tokens": max(full_lengths, default=0), "errors": errors}


def build(sft_train: Path, grpo_train: Path, output_dir: Path, tokenizer: Any, max_length: int = 5120) -> dict[str, Any]:
    validate_agent_chat_template(getattr(tokenizer, "chat_template", None))
    sft_rows = _read_jsonl(sft_train)
    mapping = _grpo_mapping(grpo_train)
    if len(sft_rows) != 300:
        raise ValueError(f"balanced SFT train must contain exactly 300 rows, got {len(sft_rows)}")
    pairs: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    source_contributions: Counter[str] = Counter()
    for raw in sft_rows:
        messages = copy.deepcopy(raw.get("messages") or [])
        if not messages or not isinstance(messages[-1], dict) or messages[-1].get("role") != "assistant":
            raise ValueError("SFT row must end with exactly one assistant completion")
        chosen = messages.pop()
        task_id = str(raw.get("scenario_id") or "")
        contract = mapping.get(task_id)
        if contract is None:
            raise ValueError(f"SFT scenario has no original GRPO train mapping: {task_id}")
        source_id = str(contract.get("source_task_id") or "")
        if not source_id:
            raise ValueError(f"GRPO contract lacks source_task_id: {task_id}")
        source_contributions[source_id] += 1
        action, arguments = _call(chosen)
        evidence, phrases = _visible_violation(contract)
        if action not in ACTIONS or action != str(contract.get("target_action") or ""):
            raise ValueError(f"SFT action differs from original GRPO contract: {task_id}")
        if _has_forbidden_fields(arguments) or set(arguments) != {"reason"}:
            raise ValueError(f"SFT completion exposes non-policy argument fields: {task_id}")
        chosen_reason = str(arguments.get("reason") or "")
        checks = verifier_reason_quality_checks(reason=chosen_reason, target_action=action, grounding_phrases=phrases, evidence=evidence)
        if not all(checks.values()):
            raise ValueError(f"SFT chosen reason failed full quality: {task_id}")
        tools = copy.deepcopy(raw.get("tools") or [])
        _chosen_full, chosen_tokens = rendered_lengths(tokenizer, messages, tools, chosen, max_length)
        a_args = {**arguments, "reason": evidence}
        rejected_a = _assistant(action, a_args)
        _, a_tokens = rendered_lengths(tokenizer, messages, tools, rejected_a, max_length)
        pending.append({"task_id": task_id, "source_task_id": source_id, "context_hash": _digest({"messages": messages, "tools": tools}), "action": action, "evidence": evidence, "grounding_phrases": phrases, "messages": messages, "tools": tools, "chosen": chosen, "rejected_a": rejected_a, "chosen_tokens": chosen_tokens, "a_delta": chosen_tokens - a_tokens, "b_candidates": _length_compensation_candidates(tokenizer, messages, tools, action, arguments, evidence, phrases, chosen_tokens, max_length)})
    if len(source_contributions) != 40:
        raise ValueError(f"optimization scope must be exactly 40 source_task_id values, got {len(source_contributions)}")
    if set(source_contributions.values()) - {7, 8}:
        raise ValueError("each of the 40 optimization sources must contribute seven or eight SFT rows")
    for action in ACTIONS:
        action_items = [item for item in pending if item["action"] == action]
        if len(action_items) != 100:
            raise ValueError(f"balanced SFT must contain exactly 100 {action} rows")
        selected_reasons = select_length_compensation_dp(action_items)
        for item, b_reason in zip(action_items, selected_reasons):
            common = {"schema_version": SCHEMA_VERSION, "family": None, "source_task_id": item["source_task_id"], "context_hash": item["context_hash"], "action": action, "evidence": item["evidence"], "grounding_phrases": item["grounding_phrases"], "messages": item["messages"], "tools": item["tools"], "chosen": item["chosen"], "reason_codes": ["SAME_ACTION_REASON_QUALITY_CONTRACT"]}
            pairs.append({**common, "pair_id": f"reason-quality:A_evidence_only:{item['task_id']}", "family": f"{action}:A_evidence_only", "negative_type": "A_evidence_only", "rejected": item["rejected_a"]})
            pairs.append({**common, "pair_id": f"reason-quality:B_wrong_action_rationale:{item['task_id']}", "family": f"{action}:B_wrong_action_rationale", "negative_type": "B_wrong_action_rationale", "rejected": _assistant(action, {"reason": b_reason})})
    ordered = _round_robin(pairs)
    audit = audit_pairs(ordered, tokenizer, max_length)
    manifest = {"schema_version": SCHEMA_VERSION, "status": "passed" if audit["ready"] else "rejected", "dataset_version": "verifier-reason-dpo-" + _digest([row["pair_id"] for row in ordered])[:16], "preference_evidence_policy": SCHEMA_VERSION, "requires_verifier_success_over_failure": False, "split_policy": "train-only", "splits": {"train": len(ordered)}, "source_files_read": [sft_train.as_posix(), grpo_train.as_posix()], "official_validation_or_test_used": False, "source_sha256": {"rationale_first_balanced_sft_train": _sha256(sft_train), "original_grpo_train": _sha256(grpo_train)}, "source_task_id_count": len(source_contributions), "source_contribution_counts": dict(sorted(source_contributions.items())), "chat_template_sha256": AGENT_CHAT_TEMPLATE_SHA256, "chat_template_kwargs": AGENT_CHAT_TEMPLATE_KWARGS, "max_length": max_length, "sampling_order": "action_x_negative_type_round_robin.v1", "audit": audit}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in ordered), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-train", type=Path, required=True)
    parser.add_argument("--grpo-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True, help="audited V11 step-12 checkpoint tokenizer")
    parser.add_argument("--max-length", type=int, default=5120)
    args = parser.parse_args()
    if args.max_length < 5120:
        raise ValueError("quarantine DPO requires --max-length >= 5120")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=False)
    install_agent_chat_template(tokenizer)
    report = build(args.sft_train, args.grpo_train, args.output_dir, tokenizer, args.max_length)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
