"""Build leakage-safe preferences that penalize repeated tool calls.

The positive response is an audited, grounded single tool call from the Stage32
training split. The synthetic negative repeats that exact call twice. This
isolates the production failure mode without inventing a different action or
ungrounded arguments and without consuming frozen evaluation prompts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ACTIONS = ("search_pois", "ask_user", "propose_tradeoff", "abort")
SPLITS = ("train", "validation", "test")
# Fixed after checking action-family availability only; no model outputs or
# evaluation labels were used to choose the stratification salt.
SPLIT_SALT = "stage35-single-action-contract-v1-13:"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)


def _ngrams(text: str, width: int = 4) -> set[str]:
    if len(text) <= width:
        return {text} if text else set()
    return {text[index : index + width] for index in range(len(text) - width + 1)}


def _similarity(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _original_request(messages: list[dict[str, Any]]) -> str:
    users = [message for message in messages if message.get("role") == "user"]
    if not users or not isinstance(users[-1].get("content"), str):
        return ""
    content = str(users[-1]["content"])
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content
    return str(payload.get("original_request") or "")


def _forbidden_requests(paths: list[Path]) -> list[str]:
    requests = []
    for path in paths:
        for row in _read_jsonl(path):
            request = _original_request(row.get("messages") or [])
            if request:
                requests.append(request)
    return requests


def _target_split(scenario_id: str) -> str:
    bucket = int(
        hashlib.sha256((SPLIT_SALT + scenario_id).encode()).hexdigest()[:8], 16
    )
    bucket %= 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _assistant(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    response = copy.deepcopy(row["messages"][-1])
    calls = response.get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError(f"source must contain one tool call: {row.get('example_id')}")
    action = str(calls[0]["function"]["name"])
    if action not in ACTIONS:
        raise ValueError(f"unsupported Stage35 action: {action}")
    if action not in {str(tool["function"]["name"]) for tool in row.get("tools") or []}:
        raise ValueError(f"chosen action is not exposed: {row.get('example_id')}")
    return action, response


def _build_pair(row: dict[str, Any], context_hash: str) -> dict[str, Any]:
    action, chosen = _assistant(row)
    rejected = copy.deepcopy(chosen)
    rejected["tool_calls"] = [
        copy.deepcopy(chosen["tool_calls"][0]),
        copy.deepcopy(chosen["tool_calls"][0]),
    ]
    pair_hash = _hash(
        {
            "context_hash": context_hash,
            "chosen": chosen,
            "rejected": rejected,
        }
    )
    return {
        "schema_version": "teacher-preference-pair.v1",
        "pair_id": f"stage35-single-action-{pair_hash[:20]}",
        "task_id": row["scenario_id"],
        "family": f"single_action_{action}",
        "context_hash": context_hash,
        "messages": copy.deepcopy(row["messages"][:-1]),
        "tools": copy.deepcopy(row["tools"]),
        "chosen": chosen,
        "rejected": rejected,
        "chosen_trajectory_id": row["trajectory_id"],
        "rejected_trajectory_id": f"synthetic-duplicate-{context_hash[:20]}",
        "reason_codes": [
            "SINGLE_ACTION_CONTRACT_OVER_DUPLICATE_CALL",
            "FEWER_TOOL_CALLS",
        ],
        "reward_margin": 1.0,
    }


def _quotas(per_action: int) -> dict[str, int]:
    if per_action < 3:
        raise ValueError("per_action must be at least 3")
    validation = max(1, round(per_action * 0.1))
    test = max(1, round(per_action * 0.1))
    return {
        "train": per_action - validation - test,
        "validation": validation,
        "test": test,
    }


def build(
    source_file: Path,
    output_dir: Path,
    *,
    forbidden_files: list[Path],
    per_action: int = 60,
    similarity_threshold: float = 0.82,
) -> dict[str, Any]:
    if not forbidden_files:
        raise ValueError("at least one frozen evaluation file is required")
    source_rows = _read_jsonl(source_file)
    forbidden_text = [_normalize(item) for item in _forbidden_requests(forbidden_files)]
    if not forbidden_text:
        raise ValueError("frozen evaluation files contain no auditable user requests")
    forbidden_exact = set(forbidden_text)
    forbidden_ngrams = [_ngrams(item) for item in forbidden_text]
    candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {
        (split, action): [] for split in SPLITS for action in ACTIONS
    }
    seen_contexts: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    maximum_similarity = 0.0

    for row in source_rows:
        if row.get("split") != "train":
            rejection_counts["NON_TRAIN_SOURCE"] += 1
            continue
        try:
            action, _ = _assistant(row)
        except ValueError:
            rejection_counts["INVALID_SOURCE_RESPONSE"] += 1
            continue
        messages = row["messages"][:-1]
        context_hash = _hash({"messages": messages, "tools": row["tools"]})
        if context_hash in seen_contexts:
            rejection_counts["DUPLICATE_MODEL_CONTEXT"] += 1
            continue
        request = _normalize(_original_request(messages))
        similarity = max(
            (_similarity(_ngrams(request), grams) for grams in forbidden_ngrams),
            default=0.0,
        )
        maximum_similarity = max(maximum_similarity, similarity)
        if request and request in forbidden_exact:
            rejection_counts["FROZEN_EXACT_OVERLAP"] += 1
            continue
        if similarity >= similarity_threshold:
            rejection_counts["FROZEN_NEAR_OVERLAP"] += 1
            continue
        seen_contexts.add(context_hash)
        split = _target_split(str(row["scenario_id"]))
        candidates[(split, action)].append((context_hash, row))

    quotas = _quotas(per_action)
    selected: dict[str, list[tuple[str, dict[str, Any]]]] = {
        split: [] for split in SPLITS
    }
    for split in SPLITS:
        for action in ACTIONS:
            available = sorted(
                candidates[(split, action)], key=lambda item: _hash(item[0])
            )
            required = quotas[split]
            if len(available) < required:
                raise ValueError(
                    f"insufficient {split}/{action} rows: {len(available)} < {required}"
                )
            selected[split].extend(available[:required])

    preferences: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    sft_replay: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for context_hash, row in sorted(selected[split], key=lambda item: item[0]):
            pair = _build_pair(row, context_hash)
            preferences[split].append(pair)
            anchor = copy.deepcopy(row)
            anchor["split"] = split
            anchor["example_id"] = f"stage35-anchor:{row['example_id']}"
            sft_replay[split].append(anchor)

    all_pairs = [pair for split in SPLITS for pair in preferences[split]]
    pair_ids = [pair["pair_id"] for pair in all_pairs]
    context_hashes = [pair["context_hash"] for pair in all_pairs]
    scenario_sets = {
        split: {pair["task_id"] for pair in preferences[split]} for split in SPLITS
    }
    group_overlap = any(
        scenario_sets[left] & scenario_sets[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )
    errors = []
    if len(pair_ids) != len(set(pair_ids)):
        errors.append("DUPLICATE_PAIR_ID")
    if len(context_hashes) != len(set(context_hashes)):
        errors.append("DUPLICATE_CONTEXT")
    if group_overlap:
        errors.append("SCENARIO_SPLIT_OVERLAP")
    if any(
        len(pair["chosen"]["tool_calls"]) != 1
        or len(pair["rejected"]["tool_calls"]) != 2
        or pair["rejected"]["tool_calls"][0] != pair["rejected"]["tool_calls"][1]
        for pair in all_pairs
    ):
        errors.append("INVALID_DUPLICATE_NEGATIVE")

    preference_dir = output_dir / "preferences"
    sft_dir = output_dir / "sft_replay"
    preference_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _write_jsonl(preference_dir / f"{split}.jsonl", preferences[split])
        _write_jsonl(sft_dir / f"{split}.jsonl", sft_replay[split])
    version = (
        "stage35-single-action-"
        + _hash(
            {
                split: [pair["pair_id"] for pair in preferences[split]]
                for split in SPLITS
            }
        )[:16]
    )
    manifest = {
        "schema_version": "stage35-single-action-preferences.v1",
        "status": "passed" if not errors else "rejected",
        "dataset_version": version,
        "source_file": source_file.as_posix(),
        "source_sha256": _file_sha256(source_file),
        "source_split": "train_only",
        "split_salt": SPLIT_SALT,
        "split_salt_selection": "action-availability-only; no model outputs",
        "negative_construction": "repeat_the_same_grounded_tool_call_exactly_twice",
        "sft_replay_role": "balanced_single-call_anchor; no new factual supervision",
        "preference_role": "explicit single-call-over-duplicate-call ordering",
        "per_action": per_action,
        "preference_pairs": len(all_pairs),
        "sft_replay_examples": sum(len(items) for items in sft_replay.values()),
        "split_counts": {split: len(preferences[split]) for split in SPLITS},
        "action_counts": dict(Counter(pair["family"] for pair in all_pairs)),
        "unique_pairs": len(set(pair_ids)),
        "unique_contexts": len(set(context_hashes)),
        "scenario_split_overlap": group_overlap,
        "forbidden_files": [
            {"path": path.as_posix(), "sha256": _file_sha256(path)}
            for path in forbidden_files
        ],
        "forbidden_requests": len(forbidden_text),
        "similarity_threshold": similarity_threshold,
        "maximum_source_similarity": round(maximum_similarity, 8),
        "accepted_frozen_exact_overlap": 0,
        "accepted_frozen_near_overlap": 0,
        "rejection_counts": dict(rejection_counts),
        "errors": errors,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    training_manifest = {
        "schema_version": "stage35-single-action-training-preferences.v1",
        "status": manifest["status"],
        "dataset_version": version,
        "requires_verifier_success_over_failure": False,
        "preference_evidence_policy": (
            "verifier_success_or_deterministic_single_action_contract"
        ),
        "split_counts": manifest["split_counts"],
        "family_counts": manifest["action_counts"],
        "unique_pairs": manifest["unique_pairs"],
        "unique_contexts": manifest["unique_contexts"],
        "context_split_overlap": 0,
        "frozen_holdout_payload_overlap": 0,
        "run_scope_constraint": "targeted_smoke_only_until_broad_regression_passes",
        "errors": errors,
    }
    (preference_dir / "manifest.json").write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-file", type=Path, action="append", required=True)
    parser.add_argument("--per-action", type=int, default=60)
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    args = parser.parse_args()
    manifest = build(
        args.source_file,
        args.output_dir,
        forbidden_files=args.forbidden_file,
        per_action=args.per_action,
        similarity_threshold=args.similarity_threshold,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
