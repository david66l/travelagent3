"""Build single-call preferences under controller-isolated action schemas."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_stage35_single_action_preferences import (  # noqa: E402
    ACTIONS,
    SPLITS,
    _assistant,
    _build_pair,
    _file_sha256,
    _forbidden_requests,
    _hash,
    _ngrams,
    _normalize,
    _original_request,
    _quotas,
    _read_jsonl,
    _similarity,
    _target_split,
    _write_jsonl,
)


def isolate_action(row: dict[str, Any]) -> dict[str, Any]:
    isolated = copy.deepcopy(row)
    action, _ = _assistant(isolated)
    user_index = next(
        index
        for index in range(len(isolated["messages"]) - 2, -1, -1)
        if isolated["messages"][index].get("role") == "user"
    )
    payload = json.loads(isolated["messages"][user_index]["content"])
    payload["allowed_actions"] = [action]
    if isinstance(payload.get("current_subtask"), dict):
        payload["current_subtask"]["allowed_actions"] = [action]
    isolated["messages"][user_index]["content"] = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    isolated["tools"] = [
        tool for tool in isolated["tools"] if tool["function"]["name"] == action
    ]
    if len(isolated["tools"]) != 1:
        raise ValueError(f"isolated action must expose one tool: {row['example_id']}")
    isolated["example_id"] = f"stage35-isolated:{row['example_id']}"
    isolated["environment_version"] = (
        f"{row.get('environment_version', 'unknown')}+isolated-action.v1"
    )
    return isolated


def _excluded_contexts(preference_dir: Path | None) -> set[str]:
    if preference_dir is None:
        return set()
    return {
        str(row["context_hash"])
        for split in SPLITS
        for row in _read_jsonl(preference_dir / f"{split}.jsonl")
    }


def build(
    source_file: Path,
    output_dir: Path,
    *,
    forbidden_files: list[Path],
    excluded_preference_dir: Path | None,
    per_action: int = 60,
    similarity_threshold: float = 0.82,
) -> dict[str, Any]:
    forbidden = [_normalize(item) for item in _forbidden_requests(forbidden_files)]
    if not forbidden:
        raise ValueError("frozen evaluation files contain no auditable user requests")
    forbidden_exact = set(forbidden)
    forbidden_ngrams = [_ngrams(item) for item in forbidden]
    excluded = _excluded_contexts(excluded_preference_dir)
    candidates: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {
        (split, action): [] for split in SPLITS for action in ACTIONS
    }
    seen: set[str] = set()
    rejections: Counter[str] = Counter()
    maximum_similarity = 0.0

    for source in _read_jsonl(source_file):
        if source.get("split") != "train":
            continue
        try:
            row = isolate_action(source)
            action, _ = _assistant(row)
        except (KeyError, StopIteration, ValueError, json.JSONDecodeError):
            rejections["INVALID_SOURCE"] += 1
            continue
        messages = row["messages"][:-1]
        context_hash = _hash({"messages": messages, "tools": row["tools"]})
        if context_hash in excluded:
            rejections["EXCLUDED_EXISTING_CONTEXT"] += 1
            continue
        if context_hash in seen:
            rejections["DUPLICATE_CONTEXT"] += 1
            continue
        request = _normalize(_original_request(messages))
        similarity = max(
            (_similarity(_ngrams(request), item) for item in forbidden_ngrams),
            default=0.0,
        )
        maximum_similarity = max(maximum_similarity, similarity)
        if request and request in forbidden_exact:
            rejections["FROZEN_EXACT_OVERLAP"] += 1
            continue
        if similarity >= similarity_threshold:
            rejections["FROZEN_NEAR_OVERLAP"] += 1
            continue
        seen.add(context_hash)
        candidates[(_target_split(str(row["scenario_id"])), action)].append(
            (context_hash, row)
        )

    quotas = _quotas(per_action)
    preferences: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    sft_replay: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for action in ACTIONS:
            available = sorted(
                candidates[(split, action)], key=lambda item: _hash(item[0])
            )
            required = quotas[split]
            if len(available) < required:
                raise ValueError(
                    f"insufficient isolated {split}/{action}: {len(available)} < {required}"
                )
            for context_hash, row in available[:required]:
                pair = _build_pair(row, context_hash)
                pair["family"] = f"isolated_action_{action}"
                pair["task_id"] = f"{row['scenario_id']}:isolated:{action}"
                preferences[split].append(pair)
                anchor = copy.deepcopy(row)
                anchor["split"] = split
                sft_replay[split].append(anchor)
        preferences[split].sort(key=lambda item: item["pair_id"])
        sft_replay[split].sort(key=lambda item: item["example_id"])

    all_pairs = [row for split in SPLITS for row in preferences[split]]
    contexts = [row["context_hash"] for row in all_pairs]
    scenarios = {
        split: {row["task_id"].split(":isolated:", 1)[0] for row in rows}
        for split, rows in preferences.items()
    }
    overlap = any(
        scenarios[left] & scenarios[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    )
    errors = []
    if len(contexts) != len(set(contexts)):
        errors.append("DUPLICATE_CONTEXT")
    if overlap:
        errors.append("SCENARIO_SPLIT_OVERLAP")
    for pair in all_pairs:
        payload = json.loads(pair["messages"][-1]["content"])
        tool_names = [tool["function"]["name"] for tool in pair["tools"]]
        chosen_name = pair["chosen"]["tool_calls"][0]["function"]["name"]
        if payload.get("allowed_actions") != [chosen_name] or tool_names != [
            chosen_name
        ]:
            errors.append(f"ACTION_ISOLATION_FAILED:{pair['pair_id']}")

    preference_dir = output_dir / "preferences"
    sft_dir = output_dir / "sft_replay"
    preference_dir.mkdir(parents=True, exist_ok=True)
    sft_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _write_jsonl(preference_dir / f"{split}.jsonl", preferences[split])
        _write_jsonl(sft_dir / f"{split}.jsonl", sft_replay[split])
    version = (
        "stage35-isolated-action-"
        + _hash(
            {split: [row["pair_id"] for row in preferences[split]] for split in SPLITS}
        )[:16]
    )
    common = {
        "status": "passed" if not errors else "rejected",
        "dataset_version": version,
        "split_counts": {split: len(rows) for split, rows in preferences.items()},
        "family_counts": dict(Counter(row["family"] for row in all_pairs)),
        "unique_pairs": len({row["pair_id"] for row in all_pairs}),
        "unique_contexts": len(set(contexts)),
        "context_split_overlap": int(overlap),
        "frozen_holdout_payload_overlap": 0,
        "errors": errors,
    }
    training_manifest = {
        "schema_version": "stage35-isolated-action-training-preferences.v1",
        **common,
        "requires_verifier_success_over_failure": False,
        "preference_evidence_policy": (
            "verifier_success_or_deterministic_single_action_contract"
        ),
        "run_scope_constraint": "targeted_smoke_only_until_generation_gate_passes",
    }
    (preference_dir / "manifest.json").write_text(
        json.dumps(training_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "stage35-isolated-action-preferences.v1",
        **common,
        "source_file": source_file.as_posix(),
        "source_sha256": _file_sha256(source_file),
        "excluded_preference_dir": (
            excluded_preference_dir.as_posix() if excluded_preference_dir else None
        ),
        "excluded_contexts": len(excluded),
        "derivation": (
            "retain grounded request/evidence/arguments; reduce allowed_actions and tools "
            "to the verifier-approved chosen action"
        ),
        "per_action": per_action,
        "maximum_source_similarity": round(maximum_similarity, 8),
        "accepted_frozen_exact_overlap": 0,
        "accepted_frozen_near_overlap": 0,
        "rejection_counts": dict(rejections),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-file", type=Path, action="append", required=True)
    parser.add_argument("--exclude-preferences", type=Path)
    parser.add_argument("--per-action", type=int, default=60)
    parser.add_argument("--similarity-threshold", type=float, default=0.82)
    args = parser.parse_args()
    manifest = build(
        args.source_file,
        args.output_dir,
        forbidden_files=args.forbidden_file,
        excluded_preference_dir=args.exclude_preferences,
        per_action=args.per_action,
        similarity_threshold=args.similarity_threshold,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
