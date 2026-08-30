"""Build leakage-safe student on-policy hard negatives from audited SFT prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


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
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _assistant_tool_call(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    assistant = row["messages"][-1]
    calls = assistant.get("tool_calls") or []
    if len(calls) != 1:
        raise ValueError(f"SFT row must have one chosen tool call: {row['example_id']}")
    function = calls[0]["function"]
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return str(function["name"]), dict(arguments)


def _clean_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in message.items()
        if value is not None and not (key == "tool_calls" and not value)
    }


def _original_request(messages: list[dict[str, Any]]) -> str:
    users = [item for item in messages if item.get("role") == "user"]
    if not users or not isinstance(users[-1].get("content"), str):
        return ""
    try:
        payload = json.loads(users[-1]["content"])
    except json.JSONDecodeError:
        return str(users[-1]["content"])
    return str(payload.get("original_request") or "")


def _forbidden_requests(paths: list[Path]) -> list[str]:
    requests = []
    for path in paths:
        for row in _read_jsonl(path):
            messages = row.get("messages") or []
            request = _original_request(messages)
            if request:
                requests.append(request)
    return requests


def build_cases(
    sft_file: Path,
    output_dir: Path,
    *,
    forbidden_files: list[Path],
    similarity_threshold: float,
) -> dict[str, Any]:
    source_rows = _read_jsonl(sft_file)
    forbidden = [_normalize(item) for item in _forbidden_requests(forbidden_files)]
    forbidden_exact = set(forbidden)
    forbidden_ngrams = [_ngrams(item) for item in forbidden]
    cases = []
    index_rows = []
    seen_payloads = set()
    rejections: Counter[str] = Counter()
    max_similarity = 0.0

    for row in source_rows:
        if row.get("split") != "train":
            rejections["NON_TRAIN_SOURCE"] += 1
            continue
        messages = [_clean_message(item) for item in row["messages"][:-1]]
        chosen_action, _ = _assistant_tool_call(row)
        tool_names = [item["function"]["name"] for item in row["tools"]]
        if chosen_action not in tool_names:
            rejections["CHOSEN_ACTION_NOT_EXPOSED"] += 1
            continue
        request = _normalize(_original_request(messages))
        request_grams = _ngrams(request)
        similarity = max(
            (_similarity(request_grams, item) for item in forbidden_ngrams),
            default=0.0,
        )
        max_similarity = max(max_similarity, similarity)
        if request in forbidden_exact:
            rejections["FROZEN_HOLDOUT_EXACT_OVERLAP"] += 1
            continue
        if similarity >= similarity_threshold:
            rejections["FROZEN_HOLDOUT_NEAR_OVERLAP"] += 1
            continue
        payload_hash = _hash({"messages": messages, "tools": row["tools"]})
        if payload_hash in seen_payloads:
            rejections["DUPLICATE_MODEL_PAYLOAD"] += 1
            continue
        seen_payloads.add(payload_hash)
        case_id = f"stage32-onpolicy-{payload_hash[:20]}"
        family = str(row.get("environment_version") or row.get("quality_label") or "unknown")
        cases.append(
            {
                "case_id": case_id,
                "messages": messages,
                "tools": row["tools"],
                "allowed_actions": tool_names,
                "expected_action": chosen_action,
                "expected_arguments": None,
                "family": family,
            }
        )
        index_rows.append(
            {
                "case_id": case_id,
                "example_id": row["example_id"],
                "scenario_id": row["scenario_id"],
                "trajectory_id": row["trajectory_id"],
                "family": family,
                "messages": messages,
                "tools": row["tools"],
                "chosen": _clean_message(row["messages"][-1]),
                "context_hash": payload_hash,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "cases.jsonl", cases)
    _write_jsonl(output_dir / "index.jsonl", index_rows)
    manifest = {
        "schema_version": "stage32-onpolicy-cases.v1",
        "status": "passed" if cases and forbidden else "rejected",
        "source_file": sft_file.as_posix(),
        "source_sha256": _file_sha256(sft_file),
        "source_rows": len(source_rows),
        "exported_cases": len(cases),
        "unique_model_payloads": len(seen_payloads),
        "action_counts": dict(Counter(item["expected_action"] for item in cases)),
        "family_counts": dict(Counter(item["family"] for item in cases)),
        "forbidden_files": [path.as_posix() for path in forbidden_files],
        "forbidden_requests": len(forbidden),
        "similarity_threshold": similarity_threshold,
        "maximum_observed_similarity": round(max_similarity, 8),
        "rejection_counts": dict(rejections),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _stable_split(context_hash: str) -> str:
    bucket = int(context_hash[:8], 16) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def build_preferences(
    index_file: Path, runs_file: Path, output_dir: Path
) -> dict[str, Any]:
    index = {row["case_id"]: row for row in _read_jsonl(index_file)}
    runs = _read_jsonl(runs_file)
    pairs_by_hash = {}
    rejections: Counter[str] = Counter()
    for run in runs:
        source = index.get(run["case_id"])
        if source is None:
            rejections["UNKNOWN_CASE"] += 1
            continue
        observed = run.get("observed_actions") or []
        if run.get("http_error"):
            rejections["HTTP_ERROR_NO_REPLAYABLE_NEGATIVE"] += 1
            continue
        if len(observed) != 1:
            rejections["NOT_EXACTLY_ONE_REJECTED_ACTION"] += 1
            continue
        expected = source["chosen"]["tool_calls"][0]["function"]["name"]
        if observed[0] == expected:
            rejections["STUDENT_ACTION_MATCHED_TEACHER"] += 1
            continue
        arguments = run.get("observed_arguments")
        if not isinstance(arguments, dict):
            rejections["REJECTED_ARGUMENTS_NOT_JSON_OBJECT"] += 1
            continue
        rejected = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": observed[0], "arguments": arguments},
                }
            ],
        }
        pair_hash = _hash(
            {
                "context_hash": source["context_hash"],
                "chosen": source["chosen"],
                "rejected": rejected,
            }
        )
        pairs_by_hash.setdefault(
            pair_hash,
            {
                "schema_version": "teacher-preference-pair.v1",
                "pair_id": f"stage32-onpolicy-pref-{pair_hash[:20]}",
                "task_id": source["scenario_id"],
                "family": source["family"],
                "context_hash": source["context_hash"],
                "messages": source["messages"],
                "tools": source["tools"],
                "chosen": source["chosen"],
                "rejected": rejected,
                "chosen_trajectory_id": source["trajectory_id"],
                "rejected_trajectory_id": (
                    f"student-onpolicy-{run['case_id']}-r{run['repetition']}"
                ),
                "reason_codes": [
                    "AUDITED_TEACHER_POLICY_SUCCESS",
                    "ON_POLICY_STUDENT_ACTION_MISMATCH",
                ],
                "reward_margin": 1.0,
            },
        )

    splits = {"train": [], "validation": [], "test": []}
    for pair in sorted(pairs_by_hash.values(), key=lambda item: item["pair_id"]):
        splits[_stable_split(pair["context_hash"])].append(pair)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows)
    manifest = {
        "schema_version": "stage32-onpolicy-preferences.v1",
        "status": "passed" if splits["train"] and splits["validation"] else "rejected",
        "index_file": index_file.as_posix(),
        "index_sha256": _file_sha256(index_file),
        "runs_file": runs_file.as_posix(),
        "runs_sha256": _file_sha256(runs_file),
        "student_rollouts": len(runs),
        "unique_pairs": len(pairs_by_hash),
        "unique_contexts": len({item["context_hash"] for item in pairs_by_hash.values()}),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "family_counts": dict(Counter(item["family"] for item in pairs_by_hash.values())),
        "rejection_counts": dict(rejections),
        "frozen_evaluation_rows_used": 0,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    cases = subparsers.add_parser("cases")
    cases.add_argument("--sft-file", type=Path, required=True)
    cases.add_argument("--output-dir", type=Path, required=True)
    cases.add_argument("--forbidden-file", type=Path, action="append", required=True)
    cases.add_argument("--similarity-threshold", type=float, default=0.82)
    preferences = subparsers.add_parser("preferences")
    preferences.add_argument("--index-file", type=Path, required=True)
    preferences.add_argument("--runs-file", type=Path, required=True)
    preferences.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "cases":
        manifest = build_cases(
            args.sft_file,
            args.output_dir,
            forbidden_files=args.forbidden_file,
            similarity_threshold=args.similarity_threshold,
        )
    else:
        manifest = build_preferences(args.index_file, args.runs_file, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
