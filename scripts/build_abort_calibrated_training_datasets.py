"""Merge a bounded necessary-abort repair shard into the qualified SFT/DPO baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agentic.distillation import TeacherPreferencePair  # noqa: E402
from agentic.sft_dataset import DatasetManifest, SFTExample  # noqa: E402
from generate_teacher_distillation import load_holdout_contract, model_payload_hash  # noqa: E402

SPLITS = ("train", "validation", "test")
DEFAULT_REPAIR_LIMITS = {"train": 24, "validation": 4, "test": 4}


def _read(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_request(value: str) -> str:
    return re.sub(r"[^\w]+", "", value.casefold())


def _example_request(example: SFTExample) -> str:
    for message in example.messages:
        if message.role != "user" or not message.content:
            continue
        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError:
            continue
        request = str(payload.get("original_request") or "").strip()
        if request:
            return request
    return ""


def _audit_grpo_holdout_pairs(
    training_pairs: list[tuple[str, str]],
    holdout_file: Path | None,
) -> dict[str, Any]:
    if holdout_file is None:
        return {
            "holdout_file": None,
            "holdout_tasks": 0,
            "training_pairs": len(training_pairs),
            "training_requests_present": sum(bool(request) for _, request in training_pairs),
            "training_requests_missing": sum(not request for _, request in training_pairs),
            "task_id_overlap": 0,
            "exact_normalized_request_overlap": 0,
            "max_sequence_similarity": 0.0,
            "closest_pair": None,
            "passed": True,
        }
    holdout_pairs: list[tuple[str, str]] = []
    for line in holdout_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = row["task"]
        holdout_pairs.append((str(task["task_id"]), str(task["user_request"])))

    training_ids = {task_id for task_id, _ in training_pairs}
    holdout_ids = {task_id for task_id, _ in holdout_pairs}
    task_overlap = sorted(training_ids & holdout_ids)
    normalized_holdout = {_normalize_request(request) for _, request in holdout_pairs}
    exact_requests = sorted(
        {
            request
            for _, request in training_pairs
            if request and _normalize_request(request) in normalized_holdout
        }
    )
    maximum = 0.0
    closest: dict[str, str] | None = None
    for _, training_request in training_pairs:
        if not training_request:
            continue
        normalized_training = _normalize_request(training_request)
        for _, holdout_request in holdout_pairs:
            ratio = SequenceMatcher(
                None,
                normalized_training,
                _normalize_request(holdout_request),
                autojunk=False,
            ).ratio()
            if ratio > maximum:
                maximum = ratio
                closest = {
                    "training": training_request,
                    "holdout": holdout_request,
                }
    return {
        "holdout_file": str(holdout_file),
        "holdout_file_sha256": _file_sha256(holdout_file),
        "holdout_tasks": len(holdout_pairs),
        "training_pairs": len(training_pairs),
        "training_requests_present": sum(bool(request) for _, request in training_pairs),
        "training_requests_missing": sum(not request for _, request in training_pairs),
        "task_id_overlap": len(task_overlap),
        "task_id_overlap_examples": task_overlap[:10],
        "exact_normalized_request_overlap": len(exact_requests),
        "exact_normalized_request_examples": exact_requests[:10],
        "max_sequence_similarity": round(maximum, 6),
        "closest_pair": closest,
        "passed": not task_overlap and not exact_requests,
    }


def _action(example: SFTExample) -> str:
    calls = example.messages[-1].tool_calls
    if len(calls) != 1:
        raise ValueError(f"SFT example must have exactly one tool call: {example.example_id}")
    return calls[0].function.name


def _select_sft(path: Path, limit: int) -> list[SFTExample]:
    candidates = [SFTExample(**row) for row in _read(path)]
    candidates = [item for item in candidates if _action(item) == "abort"]
    if len(candidates) < limit:
        raise ValueError(f"insufficient abort SFT examples in {path}: {len(candidates)}<{limit}")
    return _stratified_select(
        candidates,
        limit,
        group=lambda item: item.scenario_id.rsplit("necessary-abort-", 1)[-1],
        identity=lambda item: item.example_id,
    )


def _select_preferences(path: Path, limit: int) -> list[TeacherPreferencePair]:
    candidates = [TeacherPreferencePair(**row) for row in _read(path)]
    candidates = [
        item
        for item in candidates
        if (item.chosen.get("tool_calls") or [{}])[0].get("function", {}).get("name")
        == "abort"
    ]
    if len(candidates) < limit:
        raise ValueError(f"insufficient abort preferences in {path}: {len(candidates)}<{limit}")
    return _stratified_select(
        candidates,
        limit,
        group=lambda item: item.family,
        identity=lambda item: item.pair_id,
    )


def _stratified_select(items, limit: int, *, group, identity):
    buckets: dict[str, list[Any]] = {}
    for item in items:
        buckets.setdefault(str(group(item)), []).append(item)
    for values in buckets.values():
        values.sort(key=lambda item: _stable_id(str(identity(item))))
    selected = []
    cursor = 0
    names = sorted(buckets)
    while len(selected) < limit:
        progressed = False
        for name in names:
            if cursor < len(buckets[name]) and len(selected) < limit:
                selected.append(buckets[name][cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    if len(selected) != limit:
        raise ValueError(f"stratified selection produced {len(selected)}<{limit}")
    return selected


def build_sft(
    base_dir: Path,
    repair_dir: Path,
    output_dir: Path,
    *,
    limits: dict[str, int],
    forbidden_holdout_dir: Path | None,
    forbidden_grpo_file: Path | None = None,
) -> dict[str, Any]:
    splits: dict[str, list[SFTExample]] = {}
    selected_repair_rows: list[SFTExample] = []
    selected_counts: dict[str, int] = {}
    for split in SPLITS:
        base = [SFTExample(**row) for row in _read(base_dir / f"{split}.jsonl")]
        repair = _select_sft(repair_dir / f"{split}.jsonl", limits[split])
        selected_repair_rows.extend(repair)
        selected_counts[split] = len(repair)
        splits[split] = sorted([*base, *repair], key=lambda item: item.example_id)

    all_rows = [item for split in SPLITS for item in splits[split]]
    ids = [item.example_id for item in all_rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate SFT example IDs after abort repair merge")
    split_scenarios = {
        split: {item.scenario_id for item in splits[split]} for split in SPLITS
    }
    if any(
        split_scenarios[left] & split_scenarios[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("SFT scenario split leakage after abort repair merge")

    prompts = [
        model_payload_hash(
            [message.model_dump(mode="json") for message in item.messages[:-1]], item.tools
        )
        for item in all_rows
    ]
    if len(prompts) != len(set(prompts)):
        raise ValueError("duplicate model-visible SFT prompts after abort repair merge")
    _, holdout = load_holdout_contract(forbidden_holdout_dir)
    overlap = set(prompts) & holdout
    if overlap:
        raise ValueError(f"SFT frozen holdout overlap: {len(overlap)}")
    grpo_contamination = _audit_grpo_holdout_pairs(
        [(item.scenario_id, _example_request(item)) for item in all_rows],
        forbidden_grpo_file,
    )
    repair_grpo_contamination = _audit_grpo_holdout_pairs(
        [(item.scenario_id, _example_request(item)) for item in selected_repair_rows],
        forbidden_grpo_file,
    )
    if not grpo_contamination["passed"] or not repair_grpo_contamination["passed"]:
        raise ValueError(
            "SFT GRPO holdout contamination: "
            f"task_ids={grpo_contamination['task_id_overlap']}, "
            "requests="
            f"{grpo_contamination['exact_normalized_request_overlap']}"
        )

    version = "sft-abort-calibrated-" + _stable_id(
        json.dumps({split: [item.example_id for item in splits[split]] for split in SPLITS})
    )[:16]
    manifest = DatasetManifest(
        dataset_version=version,
        candidate_episodes=len(all_rows),
        accepted_episodes=len(all_rows),
        rejected_episodes=0,
        exported_examples=len(all_rows),
        split_examples={split: len(splits[split]) for split in SPLITS},
        source_episodes=dict(Counter(item.source for item in all_rows)),
        quality_episodes=dict(Counter(item.quality_label for item in all_rows)),
        rejection_codes={},
        environment_versions=sorted({item.environment_version for item in all_rows}),
        policy_versions=sorted(
            {f"{item.policy_name}:{item.policy_version}" for item in all_rows}
        ),
        split_group_overlap=False,
        excluded_policy_steps=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _write_jsonl(
            output_dir / f"{split}.jsonl",
            [item.model_dump(mode="json") for item in splits[split]],
        )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    derivation = {
        "schema_version": "abort-calibrated-sft-derivation.v1",
        "status": "passed",
        "dataset_version": version,
        "base_dir": str(base_dir),
        "base_manifest_sha256": _file_sha256(base_dir / "manifest.json"),
        "repair_dir": str(repair_dir),
        "repair_manifest_sha256": _file_sha256(repair_dir / "manifest.json"),
        "selected_repair_counts": selected_counts,
        "selected_repair_kind_counts": dict(
            Counter(
                item.scenario_id.rsplit("necessary-abort-", 1)[-1]
                for split in SPLITS
                for item in splits[split]
                if item.environment_version == "necessary-abort-decision.v1"
            )
        ),
        "action_counts": dict(Counter(_action(item) for item in all_rows)),
        "train_action_counts": dict(Counter(_action(item) for item in splits["train"])),
        "unique_model_visible_prompts": len(set(prompts)),
        "frozen_holdout_payload_overlap": 0,
        "merged_grpo_holdout_contamination_audit": grpo_contamination,
        "repair_grpo_holdout_contamination_audit": repair_grpo_contamination,
        "scenario_split_overlap": 0,
    }
    (output_dir / "derivation.json").write_text(
        json.dumps(derivation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return derivation


def build_preferences(
    base_dir: Path,
    repair_dir: Path,
    output_dir: Path,
    *,
    limits: dict[str, int],
    forbidden_holdout_dir: Path | None,
) -> dict[str, Any]:
    base_manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    if base_manifest.get("status") != "passed" or not base_manifest.get(
        "requires_verifier_success_over_failure"
    ):
        raise ValueError("base preference dataset is not verifier-qualified")
    splits: dict[str, list[TeacherPreferencePair]] = {}
    selected_counts: dict[str, int] = {}
    for split in SPLITS:
        base = [TeacherPreferencePair(**row) for row in _read(base_dir / f"{split}.jsonl")]
        repair = _select_preferences(repair_dir / f"{split}.jsonl", limits[split])
        selected_counts[split] = len(repair)
        splits[split] = sorted([*base, *repair], key=lambda item: item.pair_id)

    all_rows = [item for split in SPLITS for item in splits[split]]
    pair_ids = [item.pair_id for item in all_rows]
    contexts = [item.context_hash for item in all_rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate DPO pair IDs after abort repair merge")
    if len(contexts) != len(set(contexts)):
        raise ValueError("duplicate DPO contexts after abort repair merge")
    if any("VERIFIER_SUCCESS_OVER_FAILURE" not in item.reason_codes for item in all_rows):
        raise ValueError("non-verifier preference entered abort repair merge")

    _, holdout = load_holdout_contract(forbidden_holdout_dir)
    prompts = [model_payload_hash(item.messages, item.tools) for item in all_rows]
    overlap = set(prompts) & holdout
    if overlap:
        raise ValueError(f"DPO frozen holdout overlap: {len(overlap)}")
    version = "preference-abort-calibrated-" + _stable_id(
        json.dumps({split: [item.pair_id for item in splits[split]] for split in SPLITS})
    )[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _write_jsonl(
            output_dir / f"{split}.jsonl",
            [item.model_dump(mode="json") for item in splits[split]],
        )
    family_counts = Counter(item.family for item in all_rows)
    manifest = {
        "schema_version": "balanced-preference-dataset.v1",
        "status": "passed",
        "dataset_version": version,
        "base_dir": str(base_dir),
        "base_manifest_sha256": _file_sha256(base_dir / "manifest.json"),
        "repair_dir": str(repair_dir),
        "repair_manifest_sha256": _file_sha256(repair_dir / "manifest.json"),
        "requires_verifier_success_over_failure": True,
        "selected_repair_counts": selected_counts,
        "selected_repair_family_counts": dict(
            Counter(
                item.family
                for split in SPLITS
                for item in splits[split]
                if item.family.startswith("necessary_abort_")
            )
        ),
        "split_counts": {split: len(splits[split]) for split in SPLITS},
        "family_counts": dict(family_counts),
        "train_family_counts": dict(Counter(item.family for item in splits["train"])),
        "unique_pairs": len(set(pair_ids)),
        "unique_contexts": len(set(contexts)),
        "context_split_overlap": 0,
        "frozen_holdout_payload_overlap": 0,
        "errors": [],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build(args: argparse.Namespace) -> dict[str, Any]:
    limits = {
        "train": args.train_repair_examples,
        "validation": args.validation_repair_examples,
        "test": args.test_repair_examples,
    }
    sft = build_sft(
        args.base_sft_dir,
        args.repair_root / "sft",
        args.output_sft_dir,
        limits=limits,
        forbidden_holdout_dir=args.forbidden_holdout_dir,
        forbidden_grpo_file=args.forbidden_grpo_file,
    )
    if args.sft_only:
        return {"status": "passed", "limits": limits, "sft": sft}
    if args.base_preference_dir is None or args.output_preference_dir is None:
        raise ValueError("preference directories are required unless --sft-only is set")
    preferences = build_preferences(
        args.base_preference_dir,
        args.repair_root / "preferences",
        args.output_preference_dir,
        limits=limits,
        forbidden_holdout_dir=args.forbidden_holdout_dir,
    )
    return {"status": "passed", "limits": limits, "sft": sft, "preferences": preferences}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sft-dir", type=Path, required=True)
    parser.add_argument("--base-preference-dir", type=Path)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--output-sft-dir", type=Path, required=True)
    parser.add_argument("--output-preference-dir", type=Path)
    parser.add_argument("--forbidden-holdout-dir", type=Path)
    parser.add_argument(
        "--forbidden-grpo-file",
        type=Path,
        help="Frozen GRPO JSONL whose task IDs and user requests must not enter SFT.",
    )
    parser.add_argument(
        "--sft-only",
        action="store_true",
        help="Build only the SFT dataset; preference paths are then optional.",
    )
    parser.add_argument("--train-repair-examples", type=int, default=24)
    parser.add_argument("--validation-repair-examples", type=int, default=4)
    parser.add_argument("--test-repair-examples", type=int, default=4)
    args = parser.parse_args()
    if min(
        args.train_repair_examples,
        args.validation_repair_examples,
        args.test_repair_examples,
    ) < 1:
        parser.error("all repair split limits must be positive")
    if not args.sft_only and (
        args.base_preference_dir is None or args.output_preference_dir is None
    ):
        parser.error(
            "--base-preference-dir and --output-preference-dir are required "
            "unless --sft-only is set"
        )
    print(json.dumps(build(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
