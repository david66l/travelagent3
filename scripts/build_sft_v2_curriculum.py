"""Build an action-balanced, schema-conflict-aware policy SFT curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.sft_curriculum import (  # noqa: E402
    DECISION_ACTIONS,
    SPLITS,
    load_sft_dataset,
    minimize_row_context,
    policy_context,
    target_call,
)
from agentic.sft_dataset import DatasetManifest, SFTExample  # noqa: E402


def build(
    source_dir: Path,
    output_dir: Path,
    *,
    replay_per_action: int = 128,
    poi_minimal_examples: int = 256,
    poi_conflict_examples: int = 128,
) -> DatasetManifest:
    rows_by_split = load_sft_dataset(source_dir)
    train_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows_by_split["train"]:
        action, _ = target_call(row)
        train_groups[action].append(row)

    output: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    curriculum_buckets: Counter[str] = Counter()

    for action, rows in sorted(train_groups.items()):
        ordered = _stable_rows(rows)
        contexts = [policy_context(row) for row in ordered]
        is_decision_group = action in DECISION_ACTIONS
        has_recovery = any(context.get("failure_summary") for context in contexts)
        if is_decision_group or has_recovery:
            for row in ordered:
                output["train"].append(_prepare(row, "train", "decision", minimize=True))
                curriculum_buckets[f"decision:{action}"] += 1
            continue

        if action == "get_poi_detail":
            minimal = ordered[:poi_minimal_examples]
            remaining = ordered[poi_minimal_examples:]
            conflict_pool = remaining if len(remaining) >= poi_conflict_examples else ordered
            conflicts = conflict_pool[:poi_conflict_examples]
            for row in minimal:
                output["train"].append(_prepare(row, "train", "minimal", minimize=True))
                curriculum_buckets["minimal:get_poi_detail"] += 1
            for row in conflicts:
                # Keep the original controller-rich context as an adversarial
                # positive: the only target remains the schema-valid empty call.
                output["train"].append(_prepare(row, "train", "conflict", minimize=False))
                curriculum_buckets["conflict:get_poi_detail"] += 1
            continue

        for row in ordered[:replay_per_action]:
            output["train"].append(_prepare(row, "train", "replay", minimize=True))
            curriculum_buckets[f"replay:{action}"] += 1

    for split in ("validation", "test"):
        for row in rows_by_split[split]:
            # Preserve controller-rich held-out prompts as a harder schema-copy
            # challenge. Production-aligned sequential evaluation is performed
            # separately through the live prompt projection.
            output[split].append(_prepare(row, split, "challenge_eval", minimize=False))

    output["train"] = _deduplicate_model_payloads(output["train"])
    for split in SPLITS:
        output[split].sort(key=lambda row: str(row["example_id"]))
        _assert_unique_example_ids(output[split])

    scenario_splits = {
        split: {str(row.get("scenario_id") or "") for row in rows}
        for split, rows in output.items()
    }
    overlap = any(
        scenario_splits[left] & scenario_splits[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    if overlap:
        raise ValueError("SFT v2 curriculum split leakage detected")

    all_rows = [row for split in SPLITS for row in output[split]]
    version_payload = {
        split: [row["example_id"] for row in output[split]] for split in SPLITS
    }
    dataset_version = "sft-v2-" + _hash(version_payload)[:16]
    manifest = DatasetManifest(
        dataset_version=dataset_version,
        candidate_episodes=len(all_rows),
        accepted_episodes=len(all_rows),
        rejected_episodes=0,
        exported_examples=len(all_rows),
        split_examples={split: len(output[split]) for split in SPLITS},
        source_episodes=dict(Counter(str(row.get("source") or "unknown") for row in all_rows)),
        quality_episodes=dict(
            Counter(str(row.get("quality_label") or "unknown") for row in all_rows)
        ),
        rejection_codes={},
        environment_versions=sorted(
            {str(row.get("environment_version") or "unknown") for row in all_rows}
        ),
        policy_versions=sorted(
            {
                f"{row.get('policy_name', 'unknown')}:{row.get('policy_version', 'unknown')}"
                for row in all_rows
            }
        ),
        split_group_overlap=False,
        excluded_policy_steps=0,
    )

    challenge = [
        _prepare(row, str(row["split"]), "challenge", minimize=False)
        for split in ("validation", "test")
        for row in rows_by_split[split]
        if target_call(row)[0] == "get_poi_detail"
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _write_jsonl(output_dir / f"{split}.jsonl", output[split])
    _write_jsonl(output_dir / "challenge.jsonl", challenge)
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    derivation = {
        "schema_version": "agent-policy-sft-v2-derivation.v1",
        "source_dataset": str(source_dir),
        "source_manifest": json.loads((source_dir / "manifest.json").read_text(encoding="utf-8")),
        "selection": {
            "all_decision_and_recovery_rows_retained": True,
            "replay_per_deterministic_action": replay_per_action,
            "get_poi_detail_minimal_rows": poi_minimal_examples,
            "get_poi_detail_conflict_rows": poi_conflict_examples,
            "validation_and_test_training_authorized": False,
        },
        "curriculum_buckets": dict(curriculum_buckets),
        "challenge_rows": len(challenge),
        "challenge_used_for_training": False,
    }
    (output_dir / "derivation.json").write_text(
        json.dumps(derivation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _prepare(
    row: dict[str, Any],
    split: str,
    bucket: str,
    *,
    minimize: bool,
) -> dict[str, Any]:
    prepared = minimize_row_context(row) if minimize else json.loads(json.dumps(row))
    prepared["split"] = split
    prepared["example_id"] = f"sft-v2:{bucket}:{row['example_id']}"
    # Validate against the production training schema before writing anything.
    return SFTExample(**prepared).model_dump(mode="json")


def _stable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _hash(str(row.get("example_id") or "")))


def _assert_unique_example_ids(rows: list[dict[str, Any]]) -> None:
    ids = [str(row.get("example_id") or "") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example_id in SFT v2 curriculum")


def _deduplicate_model_payloads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one copy of an identical prompt, tool set and target completion."""
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        signature = _hash({"messages": row.get("messages"), "tools": row.get("tools")})
        unique.setdefault(signature, row)
    return list(unique.values())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "\n".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
    )
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def _hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-per-action", type=int, default=128)
    parser.add_argument("--poi-minimal-examples", type=int, default=256)
    parser.add_argument("--poi-conflict-examples", type=int, default=128)
    args = parser.parse_args()
    manifest = build(
        args.source_dir,
        args.output_dir,
        replay_per_action=args.replay_per_action,
        poi_minimal_examples=args.poi_minimal_examples,
        poi_conflict_examples=args.poi_conflict_examples,
    )
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
