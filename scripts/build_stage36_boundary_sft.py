"""Package audited single-action replay anchors for boundary-weighted SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.sft_dataset import DatasetManifest, SFTExample  # noqa: E402

SPLITS = ("train", "validation", "test")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def build(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_manifest_path = source_dir.parent / "manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError(f"upstream audit manifest missing: {source_manifest_path}")
    upstream = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if upstream.get("status") != "passed":
        raise ValueError("upstream isolated-action dataset did not pass its audit")
    if upstream.get("frozen_holdout_payload_overlap") != 0:
        raise ValueError("upstream dataset overlaps the frozen holdout")

    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    visible_payloads: set[str] = set()
    scenario_splits: dict[str, set[str]] = {}
    source_hashes: dict[str, str] = {}
    quality_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    environments: set[str] = set()
    policies: set[str] = set()
    action_counts: Counter[str] = Counter()
    errors: list[str] = []

    for split in SPLITS:
        source_path = source_dir / f"{split}.jsonl"
        source_hashes[split] = _sha256(source_path)
        rows = _read_jsonl(source_path)
        rows_by_split[split] = rows
        for raw in rows:
            example = SFTExample(**raw)
            if example.split != split:
                errors.append(f"SPLIT_LABEL_MISMATCH:{example.example_id}")
            if example.example_id in all_ids:
                errors.append(f"DUPLICATE_EXAMPLE_ID:{example.example_id}")
            all_ids.add(example.example_id)
            scenario_splits.setdefault(example.scenario_id, set()).add(split)
            final = example.messages[-1]
            if final.role != "assistant" or len(final.tool_calls) != 1:
                errors.append(f"NOT_SINGLE_TOOL_CALL:{example.example_id}")
                continue
            action = final.tool_calls[0].function.name
            exposed = [tool["function"]["name"] for tool in example.tools]
            if exposed != [action]:
                errors.append(f"ACTION_NOT_ISOLATED:{example.example_id}")
            action_counts[action] += 1
            visible = _digest(
                {
                    "messages": [message.model_dump(mode="json") for message in example.messages],
                    "tools": example.tools,
                }
            )
            if visible in visible_payloads:
                errors.append(f"DUPLICATE_MODEL_PAYLOAD:{example.example_id}")
            visible_payloads.add(visible)
            quality_counts[example.quality_label] += 1
            source_counts[example.source] += 1
            environments.add(example.environment_version)
            policies.add(f"{example.policy_name}:{example.policy_version}")

    overlap = {
        scenario: sorted(splits)
        for scenario, splits in scenario_splits.items()
        if len(splits) > 1
    }
    if overlap:
        errors.append(f"SCENARIO_SPLIT_OVERLAP:{len(overlap)}")
    expected_counts = upstream.get("split_counts")
    actual_counts = {split: len(rows) for split, rows in rows_by_split.items()}
    if expected_counts != actual_counts:
        errors.append("UPSTREAM_SPLIT_COUNT_MISMATCH")
    if len(action_counts) < 2:
        errors.append("ACTION_DIVERSITY_BELOW_TWO")

    version = "stage36-boundary-sft-" + _digest(
        {"source_hashes": source_hashes, "ids": sorted(all_ids)}
    )[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows)

    standard_manifest = DatasetManifest(
        dataset_version=version,
        created_at=datetime.now(timezone.utc),
        candidate_episodes=len(all_ids),
        accepted_episodes=len(all_ids),
        rejected_episodes=0,
        exported_examples=len(all_ids),
        split_examples=actual_counts,
        source_episodes=dict(source_counts),
        quality_episodes=dict(quality_counts),
        rejection_codes={},
        environment_versions=sorted(environments),
        policy_versions=sorted(policies),
        split_group_overlap=bool(overlap),
    )
    (output_dir / "manifest.json").write_text(
        standard_manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    audit = {
        "schema_version": "stage36-boundary-sft-audit.v1",
        "status": "passed" if not errors else "rejected",
        "dataset_version": version,
        "objective": "upweight the single EOS immediately after the first </tool_call>",
        "source_dir": source_dir.as_posix(),
        "upstream_dataset_version": upstream.get("dataset_version"),
        "upstream_manifest_sha256": _sha256(source_manifest_path),
        "source_split_sha256": source_hashes,
        "split_counts": actual_counts,
        "action_counts": dict(sorted(action_counts.items())),
        "unique_examples": len(all_ids),
        "unique_model_visible_payloads": len(visible_payloads),
        "scenario_split_overlap": len(overlap),
        "frozen_holdout_payload_overlap": 0,
        "data_scope": "stage32 audited train replay only; no frozen evaluation labels",
        "errors": errors,
    }
    (output_dir / "boundary_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.source_dir, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
