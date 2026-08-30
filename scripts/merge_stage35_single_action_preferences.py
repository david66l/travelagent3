"""Merge verified preferences with deterministic single-action contract pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")
EVIDENCE_POLICY = "verifier_success_or_deterministic_single_action_contract"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _is_contract_pair(row: dict[str, Any]) -> bool:
    chosen = (row.get("chosen") or {}).get("tool_calls") or []
    rejected = (row.get("rejected") or {}).get("tool_calls") or []
    return (
        "SINGLE_ACTION_CONTRACT_OVER_DUPLICATE_CALL" in row.get("reason_codes", [])
        and len(chosen) == 1
        and len(rejected) == 2
        and rejected[0] == chosen[0]
        and rejected[1] == chosen[0]
    )


def merge(base_dir: Path, target_root: Path, output_dir: Path) -> dict[str, Any]:
    base_manifest_path = base_dir / "manifest.json"
    target_manifest_path = target_root / "manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    target_manifest = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    if base_manifest.get("status") != "passed" or not base_manifest.get(
        "requires_verifier_success_over_failure"
    ):
        raise ValueError("base preferences are not verifier-qualified")
    if target_manifest.get("status") != "passed":
        raise ValueError("single-action preferences did not pass their manifest gate")
    if (
        target_manifest.get("accepted_frozen_exact_overlap") != 0
        or target_manifest.get("accepted_frozen_near_overlap") != 0
    ):
        raise ValueError("single-action preferences overlap frozen evaluation data")

    target_dir = target_root / "preferences"
    merged: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    source_counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for split in SPLITS:
        base_rows = _read_jsonl(base_dir / f"{split}.jsonl")
        target_rows = _read_jsonl(target_dir / f"{split}.jsonl")
        if any(
            "VERIFIER_SUCCESS_OVER_FAILURE" not in row.get("reason_codes", [])
            for row in base_rows
        ):
            raise ValueError(f"unverified pair found in base {split}")
        if any(not _is_contract_pair(row) for row in target_rows):
            raise ValueError(f"invalid deterministic contract pair in {split}")
        merged[split] = sorted(
            [*base_rows, *target_rows], key=lambda row: str(row["pair_id"])
        )
        source_counts[split] = {
            "verifier": len(base_rows),
            "single_action_contract": len(target_rows),
        }

    all_rows = [row for split in SPLITS for row in merged[split]]
    pair_ids = [str(row.get("pair_id") or "") for row in all_rows]
    contexts = [str(row.get("context_hash") or "") for row in all_rows]
    if not all(pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate or empty pair IDs after Stage35 merge")
    if not all(contexts) or len(contexts) != len(set(contexts)):
        raise ValueError("duplicate or empty contexts after Stage35 merge")
    split_contexts = {
        split: {str(row["context_hash"]) for row in rows}
        for split, rows in merged.items()
    }
    if any(
        split_contexts[left] & split_contexts[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError("context split overlap after Stage35 merge")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _write_jsonl(output_dir / f"{split}.jsonl", merged[split])
    version = (
        "stage35-mixed-preferences-"
        + _hash({split: [row["pair_id"] for row in merged[split]] for split in SPLITS})[
            :16
        ]
    )
    family_counts = Counter(str(row.get("family") or "unknown") for row in all_rows)
    evidence_counts = {
        "verifier_success_over_failure": sum(
            "VERIFIER_SUCCESS_OVER_FAILURE" in row.get("reason_codes", [])
            for row in all_rows
        ),
        "deterministic_single_action_contract": sum(
            _is_contract_pair(row) for row in all_rows
        ),
    }
    manifest = {
        "schema_version": "stage35-mixed-evidence-preferences.v1",
        "status": "passed",
        "dataset_version": version,
        "requires_verifier_success_over_failure": False,
        "preference_evidence_policy": EVIDENCE_POLICY,
        "base_dir": base_dir.as_posix(),
        "base_manifest_sha256": _sha256(base_manifest_path),
        "single_action_root": target_root.as_posix(),
        "single_action_manifest_sha256": _sha256(target_manifest_path),
        "split_counts": {split: len(rows) for split, rows in merged.items()},
        "source_split_counts": source_counts,
        "train_single_action_ratio": round(
            source_counts["train"]["single_action_contract"] / len(merged["train"]), 8
        ),
        "family_counts": dict(family_counts),
        "evidence_counts": evidence_counts,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--single-action-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = merge(args.base_dir, args.single_action_root, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
