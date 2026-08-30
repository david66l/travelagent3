"""Build a family-balanced DPO curriculum from the verified preference pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TRAIN_LIMITS = {
    "clarification": 10_000,
    "search": 256,
    "recovery": 256,
    "tradeoff": 10_000,
}


def _read(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"preference split is empty: {path}")
    return rows


def _stable_order(row: dict[str, Any]) -> str:
    return hashlib.sha256(str(row["pair_id"]).encode()).hexdigest()


def build(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "passed":
        raise ValueError("source preference manifest did not pass")
    if not source_manifest.get("requires_verifier_success_over_failure"):
        raise ValueError("source preferences lack the verifier success-over-failure contract")

    source_train = _read(source_dir / "train.jsonl")
    available = Counter(str(row["family"]) for row in source_train)
    unknown = sorted(set(available) - set(DEFAULT_TRAIN_LIMITS))
    if unknown:
        raise ValueError(f"unclassified preference families: {unknown}")

    selected_train: list[dict[str, Any]] = []
    for family, limit in DEFAULT_TRAIN_LIMITS.items():
        candidates = sorted(
            (row for row in source_train if row["family"] == family),
            key=_stable_order,
        )
        selected_train.extend(candidates[:limit])
    selected_train.sort(key=lambda row: str(row["pair_id"]))
    splits = {
        "train": selected_train,
        "validation": _read(source_dir / "validation.jsonl"),
        "test": _read(source_dir / "test.jsonl"),
    }

    pair_ids: set[str] = set()
    split_contexts: dict[str, set[str]] = {}
    for split, rows in splits.items():
        split_contexts[split] = {str(row["context_hash"]) for row in rows}
        for row in rows:
            pair_id = str(row["pair_id"])
            if pair_id in pair_ids:
                raise ValueError(f"duplicate preference pair across splits: {pair_id}")
            pair_ids.add(pair_id)
            if "VERIFIER_SUCCESS_OVER_FAILURE" not in row.get("reason_codes", []):
                raise ValueError(f"unverified preference selected: {pair_id}")
    overlap = any(
        split_contexts[left] & split_contexts[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    if overlap:
        raise ValueError("balanced preference context split leakage detected")

    version_payload = {split: [row["pair_id"] for row in rows] for split, rows in splits.items()}
    version = "preference-balanced-" + hashlib.sha256(
        json.dumps(version_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (output_dir / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": "balanced-preference-dataset.v1",
        "status": "passed",
        "dataset_version": version,
        "source_dir": str(source_dir),
        "source_manifest_sha256": hashlib.sha256(source_manifest_path.read_bytes()).hexdigest(),
        "requires_verifier_success_over_failure": True,
        "train_limits": DEFAULT_TRAIN_LIMITS,
        "source_train_family_counts": dict(available),
        "selected_train_family_counts": dict(Counter(row["family"] for row in selected_train)),
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "unique_pairs": len(pair_ids),
        "context_split_overlap": 0,
        "frozen_holdout_payload_overlap": source_manifest.get("frozen_holdout_payload_overlap"),
        "errors": [],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
