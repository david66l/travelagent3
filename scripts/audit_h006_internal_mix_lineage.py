"""Hash-only isolation audit for a final H006 internal SFT mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend" / "src"))

from evaluation.posttraining_promotion_protocol import LineageRecord  # noqa: E402
from scripts.audit_h006_data_lineage import _source_hashes, _source_id, _tool_snapshot_hash  # noqa: E402


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(dataset_dir: Path, forbidden: list[Path]) -> dict[str, Any]:
    examples = {
        split: rows(dataset_dir / f"{split}.jsonl") for split in ("train", "validation")
    }
    lineage = [LineageRecord.model_validate(item) for item in rows(dataset_dir / "lineage.jsonl")]
    indexed = {item.record_id: item for item in lineage}
    ids = [item["example_id"] for values in examples.values() for item in values]
    if len(indexed) != len(lineage):
        raise ValueError("duplicate lineage record ids")
    source_roles: dict[str, set[str]] = {}
    for split, values in examples.items():
        expected = "train" if split == "train" else "train-shadow"
        for item in values:
            record = indexed.get(item["example_id"])
            if record is None or record.split != expected:
                raise ValueError("example lineage missing or in wrong split")
            source_roles.setdefault(record.source_state_id, set()).add(expected)
    forbidden_rows = [item for path in forbidden for item in rows(path)]
    forbidden_ids = {_source_id(item) for item in forbidden_rows}
    forbidden_hashes = {value for item in forbidden_rows for value in _source_hashes(item)}
    forbidden_tools = {_tool_snapshot_hash(item) for item in forbidden_rows}
    candidate_ids = {item.source_state_id for item in lineage}
    candidate_hashes = {item.source_hash for item in lineage}
    candidate_tools = {item.tool_snapshot_hash for item in lineage}
    gates = {
        "all_examples_registered_once": len(ids) == len(set(ids)) == len(indexed),
        "train_shadow_source_isolation": all(len(values) == 1 for values in source_roles.values()),
        "known_forbidden_source_ids_zero": not (candidate_ids & forbidden_ids),
        "known_forbidden_source_hashes_zero": not (candidate_hashes & forbidden_hashes),
        "known_forbidden_tool_hashes_zero": not (candidate_tools & forbidden_tools),
    }
    return {
        "schema_version": "h006-internal-mix-lineage-audit.v1",
        "passed_for_internal_training": all(gates.values()),
        "promotion_lineage_cleared": False,
        "promotion_note": "protected promotion/sealed hash registry remains a separate mandatory gate",
        "gates": gates,
        "counts": {
            "examples": len(ids), "lineage_records": len(lineage),
            "independent_sources": len(candidate_ids), "known_forbidden_rows": len(forbidden_rows),
        },
        "overlap_counts": {
            "source_id": len(candidate_ids & forbidden_ids),
            "source_hash": len(candidate_hashes & forbidden_hashes),
            "tool_hash": len(candidate_tools & forbidden_tools),
        },
        "input_sha256": {
            str(path): sha256(path) for path in [
                dataset_dir / "train.jsonl", dataset_dir / "validation.jsonl",
                dataset_dir / "lineage.jsonl", *forbidden,
            ]
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--forbidden-grpo", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite audit: {args.output}")
    report = audit(args.dataset_dir, args.forbidden_grpo)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed_for_internal_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
