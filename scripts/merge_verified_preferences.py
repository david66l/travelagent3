"""Merge verifier-grounded preference shards into deterministic train splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.distillation import TeacherPreferencePair  # noqa: E402


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_hash(pair: TeacherPreferencePair) -> str:
    return _canonical_hash({"messages": pair.messages, "tools": pair.tools})


def _load_holdout_payloads(path: Path | None) -> set[str]:
    if path is None:
        return set()
    hashes: set[str] = set()
    for split in ("regular", "hard", "adversarial"):
        split_path = path / f"{split}.jsonl"
        if not split_path.is_file():
            raise FileNotFoundError(f"missing frozen holdout split: {split_path}")
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                messages = [
                    {
                        key: value
                        for key, value in message.items()
                        if value is not None
                        and not (key == "tool_calls" and not value)
                    }
                    for message in item["messages"]
                ]
                hashes.add(_canonical_hash({"messages": messages, "tools": item["tools"]}))
    return hashes


def load_sources(
    source_dirs: list[Path],
) -> tuple[list[TeacherPreferencePair], list[dict[str, Any]]]:
    by_id: dict[str, TeacherPreferencePair] = {}
    sources: list[dict[str, Any]] = []
    for source_dir in source_dirs:
        path = source_dir / "preference_pairs.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing preference shard: {path}")
        source_count = 0
        duplicate_count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            source_count += 1
            pair = TeacherPreferencePair(**json.loads(line))
            existing = by_id.get(pair.pair_id)
            if existing is not None:
                if existing != pair:
                    raise ValueError(f"conflicting duplicate pair_id: {pair.pair_id}")
                duplicate_count += 1
                continue
            by_id[pair.pair_id] = pair
        sources.append(
            {
                "directory": str(source_dir),
                "preference_file_sha256": _file_sha256(path),
                "rows": source_count,
                "exact_duplicates_dropped": duplicate_count,
            }
        )
    return list(by_id.values()), sources


def deterministic_split(
    pairs: list[TeacherPreferencePair],
) -> dict[str, list[TeacherPreferencePair]]:
    if len(pairs) < 3:
        raise ValueError("at least three unique pairs are required for train/validation/test")
    ordered = sorted(pairs, key=lambda item: _canonical_hash(item.pair_id))
    validation_count = max(1, round(len(ordered) * 0.1))
    test_count = max(1, round(len(ordered) * 0.1))
    if validation_count + test_count >= len(ordered):
        raise ValueError("not enough pairs to retain a training split")
    return {
        "validation": ordered[:validation_count],
        "test": ordered[validation_count : validation_count + test_count],
        "train": ordered[validation_count + test_count :],
    }


def merge(
    source_dirs: list[Path],
    output_dir: Path,
    *,
    forbidden_holdout_dir: Path | None = None,
    require_verifier_failure: bool = True,
) -> dict[str, Any]:
    pairs, sources = load_sources(source_dirs)
    errors: list[str] = []
    if require_verifier_failure:
        weak = [
            pair.pair_id
            for pair in pairs
            if "VERIFIER_SUCCESS_OVER_FAILURE" not in pair.reason_codes
        ]
        if weak:
            errors.append(f"NON_VERIFIER_PREFERENCES:{len(weak)}")

    pair_ids = [pair.pair_id for pair in pairs]
    context_hashes = [pair.context_hash for pair in pairs]
    payload_hashes = [_payload_hash(pair) for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        errors.append("PAIR_ID_DUPLICATE")
    if len(context_hashes) != len(set(context_hashes)):
        errors.append("CONTEXT_HASH_DUPLICATE")
    if len(payload_hashes) != len(set(payload_hashes)):
        errors.append("MODEL_PAYLOAD_DUPLICATE")
    holdout_overlap = set(payload_hashes) & _load_holdout_payloads(
        forbidden_holdout_dir
    )
    if holdout_overlap:
        errors.append(f"FROZEN_HOLDOUT_PAYLOAD_OVERLAP:{len(holdout_overlap)}")

    splits = deterministic_split(pairs)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        _write_jsonl(
            output_dir / f"{split}.jsonl",
            [item.model_dump(mode="json") for item in rows],
        )
    manifest = {
        "schema_version": "verified-preference-dataset.v1",
        "status": "passed" if not errors else "rejected",
        "requires_verifier_success_over_failure": require_verifier_failure,
        "sources": sources,
        "unique_pairs": len(pairs),
        "unique_contexts": len(set(context_hashes)),
        "unique_model_payloads": len(set(payload_hashes)),
        "family_counts": dict(Counter(pair.family for pair in pairs)),
        "reason_counts": dict(
            Counter(reason for pair in pairs for reason in pair.reason_codes)
        ),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "frozen_holdout_payload_overlap": len(holdout_overlap),
        "errors": errors,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-holdout-dir", type=Path)
    parser.add_argument(
        "--allow-non-verifier-preferences",
        action="store_true",
        help="Allow efficiency-only pairs that lack a verified success/failure difference.",
    )
    args = parser.parse_args()
    manifest = merge(
        args.source_dir,
        args.output_dir,
        forbidden_holdout_dir=args.forbidden_holdout_dir,
        require_verifier_failure=not args.allow_non_verifier_preferences,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
