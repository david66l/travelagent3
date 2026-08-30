"""Merge audited SFT sources with prompt deduplication and group-safe resplitting."""

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

from agentic.sft_dataset import DatasetManifest, SFTExample  # noqa: E402


SPLITS = ("train", "validation", "test")


def read_source(path: Path, source_name: str) -> list[tuple[str, SFTExample]]:
    rows = []
    for split in SPLITS:
        split_path = path / f"{split}.jsonl"
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append((source_name, SFTExample.model_validate_json(line)))
    return rows


def merge(source_specs: list[tuple[str, Path]], output_dir: Path) -> dict[str, Any]:
    if len(source_specs) < 2:
        raise ValueError("at least two SFT sources are required")
    rows = [
        item
        for source_name, source_path in source_specs
        for item in read_source(source_path, source_name)
    ]
    groups: dict[str, list[tuple[str, SFTExample]]] = defaultdict(list)
    for source_name, example in rows:
        groups[_prompt_hash(example)].append((source_name, example))

    retained: list[tuple[str, SFTExample]] = []
    duplicate_rows = 0
    conflicts = []
    for prompt_hash, examples in groups.items():
        responses = {
            _hash(item.messages[-1].model_dump(mode="json", exclude_none=True))
            for _, item in examples
        }
        if len(responses) > 1:
            conflicts.append(
                {
                    "prompt_hash": prompt_hash,
                    "row_count": len(examples),
                    "source_counts": dict(Counter(source for source, _ in examples)),
                    "finding_hash": _hash(
                        {
                            "prompt_hash": prompt_hash,
                            "responses": sorted(responses),
                        }
                    ),
                }
            )
            continue
        # Prefer the later source so Stage32 provenance survives exact replay.
        selected = max(
            examples,
            key=lambda item: (
                next(
                    index
                    for index, (name, _) in enumerate(source_specs)
                    if name == item[0]
                ),
                item[1].example_id,
            ),
        )
        retained.append(selected)
        duplicate_rows += len(examples) - 1

    output: dict[str, list[SFTExample]] = {split: [] for split in SPLITS}
    retained_source_counts: Counter[str] = Counter()
    for source_name, example in retained:
        split = _group_split(example.scenario_id)
        payload = example.model_dump(mode="json")
        payload["split"] = split
        payload["example_id"] = f"stage32:{source_name}:{example.example_id}"
        output[split].append(SFTExample(**payload))
        retained_source_counts[source_name] += 1
    for split in SPLITS:
        output[split].sort(key=lambda item: item.example_id)

    all_rows = [item for split in SPLITS for item in output[split]]
    if not all(output[split] for split in SPLITS):
        raise ValueError("merged SFT must populate train, validation and test")
    prompts = [_prompt_hash(item) for item in all_rows]
    if len(prompts) != len(set(prompts)):
        raise ValueError("duplicate model-visible prompt after merge")
    scenarios = {
        split: {item.scenario_id for item in output[split]} for split in SPLITS
    }
    if any(
        scenarios[left] & scenarios[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise ValueError("scenario split overlap after merge")

    conflict_rate = sum(item["row_count"] for item in conflicts) / len(rows)
    errors = []
    if conflict_rate > 0.05:
        errors.append(f"LABEL_CONFLICT_RATE_TOO_HIGH:{conflict_rate:.6f}>0.05")
    version = "sft-stage32-student-" + _hash(
        {split: [item.example_id for item in output[split]] for split in SPLITS}
    )[:16]
    manifest = DatasetManifest(
        dataset_version=version,
        candidate_episodes=len(rows),
        accepted_episodes=len(all_rows),
        rejected_episodes=len(rows) - len(all_rows),
        exported_examples=len(all_rows),
        split_examples={split: len(output[split]) for split in SPLITS},
        source_episodes=dict(retained_source_counts),
        quality_episodes=dict(Counter(item.quality_label for item in all_rows)),
        rejection_codes={
            "DUPLICATE_MODEL_PAYLOAD": duplicate_rows,
            "MODEL_PAYLOAD_LABEL_CONFLICT": sum(
                item["row_count"] for item in conflicts
            ),
        },
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
            [item.model_dump(mode="json") for item in output[split]],
        )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    _write_jsonl(output_dir / "label_conflicts.jsonl", conflicts)
    derivation = {
        "schema_version": "stage32-student-sft-merge.v1",
        "status": "passed" if not errors else "rejected",
        "dataset_version": version,
        "sources": [
            {
                "name": name,
                "path": str(path),
                "manifest_sha256": _sha256(path / "manifest.json"),
                "examples": len(read_source(path, name)),
            }
            for name, path in source_specs
        ],
        "input_rows": len(rows),
        "exported_examples": len(all_rows),
        "retained_source_counts": dict(retained_source_counts),
        "duplicates_dropped": duplicate_rows,
        "label_conflict_groups_quarantined": len(conflicts),
        "label_conflict_rows_quarantined": sum(
            item["row_count"] for item in conflicts
        ),
        "label_conflict_rate": round(conflict_rate, 8),
        "split_counts": manifest.split_examples,
        "action_counts": dict(Counter(_action(item) for item in all_rows)),
        "errors": errors,
    }
    (output_dir / "derivation.json").write_text(
        json.dumps(derivation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return derivation


def _prompt_hash(example: SFTExample) -> str:
    return _hash(
        {
            "messages": [
                item.model_dump(mode="json", exclude_none=True)
                for item in example.messages[:-1]
            ],
            "tools": example.tools,
        }
    )


def _action(example: SFTExample) -> str:
    calls = example.messages[-1].tool_calls
    return calls[0].function.name if len(calls) == 1 else "invalid"


def _group_split(value: str) -> str:
    bucket = int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="NAME=DATASET_DIR; repeat for each source in precedence order",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = []
    for value in args.source:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error("--source must use NAME=DATASET_DIR")
        sources.append((name, Path(path)))
    report = merge(sources, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
