"""Build an environment-balanced formal SFT curriculum from the audited pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.sft_dataset import DatasetManifest, SFTExample  # noqa: E402

DEFAULT_TRAIN_LIMITS = {
    "travel-curriculum.v1": 128,
    "travel-priority-search.v1": 128,
    "travel-adaptive-recovery.v1": 256,
    "verified-preference.v1": 10_000,
    "travel-tradeoff-decision.v1": 10_000,
}


def _read(path: Path) -> list[SFTExample]:
    return [
        SFTExample(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stable_order(example: SFTExample) -> str:
    return hashlib.sha256(example.example_id.encode()).hexdigest()


def build(source_dir: Path, output_dir: Path) -> dict[str, object]:
    source_train = _read(source_dir / "train.jsonl")
    selected_train: list[SFTExample] = []
    available = Counter(item.environment_version for item in source_train)
    for environment, limit in DEFAULT_TRAIN_LIMITS.items():
        candidates = sorted(
            (item for item in source_train if item.environment_version == environment),
            key=_stable_order,
        )
        selected_train.extend(candidates[:limit])
    unknown = sorted(set(available) - set(DEFAULT_TRAIN_LIMITS))
    if unknown:
        raise ValueError(f"unclassified SFT environments: {unknown}")
    selected_train.sort(key=lambda item: item.example_id)
    splits = {
        "train": selected_train,
        "validation": _read(source_dir / "validation.jsonl"),
        "test": _read(source_dir / "test.jsonl"),
    }
    all_rows = [item for rows in splits.values() for item in rows]
    if len({item.example_id for item in all_rows}) != len(all_rows):
        raise ValueError("balanced SFT contains duplicate example IDs")
    split_scenarios = {
        split: {item.scenario_id for item in rows} for split, rows in splits.items()
    }
    overlap = any(
        split_scenarios[left] & split_scenarios[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    if overlap:
        raise ValueError("balanced SFT scenario split leakage detected")

    version = "sft-balanced-" + hashlib.sha256(
        json.dumps(
            {split: [item.example_id for item in rows] for split, rows in splits.items()},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    manifest = DatasetManifest(
        dataset_version=version,
        candidate_episodes=len(all_rows),
        accepted_episodes=len(all_rows),
        rejected_episodes=0,
        exported_examples=len(all_rows),
        split_examples={split: len(rows) for split, rows in splits.items()},
        source_episodes=dict(Counter(item.source for item in all_rows)),
        quality_episodes=dict(Counter(item.quality_label for item in all_rows)),
        rejection_codes={},
        environment_versions=sorted({item.environment_version for item in all_rows}),
        policy_versions=sorted({f"{item.policy_name}:{item.policy_version}" for item in all_rows}),
        split_group_overlap=False,
        excluded_policy_steps=len(source_train) - len(selected_train),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        (output_dir / f"{split}.jsonl").write_text(
            "\n".join(item.model_dump_json() for item in rows) + "\n",
            encoding="utf-8",
        )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    derivation = {
        "schema_version": "balanced-student-sft-derivation.v1",
        "source_dir": str(source_dir),
        "source_train_examples": len(source_train),
        "train_limits": DEFAULT_TRAIN_LIMITS,
        "source_train_environment_counts": dict(available),
        "selected_train_environment_counts": dict(
            Counter(item.environment_version for item in selected_train)
        ),
        "selected_train_action_counts": dict(
            Counter(item.messages[-1].tool_calls[0].function.name for item in selected_train)
        ),
        "split_counts": {split: len(rows) for split, rows in splits.items()},
        "dataset_version": version,
        "scenario_split_overlap": 0,
    }
    (output_dir / "derivation.json").write_text(
        json.dumps(derivation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return derivation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
