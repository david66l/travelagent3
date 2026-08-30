"""Build a leakage-safe recovery/replay SFT repair curriculum."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.sft_dataset import DatasetManifest, SFTExample  # noqa: E402
from agentic.training import load_jsonl  # noqa: E402


def _context(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(row["messages"][1]["content"])


def _is_recovery(row: dict[str, Any]) -> bool:
    return any(
        item.get("retryable") and item.get("code") == "UPSTREAM_TIMEOUT"
        for item in _context(row).get("failure_summary") or []
    )


def _family(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(_context(row).get("soft_preferences", {}).get("interests") or ["unknown"])


def _split_recovery(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_family(row)].append(row)
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # Each family contributes to every split. The official validation/test
    # files are deliberately not read, so final held-out evaluation stays clean.
    for family in sorted(groups):
        ordered = sorted(groups[family], key=lambda item: item["scenario_id"])
        for index, row in enumerate(ordered):
            bucket = index % 20
            split = "validation" if bucket < 3 else ("test" if bucket < 6 else "train")
            result[split].append(row)
    return result


def _select_replay(
    rows: list[dict[str, Any]], counts: dict[str, int]
) -> dict[str, list[dict[str, Any]]]:
    candidates = [row for row in rows if not _is_recovery(row)]
    # Stable hash ordering avoids depending on source file order and naturally
    # mixes action/family templates.
    ordered = sorted(
        candidates,
        key=lambda row: hashlib.sha256(row["scenario_id"].encode()).hexdigest(),
    )
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cursor = 0
    for split in ("train", "validation", "test"):
        result[split] = ordered[cursor : cursor + counts[split]]
        cursor += counts[split]
    return result


def build(source_dir: Path, output_dir: Path) -> DatasetManifest:
    source_rows = load_jsonl(source_dir / "train.jsonl")
    recovery = [row for row in source_rows if _is_recovery(row)]
    if not recovery:
        raise ValueError("source training split contains no verified recovery decisions")
    recovery_splits = _split_recovery(recovery)
    counts = {split: len(recovery_splits[split]) for split in ("train", "validation", "test")}
    replay_splits = _select_replay(source_rows, counts)

    output: dict[str, list[SFTExample]] = {}
    for split in ("train", "validation", "test"):
        combined = []
        for kind, rows in (
            ("recovery", recovery_splits[split]),
            ("replay", replay_splits[split]),
        ):
            for row in rows:
                updated = deepcopy(row)
                updated["split"] = split
                updated["example_id"] = f"repair:{kind}:{row['example_id']}"
                combined.append(SFTExample(**updated))
        output[split] = sorted(combined, key=lambda item: item.example_id)

    scenario_splits = {
        split: {item.scenario_id for item in rows} for split, rows in output.items()
    }
    if any(
        scenario_splits[left] & scenario_splits[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("repair curriculum split leakage detected")

    version_payload = {
        split: [item.example_id for item in rows] for split, rows in output.items()
    }
    version = "sft-repair-" + hashlib.sha256(
        json.dumps(version_payload, sort_keys=True).encode()
    ).hexdigest()[:16]
    all_rows = [item for rows in output.values() for item in rows]
    manifest = DatasetManifest(
        dataset_version=version,
        candidate_episodes=len(all_rows),
        accepted_episodes=len(all_rows),
        rejected_episodes=0,
        exported_examples=len(all_rows),
        split_examples={split: len(rows) for split, rows in output.items()},
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
    for split, rows in output.items():
        (output_dir / f"{split}.jsonl").write_text(
            "\n".join(item.model_dump_json() for item in rows) + "\n",
            encoding="utf-8",
        )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "derivation.json").write_text(
        json.dumps(
            {
                "source_dataset": str(source_dir),
                "source_split": "train only",
                "official_validation_or_test_used": False,
                "recovery_examples": sum(counts.values()),
                "replay_ratio": "1:1",
                "recovery_families": {
                    "|".join(family): count
                    for family, count in Counter(_family(row) for row in recovery).items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.source_dir, args.output_dir).model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
