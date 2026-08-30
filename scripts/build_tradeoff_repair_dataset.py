"""Build leakage-safe tradeoff tool-call repair data from official SFT train."""

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


def _action(row: dict[str, Any]) -> str | None:
    calls = row["messages"][-1].get("tool_calls") or []
    return calls[0]["function"]["name"] if calls else None


def _stable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(row["scenario_id"].encode()).hexdigest(),
    )


def _split(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(_stable(rows)):
        bucket = index % 20
        split = "validation" if bucket < 3 else ("test" if bucket < 6 else "train")
        result[split].append(row)
    return result


def build(source_dir: Path, output_dir: Path) -> DatasetManifest:
    source = load_jsonl(source_dir / "train.jsonl")
    tradeoff = [row for row in source if _action(row) in {"propose_tradeoff", "abort"}]
    if not tradeoff:
        raise ValueError("official train split contains no verified tradeoff decisions")
    tradeoff_splits = _split(tradeoff)
    counts = {split: len(tradeoff_splits[split]) for split in ("train", "validation", "test")}

    protected = {row["scenario_id"] for row in tradeoff}
    replay_pool = _stable(
        [
            row
            for row in source
            if row["scenario_id"] not in protected and _action(row) in {"search_pois", "ask_user"}
        ]
    )
    replay_splits: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for split in ("train", "validation", "test"):
        replay_splits[split] = replay_pool[cursor : cursor + counts[split]]
        cursor += counts[split]

    output: dict[str, list[SFTExample]] = {}
    for split in ("train", "validation", "test"):
        combined: list[SFTExample] = []
        for kind, rows in (
            ("tradeoff", tradeoff_splits[split]),
            ("replay", replay_splits[split]),
        ):
            if len(rows) != counts[split]:
                raise ValueError(f"insufficient {kind} examples for {split}")
            for row in rows:
                updated = deepcopy(row)
                updated["split"] = split
                updated["example_id"] = f"tradeoff-repair:{kind}:{row['example_id']}"
                combined.append(SFTExample(**updated))
        output[split] = sorted(combined, key=lambda item: item.example_id)

    scenario_sets = {split: {item.scenario_id for item in rows} for split, rows in output.items()}
    if any(
        scenario_sets[left] & scenario_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("tradeoff repair split leakage detected")

    version = "sft-tradeoff-repair-" + hashlib.sha256(
        json.dumps(
            {split: [item.example_id for item in rows] for split, rows in output.items()},
            sort_keys=True,
        ).encode()
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
        policy_versions=sorted({f"{item.policy_name}:{item.policy_version}" for item in all_rows}),
        split_group_overlap=False,
        excluded_policy_steps=0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in output.items():
        (output_dir / f"{split}.jsonl").write_text(
            "\n".join(item.model_dump_json() for item in rows) + "\n", encoding="utf-8"
        )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "derivation.json").write_text(
        json.dumps(
            {
                "source_split": "official train only",
                "official_validation_or_test_used": False,
                "tradeoff_examples": len(tradeoff),
                "replay_ratio": "1:1",
                "target_failure": "NF4 invalid-or-empty tradeoff tool call",
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
