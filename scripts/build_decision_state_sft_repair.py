"""Convert verified decision-state failures into a tiny SFT bridge corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.grpo_training import load_grpo_corpus  # noqa: E402
from agentic.policy_actions import policy_action_schemas  # noqa: E402
from agentic.sft_dataset import DatasetManifest, SFTExample  # noqa: E402

_ACTIONS = ["retrieve_city_knowledge", "get_poi_detail", "get_route_matrix"]


def _examples(path: Path, splits: list[str]) -> list[SFTExample]:
    rows = load_grpo_corpus(path)
    if len(rows) != len(splits):
        raise ValueError("split assignment does not match decision-state rows")
    examples = []
    for row, split in zip(rows, splits, strict=True):
        state = row.snapshot.hidden_test_facts.get("grpo_decision_state")
        if not isinstance(state, dict) or state.get("target_action") != "get_poi_detail":
            raise ValueError("unsupported or missing decision-state target")
        messages = [dict(message) for message in state.get("prompt_messages") or []]
        if not messages or messages[-1].get("role") != "tool":
            raise ValueError("verified decision-state history is missing")
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_poi_detail", "arguments": {}},
                    }
                ],
            }
        )
        examples.append(
            SFTExample(
                example_id=f"repair:{row.task.task_id}",
                scenario_id=row.task.task_id,
                trajectory_id=str(state.get("source_trajectory_id") or row.task.task_id),
                step_index=len(state.get("prefix_actions") or []),
                split=split,
                quality_label="validated_plan",
                source="synthetic",
                environment_version=row.snapshot.environment_version,
                policy_name="verified-decision-state-repair",
                policy_version="v1",
                messages=messages,
                tools=policy_action_schemas(_ACTIONS),
            )
        )
    return examples


def _write(path: Path, rows: list[SFTExample]) -> None:
    path.write_text(
        "\n".join(row.model_dump_json() for row in rows) + "\n",
        encoding="utf-8",
    )


def build(source_dir: Path, output_dir: Path) -> dict[str, Any]:
    train_count = len(load_grpo_corpus(source_dir / "train.jsonl"))
    train = _examples(source_dir / "train.jsonl", ["train"] * train_count)
    heldout_rows = load_grpo_corpus(source_dir / "validation.jsonl")
    validation_count = len(heldout_rows) // 2
    heldout = _examples(
        source_dir / "validation.jsonl",
        ["validation"] * validation_count
        + ["test"] * (len(heldout_rows) - validation_count),
    )
    validation = [row for row in heldout if row.split == "validation"]
    test = [row for row in heldout if row.split == "test"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / "train.jsonl", train)
    _write(output_dir / "validation.jsonl", validation)
    _write(output_dir / "test.jsonl", test)
    version_payload = "\n".join(row.model_dump_json() for row in [*train, *heldout])
    dataset_version = "decision-repair-" + hashlib.sha256(version_payload.encode()).hexdigest()[:16]
    manifest = DatasetManifest(
        dataset_version=dataset_version,
        created_at=datetime.now(UTC),
        candidate_episodes=len(train) + len(heldout),
        accepted_episodes=len(train) + len(heldout),
        rejected_episodes=0,
        exported_examples=len(train) + len(heldout),
        split_examples={"train": len(train), "validation": len(validation), "test": len(test)},
        source_episodes={"synthetic": len(train) + len(heldout)},
        quality_episodes={"validated_plan": len(train) + len(heldout)},
        rejection_codes={},
        environment_versions=sorted({row.environment_version for row in [*train, *heldout]}),
        policy_versions=["v1"],
        split_group_overlap=False,
    )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return {"dataset_version": dataset_version, "counts": manifest.split_examples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output_dir), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
