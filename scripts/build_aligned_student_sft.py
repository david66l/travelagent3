"""Repair teacher SFT labels and align its splits with verified preferences."""

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
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from agentic.distillation import TeacherPreferencePair  # noqa: E402
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT  # noqa: E402
from agentic.sft_dataset import DatasetManifest, SFTExample  # noqa: E402
from generate_teacher_distillation import (  # noqa: E402
    load_holdout_contract,
    model_payload_hash,
)

SPLITS = ("train", "validation", "test")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _action(example: SFTExample) -> str:
    calls = example.messages[-1].tool_calls
    if len(calls) != 1:
        raise ValueError(f"example must contain exactly one tool call: {example.example_id}")
    return calls[0].function.name


def _prompt_hash(example: SFTExample) -> str:
    return model_payload_hash(
        [message.model_dump(mode="json") for message in example.messages[:-1]],
        example.tools,
    )


def _preference_example(pair: TeacherPreferencePair, split: str) -> SFTExample:
    calls = pair.chosen.get("tool_calls") or []
    if len(calls) != 1 or calls[0]["function"]["name"] != "ask_user":
        raise ValueError(f"clarification preference is not an ask_user decision: {pair.pair_id}")
    return SFTExample(
        example_id=f"verified-preference:{pair.pair_id}",
        scenario_id=pair.task_id,
        trajectory_id=pair.chosen_trajectory_id,
        step_index=0,
        split=split,
        quality_label="clarification",
        source="teacher",
        environment_version="verified-preference.v1",
        policy_name="Qwen3-8B",
        policy_version="stage20-verifier-chosen",
        messages=[
            {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
            *pair.messages[1:],
            pair.chosen,
        ],
        tools=pair.tools,
    )


def build(
    source_sft_dir: Path,
    preference_dir: Path,
    output_dir: Path,
    *,
    forbidden_holdout_dir: Path | None = None,
) -> dict[str, Any]:
    preferences: list[tuple[str, TeacherPreferencePair]] = []
    scenario_splits: dict[str, str] = {}
    for split in SPLITS:
        for row in _read_jsonl(preference_dir / f"{split}.jsonl"):
            pair = TeacherPreferencePair(**row)
            previous = scenario_splits.setdefault(pair.task_id, split)
            if previous != split:
                raise ValueError(f"preference scenario crosses splits: {pair.task_id}")
            preferences.append((split, pair))

    retained: list[SFTExample] = []
    removed = Counter()
    realigned = Counter()
    for split in SPLITS:
        for row in _read_jsonl(source_sft_dir / f"{split}.jsonl"):
            example = SFTExample(**row)
            action = _action(example)
            if example.quality_label == "clarification" or action == "capability_check":
                removed[f"quality:{example.quality_label}"] += 1
                removed[f"action:{action}"] += 1
                continue
            target_split = scenario_splits.get(example.scenario_id, split)
            if target_split != split:
                realigned[f"{split}->{target_split}"] += 1
            normalized_messages = [
                example.messages[0].model_copy(
                    update={"content": AGENT_TOOL_POLICY_SYSTEM_PROMPT}
                ),
                *example.messages[1:],
            ]
            retained.append(
                example.model_copy(
                    update={"split": target_split, "messages": normalized_messages}
                )
            )

    replacements = [
        _preference_example(pair, split)
        for split, pair in preferences
        if pair.family == "clarification"
    ]
    examples = retained + replacements
    if not replacements:
        raise ValueError("verified preference dataset contains no clarification replacements")

    by_prompt: dict[str, list[SFTExample]] = {}
    for example in examples:
        by_prompt.setdefault(_prompt_hash(example), []).append(example)
    duplicate_groups = {
        key: rows for key, rows in by_prompt.items() if len(rows) > 1
    }
    if duplicate_groups:
        raise ValueError(f"duplicate model-visible prompts after repair: {len(duplicate_groups)}")

    split_scenarios = {
        split: {row.scenario_id for row in examples if row.split == split}
        for split in SPLITS
    }
    overlap = set()
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap.update(split_scenarios[left] & split_scenarios[right])
    if overlap:
        raise ValueError(f"scenario split leakage after preference alignment: {len(overlap)}")

    _, holdout_payloads = load_holdout_contract(forbidden_holdout_dir)
    prompt_hashes = {_prompt_hash(example) for example in examples}
    holdout_overlap = prompt_hashes & holdout_payloads
    if holdout_overlap:
        raise ValueError(f"frozen holdout prompt overlap after repair: {len(holdout_overlap)}")

    examples.sort(key=lambda item: (item.split, item.example_id))
    split_counts = Counter(item.split for item in examples)
    version_payload = [(item.example_id, item.split, _action(item)) for item in examples]
    dataset_version = "sft-aligned-" + hashlib.sha256(
        json.dumps(version_payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()[:16]
    manifest = DatasetManifest(
        dataset_version=dataset_version,
        candidate_episodes=len(examples),
        accepted_episodes=len(examples),
        rejected_episodes=0,
        exported_examples=len(examples),
        split_examples=dict(split_counts),
        source_episodes=dict(Counter(item.source for item in examples)),
        quality_episodes=dict(Counter(item.quality_label for item in examples)),
        rejection_codes={},
        environment_versions=sorted({item.environment_version for item in examples}),
        policy_versions=sorted(
            {f"{item.policy_name}:{item.policy_version}" for item in examples}
        ),
        split_group_overlap=False,
        excluded_policy_steps=sum(removed.values()),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        rows = [item for item in examples if item.split == split]
        (output_dir / f"{split}.jsonl").write_text(
            "\n".join(item.model_dump_json() for item in rows) + "\n",
            encoding="utf-8",
        )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    derivation = {
        "schema_version": "aligned-student-sft-derivation.v1",
        "status": "passed",
        "source_sft_dir": str(source_sft_dir),
        "preference_dir": str(preference_dir),
        "source_examples": sum(len(_read_jsonl(source_sft_dir / f"{s}.jsonl")) for s in SPLITS),
        "removed_counts": dict(removed),
        "verified_clarification_replacements": len(replacements),
        "preference_scenarios": len(scenario_splits),
        "split_realignments": dict(realigned),
        "output_examples": len(examples),
        "action_counts": dict(Counter(_action(item) for item in examples)),
        "split_counts": dict(split_counts),
        "unique_model_visible_prompts": len(prompt_hashes),
        "frozen_holdout_prompt_overlap": len(holdout_overlap),
        "scenario_split_overlap": 0,
    }
    (output_dir / "derivation.json").write_text(
        json.dumps(derivation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return derivation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sft-dir", type=Path, required=True)
    parser.add_argument("--preference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-holdout-dir", type=Path)
    args = parser.parse_args()
    result = build(
        args.source_sft_dir,
        args.preference_dir,
        args.output_dir,
        forbidden_holdout_dir=args.forbidden_holdout_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
