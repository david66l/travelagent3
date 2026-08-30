"""Build verified multi-turn SFT plus replay anchors from the V3 curriculum."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.environment import TravelAgentEnvironment  # noqa: E402
from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.loop import PolicyAction, PolicyContext  # noqa: E402
from agentic.policy import ControllerFirstPolicy  # noqa: E402
from agentic.sft_dataset import DatasetManifest, EpisodeCandidate, SFTExample  # noqa: E402
from agentic.training import load_jsonl  # noqa: E402
from scripts.build_multiturn_recovery_dataset import build_multiturn_example  # noqa: E402


class _VerifiedDecisionLoopOracle:
    def __init__(self, initial: list[str], recovery: list[str]) -> None:
        self.initial = initial
        self.recovery = recovery

    async def propose(self, context: PolicyContext) -> PolicyAction:
        if "accept_candidates" in context.allowed_actions:
            return PolicyAction(action="accept_candidates")
        if "search_pois" not in context.allowed_actions:
            raise ValueError(f"unexpected delegated actions: {context.allowed_actions}")
        keywords = self.recovery if context.failure_summary else self.initial
        return PolicyAction(action="search_pois", arguments={"keywords": keywords})


async def _recovery_example(row: GRPOCorpusRow, *, split: str) -> SFTExample:
    metadata = row.snapshot.hidden_test_facts.get("decision_loop_curriculum")
    if not isinstance(metadata, dict):
        raise ValueError(f"{row.task.task_id} lacks decision-loop metadata")
    initial = [str(item) for item in metadata["initial_keywords"]]
    recovery = [str(item) for item in metadata["expected_recovery_keywords"]]
    scenario = str(metadata["scenario"])
    rollout = await TravelAgentEnvironment(row.task, row.snapshot).rollout(
        ControllerFirstPolicy(_VerifiedDecisionLoopOracle(initial, recovery))
    )
    if rollout.reward.gate_status != "passed":
        raise ValueError(f"verified teacher failed: {row.task.task_id}")
    candidate = EpisodeCandidate(
        scenario_id=row.task.task_id,
        source="synthetic",
        template_family=row.task.template_family,
        city=str(row.task.slots.get("destination") or "unknown"),
        episode=rollout.episode,
    )
    return build_multiturn_example(
        candidate,
        split=split,
        expected_error=(
            "QUERY_TOO_BROAD" if scenario == "change_arguments" else "UPSTREAM_TIMEOUT"
        ),
        require_adaptation=scenario == "change_arguments",
        example_prefix="stage3-decision-loop-v3",
    )


def _replay_rows(replay_dir: Path, split: str, count: int) -> list[SFTExample]:
    rows = [SFTExample(**row) for row in load_jsonl(replay_dir / f"{split}.jsonl")]
    if len(rows) < count:
        raise ValueError(f"insufficient {split} replay rows: {len(rows)}<{count}")
    output: list[SFTExample] = []
    for item in rows[:count]:
        payload = deepcopy(item.model_dump(mode="json"))
        payload["example_id"] = "stage3-v3-replay:" + payload["example_id"]
        output.append(SFTExample(**payload))
    return output


def _payload_digest(example: SFTExample) -> str:
    payload = {
        "messages": [message.model_dump(mode="json") for message in example.messages],
        "tools": example.tools,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def build(source_dir: Path, replay_dir: Path, output_dir: Path) -> DatasetManifest:
    source = {
        split: load_grpo_corpus(source_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    output: dict[str, list[SFTExample]] = {}
    seen_payloads: set[str] = set()
    for split, rows in source.items():
        recovery = [await _recovery_example(row, split=split) for row in rows]
        replay = _replay_rows(replay_dir, split, len(recovery))
        combined = sorted([*recovery, *replay], key=lambda item: item.example_id)
        for example in combined:
            digest = _payload_digest(example)
            if digest in seen_payloads:
                raise ValueError(f"model-visible SFT payload duplicate: {example.example_id}")
            seen_payloads.add(digest)
        output[split] = combined

    scenario_sets = {
        split: {item.scenario_id for item in rows} for split, rows in output.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if scenario_sets[left] & scenario_sets[right]:
            raise ValueError(f"SFT scenario leakage between {left} and {right}")

    version = "stage3-decision-loop-sft-v3-" + hashlib.sha256(
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
        policy_versions=sorted(
            {f"{item.policy_name}:{item.policy_version}" for item in all_rows}
        ),
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
                "schema_version": "stage3-decision-loop-sft-v3-derivation.v1",
                "source_dir": str(source_dir),
                "replay_dir": str(replay_dir),
                "recovery_to_replay_ratio": "1:1",
                "label_oracle": "immutable snapshot verified second recovery action",
                "rl_comparison_baseline": "checkpoint trained on this dataset",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        asyncio.run(build(args.source_dir, args.replay_dir, args.output_dir)).model_dump_json(
            indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
