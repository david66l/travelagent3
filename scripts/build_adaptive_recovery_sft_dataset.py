"""Build leakage-audited SFT data for evidence-conditioned query recovery."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.corpus_generation import AdaptiveRecoveryTeacherPolicy, CurriculumTeacherPolicy  # noqa: E402
from agentic.environment import TravelAgentEnvironment  # noqa: E402
from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.policy import ControllerFirstPolicy  # noqa: E402
from agentic.sft_dataset import (  # noqa: E402
    DatasetManifest,
    EpisodeCandidate,
    SFTDatasetBuilder,
    SFTExample,
)
from agentic.trajectory import EpisodeReplayVerifier  # noqa: E402
from scripts.build_adaptive_recovery_corpus import derive_adaptive_recovery  # noqa: E402
from scripts.build_multiturn_recovery_dataset import build_multiturn_example  # noqa: E402

_SPLIT_COUNTS = {"train": 256, "validation": 32, "test": 32}
_REPLAY_FAMILIES = ("clarification", "tradeoff", "search")


def _stable_key(row: GRPOCorpusRow) -> str:
    return hashlib.sha256(row.task.task_id.encode()).hexdigest()


def _partition(
    rows: list[GRPOCorpusRow], split_counts: dict[str, int]
) -> dict[str, list[GRPOCorpusRow]]:
    required = sum(split_counts.values())
    if len(rows) < required:
        raise ValueError(f"need {required} source rows, found {len(rows)}")
    ordered = sorted(rows, key=_stable_key)[:required]
    result: dict[str, list[GRPOCorpusRow]] = {}
    offset = 0
    for split in ("train", "validation", "test"):
        count = split_counts[split]
        result[split] = ordered[offset : offset + count]
        offset += count
    return result


def _replay_family(row: GRPOCorpusRow) -> str:
    if row.task.missing_slots:
        return "clarification"
    if row.task.feasibility_report.get("feasible", True) is False:
        return "tradeoff"
    return "search"


def _balanced_replay_partition(
    rows: list[GRPOCorpusRow],
    split_counts: dict[str, int],
    *,
    replay_ratio: int,
) -> dict[str, list[GRPOCorpusRow]]:
    """Allocate replay evenly across decision families without source overlap."""
    pools = {
        family: [row for row in rows if _replay_family(row) == family]
        for family in _REPLAY_FAMILIES
    }
    for family in pools:
        pools[family].sort(key=_stable_key)
    offsets = Counter()
    result: dict[str, list[GRPOCorpusRow]] = {}
    for split in ("train", "validation", "test"):
        total = split_counts[split] * replay_ratio
        base, remainder = divmod(total, len(_REPLAY_FAMILIES))
        family_counts = {
            family: base + (index < remainder)
            for index, family in enumerate(_REPLAY_FAMILIES)
        }
        selected: list[GRPOCorpusRow] = []
        for family in _REPLAY_FAMILIES:
            start = offsets[family]
            end = start + family_counts[family]
            if end > len(pools[family]):
                raise ValueError(
                    f"need {end} {family} replay rows, found {len(pools[family])}"
                )
            selected.extend(pools[family][start:end])
            offsets[family] = end
        result[split] = selected
    return result


async def _rollout_candidate(
    row: GRPOCorpusRow,
    *,
    adaptive: bool,
    semaphore: asyncio.Semaphore,
) -> EpisodeCandidate:
    async with semaphore:
        source = derive_adaptive_recovery(row) if adaptive else row
        teacher = AdaptiveRecoveryTeacherPolicy() if adaptive else CurriculumTeacherPolicy()
        rollout = await TravelAgentEnvironment(source.task, source.snapshot).rollout(
            ControllerFirstPolicy(teacher)
        )
        replay_errors = EpisodeReplayVerifier().verify(rollout.episode)
        if replay_errors or rollout.reward.gate_status != "passed":
            raise ValueError(
                f"{source.task.task_id} failed episode gates: "
                f"replay={replay_errors}, reward_gate={rollout.reward.gate_status}"
            )
        return EpisodeCandidate(
            scenario_id=source.task.task_id,
            source="synthetic",
            template_family=source.task.template_family,
            city=str(source.task.slots.get("destination") or "unknown"),
            episode=rollout.episode,
        )


def _relabel_replay(example: SFTExample, split: str) -> SFTExample:
    payload = deepcopy(example.model_dump(mode="json"))
    payload["example_id"] = f"adaptive-recovery:replay:{example.example_id}"
    payload["split"] = split
    return SFTExample(**payload)


def _audit_adaptive_example(example: SFTExample) -> dict[str, Any]:
    if [message.role for message in example.messages] != [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]:
        raise ValueError(f"{example.example_id} has an invalid multi-turn history")
    serialized = example.model_dump_json()
    if "hidden_test_facts" in serialized or "target_keywords" in serialized:
        raise ValueError(f"{example.example_id} leaks hidden evaluator facts")
    initial = json.loads(example.messages[1].content or "{}")["policy_state"]
    transition = json.loads(example.messages[3].content or "{}")
    feedback = transition["policy_state"]["failure_summary"][-1]
    first_args = example.messages[2].tool_calls[0].function.arguments
    second_args = example.messages[4].tool_calls[0].function.arguments
    interests = list(initial["soft_preferences"].get("interests") or [])
    targets = [item for item in interests if str(item) in str(feedback.get("message") or "")]
    if feedback.get("code") != "QUERY_TOO_BROAD" or not feedback.get("retryable"):
        raise ValueError(f"{example.example_id} lacks visible retryable broad-query feedback")
    if first_args == second_args:
        raise ValueError(f"{example.example_id} repeats the failed strategy")
    if len(targets) != 1 or second_args != {"keywords": [targets[0]]}:
        raise ValueError(f"{example.example_id} recovery is not grounded in visible evidence")
    if transition["last_transition"]["observations"][0]["error_code"] != "QUERY_TOO_BROAD":
        raise ValueError(f"{example.example_id} tool history does not match the failure")
    return {
        "example_id": example.example_id,
        "first_arguments": first_args,
        "visible_failure_code": feedback["code"],
        "recovery_arguments": second_args,
        "evidence_source": "policy_state.failure_summary.message",
    }


def _assert_unique_model_visible_payloads(
    output: dict[str, list[SFTExample]],
) -> int:
    """Reject ID-only diversity across every train/eval split."""
    seen: dict[str, tuple[str, str]] = {}
    duplicates: list[tuple[str, str, str, str]] = []
    for split in ("train", "validation", "test"):
        for example in output[split]:
            payload = json.dumps(
                {
                    "messages": [
                        message.model_dump(mode="json") for message in example.messages
                    ],
                    "tools": example.tools,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(payload.encode()).hexdigest()
            previous = seen.get(digest)
            if previous:
                duplicates.append(
                    (previous[0], previous[1], split, example.example_id)
                )
            else:
                seen[digest] = (split, example.example_id)
    if duplicates:
        first = duplicates[0]
        raise ValueError(
            "MODEL_VISIBLE_PAYLOAD_DUPLICATE:"
            f"{len(duplicates)} duplicates; first={first[0]}:{first[1]}="
            f"{first[2]}:{first[3]}"
        )
    return len(seen)


async def build(
    source_dir: Path,
    output_dir: Path,
    *,
    concurrency: int = 32,
    split_counts: dict[str, int] | None = None,
    replay_ratio: int = 1,
) -> DatasetManifest:
    counts = split_counts or _SPLIT_COUNTS
    if replay_ratio < 1:
        raise ValueError("replay_ratio must be at least 1")
    train_rows = load_grpo_corpus(source_dir / "train.jsonl")
    official_validation = load_grpo_corpus(source_dir / "validation.jsonl")
    official_validation_ids = {row.task.task_id for row in official_validation}
    if official_validation_ids & {row.task.task_id for row in train_rows}:
        raise ValueError("official GRPO train and validation splits overlap")

    eligible = [
        row
        for row in train_rows
        if len(row.snapshot.tool_responses.get("search_pois") or []) >= 2
        and (row.snapshot.tool_responses.get("search_pois") or [])[0].retryable
        and len(row.task.slots.get("interests") or []) >= 2
    ]
    adaptive_splits = _partition(eligible, counts)
    adaptive_source_ids = {
        row.task.task_id for rows in adaptive_splits.values() for row in rows
    }
    replay_pool = [
        row
        for row in train_rows
        if row.task.task_id not in adaptive_source_ids
        and not (
            len(row.snapshot.tool_responses.get("search_pois") or []) >= 2
            and (row.snapshot.tool_responses.get("search_pois") or [])[0].retryable
        )
    ]
    replay_counts = {split: count * replay_ratio for split, count in counts.items()}
    replay_splits = _balanced_replay_partition(
        replay_pool,
        counts,
        replay_ratio=replay_ratio,
    )

    semaphore = asyncio.Semaphore(concurrency)
    jobs: list[Any] = []
    job_meta: list[tuple[str, str]] = []
    for kind, splits in (("adaptive", adaptive_splits), ("replay", replay_splits)):
        for split in ("train", "validation", "test"):
            for row in splits[split]:
                jobs.append(
                    _rollout_candidate(
                        row,
                        adaptive=kind == "adaptive",
                        semaphore=semaphore,
                    )
                )
                job_meta.append((kind, split))
    candidates = await asyncio.gather(*jobs)

    output: dict[str, list[SFTExample]] = {key: [] for key in counts}
    audits: list[dict[str, Any]] = []
    standard_builder = SFTDatasetBuilder()
    for candidate, (kind, split) in zip(candidates, job_meta, strict=True):
        if kind == "adaptive":
            example = build_multiturn_example(
                candidate,
                split=split,
                expected_error="QUERY_TOO_BROAD",
                require_adaptation=True,
                example_prefix="adaptive-recovery",
            )
            audits.append(_audit_adaptive_example(example))
        else:
            built = standard_builder.build([candidate])
            if not built.reviews[0].accepted or len(built.examples) != 1:
                raise ValueError(
                    f"{candidate.scenario_id} is not a valid single-decision replay episode"
                )
            example = _relabel_replay(built.examples[0], split)
        output[split].append(example)

    scenario_sets = {
        split: {example.scenario_id.removesuffix("-adaptive-recovery") for example in rows}
        for split, rows in output.items()
    }
    overlap = set()
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap.update(scenario_sets[left] & scenario_sets[right])
    if overlap:
        raise ValueError(f"internal SFT split leakage detected: {sorted(overlap)[:3]}")
    if set().union(*scenario_sets.values()) & official_validation_ids:
        raise ValueError("official GRPO validation tasks leaked into adaptive SFT data")

    for split, rows in output.items():
        expected = counts[split] * (1 + replay_ratio)
        if len(rows) != expected:
            raise ValueError(f"{split} expected {expected} rows, found {len(rows)}")
        rows.sort(key=lambda item: item.example_id)

    unique_model_visible_payloads = _assert_unique_model_visible_payloads(output)

    version = "sft-adaptive-recovery-" + hashlib.sha256(
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
            "\n".join(item.model_dump_json() for item in rows) + "\n",
            encoding="utf-8",
        )
    (output_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "derivation.json").write_text(
        json.dumps(
            {
                "schema_version": "adaptive-recovery-sft-derivation.v1",
                "official_source_split": "GRPO train only",
                "official_validation_used_for_training": False,
                "adaptive_examples": sum(counts.values()),
                "replay_examples": sum(replay_counts.values()),
                "replay_ratio": f"1:{replay_ratio}",
                "replay_family_examples": dict(
                    Counter(
                        _replay_family(row)
                        for rows in replay_splits.values()
                        for row in rows
                    )
                ),
                "adaptive_acceptance_gates": [
                    "first QUERY_TOO_BROAD call failed in the real Agent Loop",
                    "failure message is present in the model-visible policy state",
                    "second call changes arguments using one grounded visible interest",
                    "second call is verified successful by the immutable environment",
                    "hidden evaluator facts are absent from serialized messages",
                ],
                "internal_source_task_overlap": sorted(overlap),
                "official_validation_task_overlap": [],
                "audited_examples": len(audits),
                "unique_model_visible_payloads": unique_model_visible_payloads,
                "audit_sample": audits[:12],
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=32)
    parser.add_argument("--replay-ratio", type=int, default=1)
    args = parser.parse_args()
    counts = {
        "train": args.train_count,
        "validation": args.validation_count,
        "test": args.test_count,
    }
    if args.concurrency < 1 or any(value < 1 for value in counts.values()):
        parser.error("concurrency and split counts must be positive")
    manifest = asyncio.run(
        build(
            args.source_dir,
            args.output_dir,
            concurrency=args.concurrency,
            split_counts=counts,
            replay_ratio=args.replay_ratio,
        )
    )
    print(manifest.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
