"""Generate verifier-ranked teacher trajectories and grounded preference pairs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.distillation import (  # noqa: E402
    TeacherCandidateRecord,
    TeacherGroupSelection,
    build_teacher_candidate,
    select_teacher_group,
)
from agentic.environment import TravelAgentEnvironment  # noqa: E402
from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.policy import ControllerFirstPolicy, NativeToolAgentPolicy  # noqa: E402
from agentic.sft_dataset import (  # noqa: E402
    DatasetBuildResult,
    EpisodeCandidate,
    SFTDatasetBuilder,
)
from core.llm_client import LLMClient  # noqa: E402
from core.settings import settings  # noqa: E402


def task_family(row: GRPOCorpusRow) -> str:
    if row.task.missing_slots:
        return "clarification"
    if row.task.feasibility_report.get("feasible", True) is False:
        return "tradeoff"
    search = row.snapshot.tool_responses.get("search_pois") or []
    if search and (search[0].error_code or search[0].data_source == "unavailable"):
        return "recovery"
    return "search"


def select_stratified(
    rows: list[GRPOCorpusRow],
    *,
    tasks_per_family: int,
    offset_per_family: int,
) -> list[GRPOCorpusRow]:
    selected: dict[str, list[GRPOCorpusRow]] = defaultdict(list)
    seen: Counter[str] = Counter()
    for row in rows:
        family = task_family(row)
        if seen[family] < offset_per_family:
            seen[family] += 1
            continue
        if len(selected[family]) < tasks_per_family:
            selected[family].append(row)
    return [row for family in sorted(selected) for row in selected[family]]


def select_candidate_group(
    candidates: list[TeacherCandidateRecord],
) -> TeacherGroupSelection:
    """Accept one verified rollout for scale, or rank a sampled candidate group."""
    if len(candidates) != 1:
        return select_teacher_group(candidates)
    chosen = candidates[0]
    if not chosen.score.successful:
        raise ValueError("single teacher candidate failed verifier gates")
    return TeacherGroupSelection(
        task_id=chosen.task_id,
        chosen=chosen,
        rejected=[],
        preference_pairs=[],
    )


def load_holdout_contract(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    task_ids: set[str] = set()
    payload_hashes: set[str] = set()
    for split in ("regular", "hard", "adversarial"):
        split_path = path / f"{split}.jsonl"
        if not split_path.is_file():
            raise FileNotFoundError(f"missing frozen holdout split: {split_path}")
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            task_ids.add(str(row["case_id"]))
            payload_hashes.add(model_payload_hash(row["messages"], row["tools"]))
    return task_ids, payload_hashes


def deduplicate_sft_result(
    result: DatasetBuildResult,
) -> tuple[DatasetBuildResult, int, list[dict[str, Any]]]:
    """Keep one deterministic label per model-visible prompt and tool schema."""
    groups: dict[str, list[Any]] = defaultdict(list)
    for example in result.examples:
        prompt_hash = model_payload_hash(
            [message.model_dump(mode="json") for message in example.messages[:-1]],
            example.tools,
        )
        groups[prompt_hash].append(example)

    selected = []
    conflicts: list[dict[str, Any]] = []
    duplicates_dropped = 0
    for prompt_hash, examples in groups.items():
        responses = {
            _canonical_hash(example.messages[-1].model_dump(mode="json")): (
                example.messages[-1].model_dump(mode="json")
            )
            for example in examples
        }
        if len(responses) != 1:
            conflicts.append(
                {
                    "prompt_hash": prompt_hash,
                    "example_ids": sorted(example.example_id for example in examples),
                    "responses": list(responses.values()),
                }
            )
            continue
        selected.append(min(examples, key=lambda item: item.example_id))
        duplicates_dropped += len(examples) - 1
    selected.sort(key=lambda item: item.example_id)

    retained_by_trajectory = Counter(item.trajectory_id for item in selected)
    reviews = [
        review.model_copy(
            update={"example_count": retained_by_trajectory[review.trajectory_id]}
        )
        for review in result.reviews
    ]
    manifest = result.manifest.model_copy(
        update={
            "dataset_version": "sft-distilled-"
            + _canonical_hash([item.example_id for item in selected])[:16],
            "exported_examples": len(selected),
            "split_examples": dict(Counter(item.split for item in selected)),
        }
    )
    return (
        DatasetBuildResult(
            manifest=manifest,
            reviews=reviews,
            examples=selected,
        ),
        duplicates_dropped,
        conflicts,
    )


async def generate(args: argparse.Namespace) -> dict[str, Any]:
    rows = select_stratified(
        load_grpo_corpus(args.corpus_file),
        tasks_per_family=args.tasks_per_family,
        offset_per_family=args.family_offset,
    )
    if not rows:
        raise ValueError("no teacher tasks selected")
    holdout_task_ids, holdout_payload_hashes = load_holdout_contract(
        args.forbidden_holdout_dir
    )
    selected_task_ids = {row.task.task_id for row in rows}
    task_overlap = sorted(selected_task_ids & holdout_task_ids)
    if task_overlap:
        raise ValueError(f"teacher tasks overlap frozen holdout: {task_overlap[:5]}")

    settings.vllm_enabled = True
    settings.vllm_base_url = args.base_url
    settings.vllm_api_key = os.environ.get(args.api_key_env, "not-needed")
    client = LLMClient()
    policy = ControllerFirstPolicy(
        NativeToolAgentPolicy(
            client,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(
        row: GRPOCorpusRow,
        sample_index: int,
    ) -> TeacherCandidateRecord:
        async with semaphore:
            rollout = await TravelAgentEnvironment(row.task, row.snapshot).rollout(policy)
            return build_teacher_candidate(
                rollout,
                family=task_family(row),
                sample_index=sample_index,
            )

    async def one_group(
        task_index: int,
        row: GRPOCorpusRow,
    ) -> tuple[str, list[TeacherCandidateRecord]]:
        candidates = await asyncio.gather(
            *(one(row, sample_index) for sample_index in range(args.candidates_per_task))
        )
        print(
            f"[{task_index}/{len(rows)}] {row.task.task_id} "
            f"family={task_family(row)} rewards="
            f"{[item.score.episode_reward for item in candidates]}",
            flush=True,
        )
        return row.task.task_id, list(candidates)

    # The semaphore bounds actual model requests across task boundaries.  This
    # keeps formal one-candidate generation parallel instead of accidentally
    # becoming serial while retaining per-task candidate groups for ranking.
    group_results = await asyncio.gather(
        *(one_group(index, row) for index, row in enumerate(rows, start=1))
    )
    candidate_groups = dict(group_results)

    chosen_candidates: list[EpisodeCandidate] = []
    preference_by_id = {}
    raw_preference_pairs = 0
    selection_rows: list[dict[str, Any]] = []
    failed_groups: list[dict[str, Any]] = []
    for row in rows:
        candidates = candidate_groups[row.task.task_id]
        try:
            selection = select_candidate_group(candidates)
        except ValueError as exc:
            failed_groups.append(
                {
                    "task_id": row.task.task_id,
                    "family": task_family(row),
                    "reason": str(exc),
                    "scores": [item.score.model_dump(mode="json") for item in candidates],
                }
            )
            continue
        chosen_candidates.append(
            EpisodeCandidate(
                scenario_id=row.task.task_id,
                source="teacher",
                template_family=row.task.template_family,
                city=str(row.task.slots.get("destination") or "unknown"),
                episode=selection.chosen.rollout.episode,
            )
        )
        raw_preference_pairs += len(selection.preference_pairs)
        for pair in selection.preference_pairs:
            preference_by_id.setdefault(pair.pair_id, pair)
        selection_rows.append(
            {
                "task_id": row.task.task_id,
                "family": task_family(row),
                "chosen_trajectory_id": selection.chosen.rollout.episode.trajectory_id,
                "chosen_score": selection.chosen.score.model_dump(mode="json"),
                "rejected": [
                    item.score.model_dump(mode="json") for item in selection.rejected
                ],
                "preference_pair_ids": [item.pair_id for item in selection.preference_pairs],
            }
        )

    preferences = list(preference_by_id.values())
    raw_sft_result = SFTDatasetBuilder().build(chosen_candidates)
    sft_result, duplicate_payloads_dropped, label_conflicts = deduplicate_sft_result(
        raw_sft_result
    )
    prompt_hashes = [
        model_payload_hash(
            [message.model_dump(mode="json") for message in example.messages[:-1]],
            example.tools,
        )
        for example in sft_result.examples
    ]
    holdout_payload_overlap = sorted(set(prompt_hashes) & holdout_payload_hashes)
    errors: list[str] = []
    if failed_groups:
        errors.append(f"TEACHER_GROUPS_WITHOUT_VERIFIED_SUCCESS:{len(failed_groups)}")
    if raw_sft_result.manifest.rejected_episodes:
        errors.append("SFT_CHOSEN_EPISODE_REJECTED")
    if holdout_payload_overlap:
        errors.append("FROZEN_HOLDOUT_PAYLOAD_OVERLAP")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        args.output_dir / "teacher_candidates.jsonl",
        [
            item.model_dump(mode="json")
            for row in rows
            for item in candidate_groups[row.task.task_id]
        ],
    )
    _write_jsonl(
        args.output_dir / "chosen_episodes.jsonl",
        [item.model_dump(mode="json") for item in chosen_candidates],
    )
    _write_jsonl(
        args.output_dir / "preference_pairs.jsonl",
        [item.model_dump(mode="json") for item in preferences],
    )
    _write_jsonl(args.output_dir / "selections.jsonl", selection_rows)
    _write_jsonl(args.output_dir / "failed_groups.jsonl", failed_groups)
    _write_jsonl(args.output_dir / "sft_label_conflicts.jsonl", label_conflicts)
    SFTDatasetBuilder().export(sft_result, args.output_dir / "sft")

    all_candidates = [
        item for group in candidate_groups.values() for item in group
    ]
    manifest = {
        "schema_version": "teacher-distillation-build.v1",
        "status": "passed" if not errors else "rejected",
        "model": args.model,
        "base_url": args.base_url,
        "corpus_file": str(args.corpus_file),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "candidates_per_task": args.candidates_per_task,
        "selection_mode": (
            "verified_single_candidate"
            if args.candidates_per_task == 1
            else "verifier_ranked_multi_candidate"
        ),
        "concurrency": args.concurrency,
        "selected_tasks": len(rows),
        "families": dict(Counter(task_family(row) for row in rows)),
        "candidate_rollouts": len(all_candidates),
        "successful_candidate_rollouts": sum(item.score.successful for item in all_candidates),
        "chosen_episodes": len(chosen_candidates),
        "failed_groups": len(failed_groups),
        "preference_pairs": len(preferences),
        "raw_preference_pairs": raw_preference_pairs,
        "duplicate_preference_pairs_dropped": raw_preference_pairs - len(preferences),
        "sft_examples": len(sft_result.examples),
        "raw_sft_examples": len(raw_sft_result.examples),
        "model_payload_duplicates_dropped": duplicate_payloads_dropped,
        "model_payload_label_conflicts_quarantined": len(label_conflicts),
        "model_payload_conflict_rows_quarantined": sum(
            len(item["example_ids"]) for item in label_conflicts
        ),
        "unique_model_payloads": len(set(prompt_hashes)),
        "model_payloads": len(prompt_hashes),
        "frozen_holdout_task_overlap": len(task_overlap),
        "frozen_holdout_payload_overlap": len(holdout_payload_overlap),
        "sft_manifest": sft_result.manifest.model_dump(mode="json"),
        "raw_sft_manifest": raw_sft_result.manifest.model_dump(mode="json"),
        "errors": errors,
    }
    (args.output_dir / "manifest.json").write_text(
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


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def model_payload_hash(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    """Hash semantically identical prompts independent of Pydantic defaults."""
    normalized_messages = []
    for message in messages:
        normalized_messages.append(
            {
                key: value
                for key, value in message.items()
                if value is not None and not (key == "tool_calls" and not value)
            }
        )
    return _canonical_hash({"messages": normalized_messages, "tools": tools})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-holdout-dir", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks-per-family", type=int, default=4)
    parser.add_argument("--family-offset", type=int, default=0)
    parser.add_argument("--candidates-per-task", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()
    if min(
        args.tasks_per_family,
        args.candidates_per_task,
        args.concurrency,
        args.max_tokens,
    ) < 1:
        parser.error("task, candidate, concurrency and token counts must be positive")
    if args.family_offset < 0 or args.temperature <= 0:
        parser.error("family-offset must be non-negative and temperature must be positive")
    manifest = asyncio.run(generate(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
