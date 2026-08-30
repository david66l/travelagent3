"""Build context-unique preferences against verified premature aborts."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_teacher_distillation import (  # noqa: E402
    load_holdout_contract,
    model_payload_hash,
    select_candidate_group,
)

from agentic.distillation import (  # noqa: E402
    TeacherCandidateRecord,
    build_teacher_candidate,
    select_teacher_group,
)
from agentic.environment import TravelAgentEnvironment  # noqa: E402
from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.loop import PolicyAction, PolicyContext  # noqa: E402
from agentic.policy import ControllerFirstPolicy  # noqa: E402


class PrematureAbortPolicy:
    """Abort at the first controller-delegated decision."""

    async def propose(self, context: PolicyContext) -> PolicyAction:
        if "abort" not in context.allowed_actions:
            raise ValueError("premature-abort fixture requires abort to be allowed")
        return PolicyAction(
            action="abort",
            arguments={"reason": "premature termination perturbation"},
        )


def _load_groups(path: Path) -> dict[str, list[TeacherCandidateRecord]]:
    groups: dict[str, list[TeacherCandidateRecord]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            candidate = TeacherCandidateRecord(**json.loads(line))
            groups[candidate.task_id].append(candidate)
    if not groups:
        raise ValueError("teacher candidate file is empty")
    return dict(groups)


def _failure_codes(candidate: TeacherCandidateRecord) -> list[str]:
    codes: set[str] = set()
    for step in candidate.rollout.episode.steps:
        verification_code = str((step.verification or {}).get("error_code") or "")
        if verification_code:
            codes.add(verification_code)
        codes.update(
            str(observation.error.code)
            for observation in step.observations
            if observation.error is not None
        )
    return sorted(codes)


def _response_key(pair: Any) -> str:
    return json.dumps(
        {"chosen": pair.chosen, "rejected": pair.rejected},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


async def build(args: argparse.Namespace) -> dict[str, Any]:
    rows: dict[str, GRPOCorpusRow] = {
        row.task.task_id: row for row in load_grpo_corpus(args.corpus_file)
    }
    groups = _load_groups(args.teacher_candidates)
    unknown = sorted(set(groups) - set(rows))
    if unknown:
        raise ValueError(f"teacher tasks missing from corpus: {unknown[:5]}")
    teacher_by_task = {
        task_id: select_candidate_group(candidates).chosen
        for task_id, candidates in groups.items()
    }
    eligible_task_ids = [
        task_id
        for task_id, teacher in teacher_by_task.items()
        if teacher.family == "clarification"
        and any(
            step.action.decision_source != "controller"
            and "abort" in step.context.allowed_actions
            for step in teacher.rollout.episode.steps
        )
    ]
    skipped_outside_clarification_abort = len(groups) - len(eligible_task_ids)
    task_ids = eligible_task_ids
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    holdout_task_ids, holdout_payload_hashes = load_holdout_contract(
        args.forbidden_holdout_dir
    )
    task_overlap = sorted(set(task_ids) & holdout_task_ids)
    if task_overlap:
        raise ValueError(f"abort tasks overlap frozen holdout: {task_overlap[:5]}")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(task_id: str):
        row = rows[task_id]
        teacher = teacher_by_task[task_id]
        async with semaphore:
            rollout = await TravelAgentEnvironment(row.task, row.snapshot).rollout(
                ControllerFirstPolicy(PrematureAbortPolicy())
            )
        sample_index = max(item.sample_index for item in groups[task_id]) + 1
        negative = build_teacher_candidate(
            rollout,
            family=teacher.family,
            sample_index=sample_index,
        )
        codes = _failure_codes(negative)
        errors: list[str] = []
        if negative.score.successful:
            errors.append("PERTURBATION_UNEXPECTEDLY_SUCCEEDED")
        if "POLICY_ABORT" not in codes:
            errors.append("MISSING_VERIFIED_POLICY_ABORT")
        pair = None
        if not errors:
            selection = select_teacher_group([teacher, negative])
            if len(selection.preference_pairs) != 1:
                errors.append("NO_UNIQUE_TRAINABLE_PREFERENCE")
            else:
                pair = selection.preference_pairs[0]
        return teacher, negative, pair, codes, errors

    results = await asyncio.gather(*(one(task_id) for task_id in task_ids))
    raw_pairs = []
    negatives = []
    selection_rows = []
    errors: list[str] = []
    for index, (task_id, result) in enumerate(zip(task_ids, results, strict=True), start=1):
        teacher, negative, pair, codes, task_errors = result
        negatives.append(negative)
        if pair is not None:
            raw_pairs.append(pair)
        errors.extend(f"{task_id}:{code}" for code in task_errors)
        selection_rows.append(
            {
                "task_id": task_id,
                "family": teacher.family,
                "chosen_trajectory_id": teacher.score.trajectory_id,
                "rejected_trajectory_id": negative.score.trajectory_id,
                "rejected_successful": negative.score.successful,
                "rejected_gate_status": negative.score.gate_status,
                "verified_failure_codes": codes,
                "raw_pair_id": pair.pair_id if pair else None,
                "errors": task_errors,
            }
        )
        if index % 250 == 0 or index == len(task_ids):
            print(f"[{index}/{len(task_ids)}] verified abort rollouts", flush=True)

    by_context: dict[str, list[Any]] = defaultdict(list)
    for pair in raw_pairs:
        by_context[pair.context_hash].append(pair)
    preferences = []
    conflict_rows = []
    exact_context_duplicates_dropped = 0
    for context_hash, pairs in by_context.items():
        response_groups = defaultdict(list)
        for pair in pairs:
            response_groups[_response_key(pair)].append(pair)
        if len(response_groups) != 1:
            conflict_rows.append(
                {
                    "context_hash": context_hash,
                    "pair_ids": sorted(pair.pair_id for pair in pairs),
                    "chosen_responses": [
                        json.loads(key)["chosen"] for key in sorted(response_groups)
                    ],
                }
            )
            continue
        preferences.append(min(pairs, key=lambda item: item.pair_id))
        exact_context_duplicates_dropped += len(pairs) - 1
    preferences.sort(key=lambda item: item.pair_id)

    payload_hashes = [
        model_payload_hash(pair.messages, pair.tools) for pair in preferences
    ]
    payload_overlap = set(payload_hashes) & holdout_payload_hashes
    if len(payload_hashes) != len(set(payload_hashes)):
        errors.append("MODEL_PAYLOAD_DUPLICATE_AFTER_CONTEXT_DEDUP")
    if payload_overlap:
        errors.append(f"FROZEN_HOLDOUT_PAYLOAD_OVERLAP:{len(payload_overlap)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        args.output_dir / "verified_negative_candidates.jsonl",
        [item.model_dump(mode="json") for item in negatives],
    )
    _write_jsonl(
        args.output_dir / "preference_pairs.jsonl",
        [item.model_dump(mode="json") for item in preferences],
    )
    _write_jsonl(args.output_dir / "selections.jsonl", selection_rows)
    _write_jsonl(args.output_dir / "preference_context_conflicts.jsonl", conflict_rows)
    manifest = {
        "schema_version": "verified-abort-preferences.v1",
        "status": "passed" if not errors else "rejected",
        "negative_generation": "executed_in_immutable_environment",
        "perturbation": "premature_abort_at_first_delegated_decision",
        "corpus_file": str(args.corpus_file),
        "teacher_candidates": str(args.teacher_candidates),
        "selected_tasks": len(task_ids),
        "skipped_outside_clarification_abort": skipped_outside_clarification_abort,
        "family_counts": dict(Counter(groups[task_id][0].family for task_id in task_ids)),
        "verified_negative_rollouts": len(negatives),
        "raw_preference_pairs": len(raw_pairs),
        "unique_preference_pairs": len(preferences),
        "unique_contexts": len({pair.context_hash for pair in preferences}),
        "unique_model_payloads": len(set(payload_hashes)),
        "exact_context_duplicates_dropped": exact_context_duplicates_dropped,
        "conflicting_contexts_quarantined": len(conflict_rows),
        "conflicting_pairs_quarantined": sum(
            len(item["pair_ids"]) for item in conflict_rows
        ),
        "frozen_holdout_task_overlap": len(task_overlap),
        "frozen_holdout_payload_overlap": len(payload_overlap),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--teacher-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-holdout-dir", type=Path)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.concurrency < 1 or (args.limit is not None and args.limit < 1):
        parser.error("concurrency and limit must be positive")
    manifest = asyncio.run(build(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
