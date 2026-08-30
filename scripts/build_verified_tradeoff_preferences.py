"""Build strong preferences for proposing a tradeoff instead of fake continuation."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
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

from agentic.corpus_generation import CurriculumTeacherPolicy  # noqa: E402
from agentic.distillation import (  # noqa: E402
    TeacherCandidateRecord,
    build_teacher_candidate,
    select_teacher_group,
)
from agentic.environment import TravelAgentEnvironment  # noqa: E402
from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.loop import PolicyAction, PolicyContext  # noqa: E402
from agentic.policy import ControllerFirstPolicy  # noqa: E402


class FakeContinuationPolicy(CurriculumTeacherPolicy):
    """Continue an infeasible request through capability_check instead of stopping."""

    async def propose(self, context: PolicyContext) -> PolicyAction:
        if context.capability.get("status") == "infeasible":
            task_id = context.current_subtask.get("task_id")
            if task_id == "capability_check" and "capability_check" in context.allowed_actions:
                return PolicyAction(action="capability_check")
            if task_id == "search_candidates":
                has_candidates = any(
                    item.get("artifact_type") == "poi_candidate_set"
                    for item in context.relevant_artifacts
                )
                return PolicyAction(
                    action="accept_candidates" if has_candidates else "search_pois",
                    arguments={} if has_candidates else {"keywords": []},
                )
            if task_id == "review_itinerary" and "accept_itinerary" in context.allowed_actions:
                return PolicyAction(action="accept_itinerary")
        return await super().propose(context)


def _load_groups(path: Path) -> dict[str, list[TeacherCandidateRecord]]:
    groups: dict[str, list[TeacherCandidateRecord]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            candidate = TeacherCandidateRecord(**json.loads(line))
            groups[candidate.task_id].append(candidate)
    if not groups:
        raise ValueError("teacher candidate file is empty")
    return dict(groups)


def _first_policy_action(candidate: TeacherCandidateRecord) -> PolicyAction | None:
    return next(
        (
            step.action
            for step in candidate.rollout.episode.steps
            if step.action.decision_source != "controller"
        ),
        None,
    )


def _valid_teacher_tradeoff(candidate: TeacherCandidateRecord) -> bool:
    action = _first_policy_action(candidate)
    if action is None or action.action != "propose_tradeoff":
        return False
    reason = str(action.arguments.get("reason") or "").strip()
    options = [str(item).strip() for item in action.arguments.get("options") or []]
    return bool(reason and 1 <= len(options) <= 3 and all(options) and len(set(options)) == len(options))


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
    task_ids = list(groups)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    holdout_task_ids, holdout_payload_hashes = load_holdout_contract(
        args.forbidden_holdout_dir
    )
    task_overlap = sorted(set(task_ids) & holdout_task_ids)
    if task_overlap:
        raise ValueError(f"tradeoff tasks overlap frozen holdout: {task_overlap[:5]}")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(task_id: str):
        try:
            teacher = select_candidate_group(groups[task_id]).chosen
        except ValueError:
            teacher = groups[task_id][0]
            return teacher, None, None, ["TEACHER_FAILED_VERIFIER_GATES"]
        if not _valid_teacher_tradeoff(teacher):
            return teacher, None, None, ["TEACHER_DID_NOT_PROPOSE_VALID_TRADEOFF"]
        async with semaphore:
            rollout = await TravelAgentEnvironment(rows[task_id].task, rows[task_id].snapshot).rollout(
                ControllerFirstPolicy(FakeContinuationPolicy())
            )
        negative = build_teacher_candidate(
            rollout,
            family="tradeoff",
            sample_index=max(item.sample_index for item in groups[task_id]) + 1,
        )
        errors: list[str] = []
        if negative.score.successful:
            errors.append("FAKE_CONTINUATION_UNEXPECTEDLY_SUCCEEDED")
        if not negative.rollout.reward.audit_metrics.get("capability_termination_mismatch"):
            errors.append("MISSING_CAPABILITY_TERMINATION_MISMATCH")
        pair = None
        if not errors:
            selection = select_teacher_group([teacher, negative])
            if len(selection.preference_pairs) != 1:
                errors.append("NO_UNIQUE_TRAINABLE_PREFERENCE")
            else:
                pair = selection.preference_pairs[0]
        return teacher, negative, pair, errors

    results = await asyncio.gather(*(one(task_id) for task_id in task_ids))
    negatives = []
    preferences = []
    selections = []
    errors: list[str] = []
    skipped_teacher = 0
    for task_id, (teacher, negative, pair, task_errors) in zip(task_ids, results, strict=True):
        if negative is None:
            skipped_teacher += 1
        else:
            negatives.append(negative)
        if pair is not None:
            preferences.append(pair)
        errors.extend(f"{task_id}:{code}" for code in task_errors if not code.startswith("TEACHER_"))
        selections.append(
            {
                "task_id": task_id,
                "teacher_action": (
                    _first_policy_action(teacher).action if _first_policy_action(teacher) else None
                ),
                "teacher_valid_tradeoff": _valid_teacher_tradeoff(teacher),
                "negative_gate_status": negative.score.gate_status if negative else None,
                "capability_termination_mismatch": bool(
                    negative
                    and negative.rollout.reward.audit_metrics.get("capability_termination_mismatch")
                ),
                "preference_pair_id": pair.pair_id if pair else None,
                "errors": task_errors,
            }
        )

    raw_preference_count = len(preferences)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for pair in preferences:
        grouped[pair.context_hash].append(pair)
    deduplicated = []
    conflicts = []
    exact_context_duplicates_dropped = 0
    for context_hash, pairs in grouped.items():
        response_groups: dict[str, list[Any]] = defaultdict(list)
        for pair in pairs:
            response_groups[_response_key(pair)].append(pair)
        if len(response_groups) != 1:
            conflicts.append(
                {
                    "context_hash": context_hash,
                    "pair_ids": sorted(pair.pair_id for pair in pairs),
                    "response_variants": len(response_groups),
                }
            )
            continue
        deduplicated.append(min(pairs, key=lambda item: item.pair_id))
        exact_context_duplicates_dropped += len(pairs) - 1
    preferences = sorted(deduplicated, key=lambda item: item.pair_id)
    payload_hashes = [model_payload_hash(pair.messages, pair.tools) for pair in preferences]
    payload_overlap = set(payload_hashes) & holdout_payload_hashes
    if len(payload_hashes) != len(set(payload_hashes)):
        errors.append("MODEL_PAYLOAD_DUPLICATE")
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
    _write_jsonl(args.output_dir / "selections.jsonl", selections)
    _write_jsonl(args.output_dir / "preference_context_conflicts.jsonl", conflicts)
    manifest = {
        "schema_version": "verified-tradeoff-preferences.v1",
        "status": "passed" if not errors else "rejected",
        "selected_tasks": len(task_ids),
        "teacher_valid_tradeoffs": len(task_ids) - skipped_teacher,
        "teacher_invalid_tradeoffs_skipped": skipped_teacher,
        "verified_negative_rollouts": len(negatives),
        "raw_preference_pairs": raw_preference_count,
        "preference_pairs": len(preferences),
        "unique_contexts": len({pair.context_hash for pair in preferences}),
        "unique_model_payloads": len(set(payload_hashes)),
        "exact_context_duplicates_dropped": exact_context_duplicates_dropped,
        "conflicting_contexts_quarantined": len(conflicts),
        "conflicting_pairs_quarantined": sum(len(item["pair_ids"]) for item in conflicts),
        "frozen_holdout_task_overlap": len(task_overlap),
        "frozen_holdout_payload_overlap": len(payload_overlap),
        "negative_generation": "executed_fake_continuation_in_immutable_environment",
        "verifier_gate": "capability_termination_mismatch",
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
