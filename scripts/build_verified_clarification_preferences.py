"""Replace stale needs-user capability checks with verified ask-user decisions."""

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
    score_teacher_rollout,
    select_teacher_group,
)
from agentic.environment import TravelAgentEnvironment  # noqa: E402
from agentic.grpo_training import GRPOCorpusRow, load_grpo_corpus  # noqa: E402
from agentic.loop import PolicyAction, PolicyContext  # noqa: E402
from agentic.policy import ControllerFirstPolicy  # noqa: E402
from agentic.reward import HierarchicalRewardEngine  # noqa: E402


FIELD_LABELS = {
    "budget_range": "旅行预算范围",
    "destination": "旅行目的地",
    "start_date": "出发日期",
    "end_date": "结束日期",
    "travel_days": "旅行天数",
}


class AskMissingInformationPolicy:
    async def propose(self, context: PolicyContext) -> PolicyAction:
        if "ask_user" not in context.allowed_actions or not context.missing_information:
            raise ValueError("clarification repair requires missing information and ask_user")
        labels = [FIELD_LABELS.get(item, item) for item in context.missing_information]
        return PolicyAction(
            action="ask_user",
            arguments={"question": f"请补充您的{'、'.join(labels)}。"},
        )


def _load_groups(path: Path) -> dict[str, list[TeacherCandidateRecord]]:
    groups: dict[str, list[TeacherCandidateRecord]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        candidate = TeacherCandidateRecord(**json.loads(line))
        reward = HierarchicalRewardEngine().score(candidate.rollout.episode)
        rollout = candidate.rollout.model_copy(update={"reward": reward})
        candidate = candidate.model_copy(
            update={"rollout": rollout, "score": score_teacher_rollout(rollout)}
        )
        groups[candidate.task_id].append(candidate)
    if not groups:
        raise ValueError("teacher candidate file is empty")
    return dict(groups)


def _contains_stale_needs_user_check(candidate: TeacherCandidateRecord) -> bool:
    return any(
        step.action.decision_source != "controller"
        and step.action.action == "capability_check"
        and str(step.context.capability.get("status") or "") == "needs_user"
        and bool(step.context.missing_information)
        for step in candidate.rollout.episode.steps
    )


async def build(args: argparse.Namespace) -> dict[str, Any]:
    rows: dict[str, GRPOCorpusRow] = {
        row.task.task_id: row for row in load_grpo_corpus(args.corpus_file)
    }
    groups = _load_groups(args.teacher_candidates)
    stale_by_task: dict[str, TeacherCandidateRecord] = {}
    for task_id, candidates in groups.items():
        stale = [item for item in candidates if _contains_stale_needs_user_check(item)]
        if stale:
            stale_by_task[task_id] = min(
                stale,
                key=lambda item: (item.score.policy_steps, item.score.trajectory_id),
            )
    unknown = sorted(set(stale_by_task) - set(rows))
    if unknown:
        raise ValueError(f"teacher tasks missing from corpus: {unknown[:5]}")
    task_ids = sorted(stale_by_task)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    holdout_task_ids, holdout_payload_hashes = load_holdout_contract(
        args.forbidden_holdout_dir
    )
    task_overlap = sorted(set(task_ids) & holdout_task_ids)
    if task_overlap:
        raise ValueError(f"clarification tasks overlap frozen holdout: {task_overlap[:5]}")

    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(task_id: str):
        row = rows[task_id]
        stale = stale_by_task[task_id]
        async with semaphore:
            rollout = await TravelAgentEnvironment(row.task, row.snapshot).rollout(
                ControllerFirstPolicy(AskMissingInformationPolicy())
            )
        corrected = build_teacher_candidate(
            rollout,
            family="clarification",
            sample_index=max(item.sample_index for item in groups[task_id]) + 1,
        )
        errors = []
        if not corrected.score.successful:
            errors.append("CORRECTED_ASK_USER_DID_NOT_SUCCEED")
        if stale.score.successful or stale.score.gate_status != "task_failed":
            errors.append("STALE_CAPABILITY_CHECK_NOT_REJECTED")
        pair = None
        if not errors:
            selection = select_teacher_group([corrected, stale])
            if len(selection.preference_pairs) != 1:
                errors.append("NO_UNIQUE_TRAINABLE_PREFERENCE")
            else:
                pair = selection.preference_pairs[0]
        return corrected, stale, pair, errors

    results = await asyncio.gather(*(one(task_id) for task_id in task_ids))
    corrected_candidates = []
    stale_candidates = []
    raw_pairs = []
    errors: list[str] = []
    for task_id, (corrected, stale, pair, task_errors) in zip(
        task_ids, results, strict=True
    ):
        corrected_candidates.append(corrected)
        stale_candidates.append(stale)
        if pair is not None:
            raw_pairs.append(pair)
        errors.extend(f"{task_id}:{code}" for code in task_errors)

    by_context: dict[str, list[Any]] = defaultdict(list)
    for pair in raw_pairs:
        by_context[pair.context_hash].append(pair)
    preferences = []
    conflicts = []
    duplicates_dropped = 0
    for context_hash, pairs in by_context.items():
        responses = {
            json.dumps(
                {"chosen": pair.chosen, "rejected": pair.rejected},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for pair in pairs
        }
        if len(responses) != 1:
            conflicts.append(
                {"context_hash": context_hash, "pair_ids": sorted(p.pair_id for p in pairs)}
            )
            continue
        preferences.append(min(pairs, key=lambda item: item.pair_id))
        duplicates_dropped += len(pairs) - 1
    preferences.sort(key=lambda item: item.pair_id)

    payload_hashes = [model_payload_hash(pair.messages, pair.tools) for pair in preferences]
    payload_overlap = set(payload_hashes) & holdout_payload_hashes
    if len(payload_hashes) != len(set(payload_hashes)):
        errors.append("MODEL_PAYLOAD_DUPLICATE_AFTER_CONTEXT_DEDUP")
    if payload_overlap:
        errors.append(f"FROZEN_HOLDOUT_PAYLOAD_OVERLAP:{len(payload_overlap)}")
    if any("VERIFIER_SUCCESS_OVER_FAILURE" not in pair.reason_codes for pair in preferences):
        errors.append("NON_VERIFIER_PREFERENCE")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "corrected_candidates.jsonl", corrected_candidates)
    _write_jsonl(args.output_dir / "rejected_stale_candidates.jsonl", stale_candidates)
    _write_jsonl(args.output_dir / "preference_pairs.jsonl", preferences)
    _write_jsonl(args.output_dir / "preference_context_conflicts.jsonl", conflicts)
    manifest = {
        "schema_version": "verified-clarification-preferences.v1",
        "status": "passed" if not errors else "rejected",
        "positive_generation": "ask_user executed in immutable environment",
        "negative_generation": "stale capability_check reverified by current reward gate",
        "eligible_tasks": len(stale_by_task),
        "selected_tasks": len(task_ids),
        "verified_corrected_rollouts": sum(item.score.successful for item in corrected_candidates),
        "rejected_stale_rollouts": sum(not item.score.successful for item in stale_candidates),
        "raw_preference_pairs": len(raw_pairs),
        "unique_preference_pairs": len(preferences),
        "unique_contexts": len({pair.context_hash for pair in preferences}),
        "unique_model_payloads": len(set(payload_hashes)),
        "exact_context_duplicates_dropped": duplicates_dropped,
        "conflicting_contexts_quarantined": len(conflicts),
        "reason_counts": dict(Counter(r for p in preferences for r in p.reason_codes)),
        "frozen_holdout_task_overlap": len(task_overlap),
        "frozen_holdout_payload_overlap": len(payload_overlap),
        "errors": errors,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    payloads = [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item
        for item in rows
    ]
    path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in payloads)
        + ("\n" if payloads else ""),
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
