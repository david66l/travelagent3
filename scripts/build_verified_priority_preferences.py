"""Build verifier-grounded preference pairs for priority-search decisions.

The rejected response is never fabricated by editing a successful trajectory.
It is executed against the same immutable environment snapshot and retained only
when the snapshot verifier attributes the failure to the perturbed model action.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_teacher_distillation import select_candidate_group  # noqa: E402

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


class PrioritySearchPerturbationPolicy(CurriculumTeacherPolicy):
    """Keep using a visible broad query even when one keyword is required."""

    async def propose(self, context: PolicyContext) -> PolicyAction:
        if "search_pois" not in context.allowed_actions:
            return await super().propose(context)
        interests = [
            str(item)
            for item in context.soft_preferences.get("interests") or []
            if str(item)
        ]
        if len(interests) < 2:
            raise ValueError("priority perturbation requires two visible interests")
        return PolicyAction(
            action="search_pois",
            arguments={"keywords": interests[:2]},
        )


def _load_teacher_groups(
    path: Path,
) -> dict[str, list[TeacherCandidateRecord]]:
    groups: dict[str, list[TeacherCandidateRecord]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            candidate = TeacherCandidateRecord(**json.loads(line))
            groups[candidate.task_id].append(candidate)
    if not groups:
        raise ValueError("teacher candidate file is empty")
    return dict(groups)


def _target_contract(row: GRPOCorpusRow, perturbation: str) -> list[str]:
    if perturbation == "priority_search":
        contract_name = "priority_search"
        response_index = 0
    elif perturbation == "recovery_repeat":
        contract_name = "adaptive_recovery"
        response_index = 1
    else:  # protected by argparse; retained for direct API callers
        raise ValueError(f"unsupported perturbation: {perturbation}")
    contract = row.snapshot.hidden_test_facts.get(contract_name) or {}
    target = [str(item) for item in contract.get("target_keywords") or []]
    responses = row.snapshot.tool_responses.get("search_pois") or []
    expected = (
        responses[response_index].expected_arguments.get("keywords")
        if len(responses) > response_index
        else None
    )
    if len(target) != 1 or expected != target:
        raise ValueError(
            f"{row.task.task_id} lacks a consistent {contract_name} contract"
        )
    if perturbation == "recovery_repeat" and (
        not responses[0].retryable or responses[0].error_code != "QUERY_TOO_BROAD"
    ):
        raise ValueError(
            f"{row.task.task_id} lacks a retryable QUERY_TOO_BROAD first response"
        )
    return target


def _failure_codes(candidate: TeacherCandidateRecord) -> list[str]:
    codes = {
        str(observation.error.code)
        for step in candidate.rollout.episode.steps
        for observation in step.observations
        if observation.error is not None
    }
    return sorted(codes)


def _model_payload_hash(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> str:
    normalized = [
        {
            key: value
            for key, value in message.items()
            if value is not None and not (key == "tool_calls" and not value)
        }
        for message in messages
    ]
    return _canonical_hash({"messages": normalized, "tools": tools})


def _load_holdout(path: Path | None) -> tuple[set[str], set[str]]:
    if path is None:
        return set(), set()
    task_ids: set[str] = set()
    payload_hashes: set[str] = set()
    for split in ("regular", "hard", "adversarial"):
        split_path = path / f"{split}.jsonl"
        if not split_path.is_file():
            raise FileNotFoundError(f"missing frozen holdout split: {split_path}")
        for line in split_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                task_ids.add(str(item["case_id"]))
                payload_hashes.add(
                    _model_payload_hash(item["messages"], item["tools"])
                )
    return task_ids, payload_hashes


async def build(args: argparse.Namespace) -> dict[str, Any]:
    perturbation = getattr(args, "perturbation", "priority_search")
    rows = {
        row.task.task_id: row
        for row in load_grpo_corpus(args.corpus_file)
    }
    teacher_groups = _load_teacher_groups(args.teacher_candidates)
    unknown = sorted(set(teacher_groups) - set(rows))
    if unknown:
        raise ValueError(f"teacher tasks missing from corpus: {unknown[:5]}")
    task_ids = list(teacher_groups)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    holdout_task_ids, holdout_payload_hashes = _load_holdout(
        args.forbidden_holdout_dir
    )
    task_overlap = sorted(set(task_ids) & holdout_task_ids)
    if task_overlap:
        raise ValueError(f"preference tasks overlap frozen holdout: {task_overlap[:5]}")

    negative_candidates: list[TeacherCandidateRecord] = []
    preference_by_id = {}
    selection_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, task_id in enumerate(task_ids, start=1):
        row = rows[task_id]
        target = _target_contract(row, perturbation)
        teacher_selection = select_candidate_group(teacher_groups[task_id])
        sample_index = max(item.sample_index for item in teacher_groups[task_id]) + 1
        rollout = await TravelAgentEnvironment(row.task, row.snapshot).rollout(
            ControllerFirstPolicy(PrioritySearchPerturbationPolicy())
        )
        negative = build_teacher_candidate(
            rollout,
            family=teacher_selection.chosen.family,
            sample_index=sample_index,
        )
        failure_codes = _failure_codes(negative)
        task_errors: list[str] = []
        if negative.score.successful:
            task_errors.append("PERTURBATION_UNEXPECTEDLY_SUCCEEDED")
        if "SNAPSHOT_ARGUMENT_MISMATCH" not in failure_codes:
            task_errors.append("MISSING_VERIFIED_ARGUMENT_MISMATCH")
        pair = None
        if not task_errors:
            combined = select_teacher_group([teacher_selection.chosen, negative])
            if len(combined.preference_pairs) != 1:
                task_errors.append("NO_UNIQUE_TRAINABLE_PREFERENCE")
            else:
                pair = combined.preference_pairs[0]
                preference_by_id.setdefault(pair.pair_id, pair)
                negative_candidates.append(negative)

        errors.extend(f"{task_id}:{code}" for code in task_errors)
        selection_rows.append(
            {
                "task_id": task_id,
                "target_keywords": target,
                "perturbed_keywords": list(
                    row.task.profile.get("interests")
                    or row.task.slots.get("interests")
                    or []
                )[:2],
                "chosen_trajectory_id": teacher_selection.chosen.score.trajectory_id,
                "rejected_trajectory_id": negative.score.trajectory_id,
                "rejected_successful": negative.score.successful,
                "rejected_gate_status": negative.score.gate_status,
                "rejected_reward": negative.score.episode_reward,
                "verified_failure_codes": failure_codes,
                "preference_pair_id": pair.pair_id if pair else None,
                "errors": task_errors,
            }
        )
        print(
            f"[{index}/{len(task_ids)}] {task_id} target={target} "
            f"failure_codes={failure_codes} pair={pair.pair_id if pair else None}",
            flush=True,
        )

    preferences = list(preference_by_id.values())
    payload_hashes = [
        _model_payload_hash(pair.messages, pair.tools) for pair in preferences
    ]
    payload_overlap = sorted(set(payload_hashes) & holdout_payload_hashes)
    if len(preferences) != len(task_ids):
        errors.append("PREFERENCE_COUNT_DOES_NOT_MATCH_TASK_COUNT")
    if len(payload_hashes) != len(set(payload_hashes)):
        errors.append("MODEL_PAYLOAD_DUPLICATE")
    if payload_overlap:
        errors.append("FROZEN_HOLDOUT_PAYLOAD_OVERLAP")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        args.output_dir / "verified_negative_candidates.jsonl",
        [item.model_dump(mode="json") for item in negative_candidates],
    )
    _write_jsonl(
        args.output_dir / "preference_pairs.jsonl",
        [item.model_dump(mode="json") for item in preferences],
    )
    _write_jsonl(args.output_dir / "selections.jsonl", selection_rows)
    manifest = {
        "schema_version": (
            "verified-priority-preferences.v1"
            if perturbation == "priority_search"
            else "verified-recovery-preferences.v1"
        ),
        "status": "passed" if not errors else "rejected",
        "negative_generation": "executed_in_immutable_environment",
        "perturbation": perturbation,
        "perturbation_description": (
            "use two grounded interests instead of the explicit single priority"
            if perturbation == "priority_search"
            else "repeat the broad query after grounded narrowing feedback"
        ),
        "corpus_file": str(args.corpus_file),
        "teacher_candidates": str(args.teacher_candidates),
        "teacher_candidates_sha256": _file_sha256(args.teacher_candidates),
        "selected_tasks": len(task_ids),
        "verified_negative_rollouts": len(negative_candidates),
        "preference_pairs": len(preferences),
        "unique_model_payloads": len(set(payload_hashes)),
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


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-file", type=Path, required=True)
    parser.add_argument("--teacher-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-holdout-dir", type=Path)
    parser.add_argument(
        "--perturbation",
        choices=("priority_search", "recovery_repeat"),
        default="priority_search",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    manifest = asyncio.run(build(args))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
