"""Reverify and merge teacher candidates into one leakage-safe SFT dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from generate_teacher_distillation import (  # noqa: E402
    deduplicate_sft_result,
    load_holdout_contract,
    model_payload_hash,
    select_candidate_group,
)

from agentic.distillation import (  # noqa: E402
    TeacherCandidateRecord,
    score_teacher_rollout,
)
from agentic.reward import HierarchicalRewardEngine  # noqa: E402
from agentic.sft_dataset import (  # noqa: E402
    DatasetBuildResult,
    EpisodeCandidate,
    SFTDatasetBuilder,
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_reverify(
    paths: list[Path],
) -> tuple[dict[str, list[TeacherCandidateRecord]], list[dict[str, Any]]]:
    groups: dict[str, list[TeacherCandidateRecord]] = defaultdict(list)
    seen_trajectories: set[str] = set()
    sources = []
    for path in paths:
        source_rows = 0
        duplicate_trajectories = 0
        currently_successful = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            source_rows += 1
            candidate = TeacherCandidateRecord(**json.loads(line))
            trajectory_id = candidate.rollout.episode.trajectory_id
            if trajectory_id in seen_trajectories:
                duplicate_trajectories += 1
                continue
            seen_trajectories.add(trajectory_id)
            current_reward = HierarchicalRewardEngine().score(candidate.rollout.episode)
            current_rollout = candidate.rollout.model_copy(
                update={"reward": current_reward}
            )
            candidate = candidate.model_copy(
                update={
                    "rollout": current_rollout,
                    "score": score_teacher_rollout(current_rollout),
                }
            )
            currently_successful += int(candidate.score.successful)
            groups[candidate.task_id].append(candidate)
        sources.append(
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "rows": source_rows,
                "duplicate_trajectories_dropped": duplicate_trajectories,
                "currently_successful": currently_successful,
                "currently_failed": source_rows - duplicate_trajectories - currently_successful,
            }
        )
    if not groups:
        raise ValueError("teacher candidate sources are empty")
    return dict(groups), sources


def _scenario_split(scenario_id: str) -> str:
    bucket = int(hashlib.sha256(scenario_id.encode()).hexdigest()[:8], 16) % 10
    return "validation" if bucket == 0 else ("test" if bucket == 1 else "train")


def resplit_by_scenario(result: DatasetBuildResult) -> DatasetBuildResult:
    examples = [
        example.model_copy(update={"split": _scenario_split(example.scenario_id)})
        for example in result.examples
    ]
    manifest = result.manifest.model_copy(
        update={
            "dataset_version": "sft-reverified-"
            + hashlib.sha256(
                json.dumps(
                    [(item.example_id, item.split) for item in examples],
                    sort_keys=True,
                ).encode()
            ).hexdigest()[:16],
            "split_examples": dict(Counter(item.split for item in examples)),
            "split_group_overlap": False,
        }
    )
    return DatasetBuildResult(
        manifest=manifest,
        reviews=result.reviews,
        examples=examples,
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    groups, sources = load_and_reverify(args.source_candidates)
    chosen = []
    failed_groups = []
    for task_id, candidates in groups.items():
        successful = [item for item in candidates if item.score.successful]
        if not successful:
            failed_groups.append(
                {
                    "task_id": task_id,
                    "families": sorted({item.family for item in candidates}),
                    "gate_statuses": sorted({item.score.gate_status for item in candidates}),
                }
            )
            continue
        selection = select_candidate_group(successful)
        candidate = selection.chosen
        chosen.append(
            EpisodeCandidate(
                scenario_id=candidate.task_id,
                source="teacher",
                template_family=candidate.family,
                city="unknown",
                episode=candidate.rollout.episode,
            )
        )

    raw_result = SFTDatasetBuilder().build(chosen)
    result, duplicates_dropped, conflicts = deduplicate_sft_result(raw_result)
    result = resplit_by_scenario(result)
    _, holdout_payloads = load_holdout_contract(args.forbidden_holdout_dir)
    payload_hashes = [
        model_payload_hash(
            [message.model_dump(mode="json") for message in example.messages[:-1]],
            example.tools,
        )
        for example in result.examples
    ]
    holdout_overlap = set(payload_hashes) & holdout_payloads
    errors = []
    if raw_result.manifest.rejected_episodes:
        errors.append("SFT_CHOSEN_EPISODE_REJECTED")
    if len(payload_hashes) != len(set(payload_hashes)):
        errors.append("MODEL_PAYLOAD_DUPLICATE")
    if holdout_overlap:
        errors.append(f"FROZEN_HOLDOUT_PAYLOAD_OVERLAP:{len(holdout_overlap)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    SFTDatasetBuilder().export(result, args.output_dir / "sft")
    _write_jsonl(args.output_dir / "failed_groups.jsonl", failed_groups)
    _write_jsonl(args.output_dir / "sft_label_conflicts.jsonl", conflicts)
    manifest = {
        "schema_version": "reverified-teacher-sft-merge.v1",
        "status": "passed" if not errors else "rejected",
        "sources": sources,
        "candidate_task_groups": len(groups),
        "currently_verified_chosen_episodes": len(chosen),
        "groups_without_current_success": len(failed_groups),
        "failed_family_counts": dict(
            Counter(family for row in failed_groups for family in row["families"])
        ),
        "raw_sft_examples": len(raw_result.examples),
        "sft_examples": len(result.examples),
        "model_payload_duplicates_dropped": duplicates_dropped,
        "model_payload_label_conflicts_quarantined": len(conflicts),
        "model_payload_conflict_rows_quarantined": sum(
            len(item["example_ids"]) for item in conflicts
        ),
        "unique_model_payloads": len(set(payload_hashes)),
        "frozen_holdout_payload_overlap": len(holdout_overlap),
        "sft_manifest": result.manifest.model_dump(mode="json"),
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
    parser.add_argument("--source-candidates", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--forbidden-holdout-dir", type=Path)
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
