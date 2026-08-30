"""Build verifier-grounded SFT/DPO data from multiple teacher candidate runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend" / "src"))

from agentic.cascade_distillation import (  # noqa: E402
    CascadeSelection,
    CascadeTeacherCandidate,
    TeacherProvenance,
    contribution_summary,
    select_cascade_group,
)
from agentic.distillation import TeacherCandidateRecord  # noqa: E402
from agentic.sft_dataset import (  # noqa: E402
    DatasetBuildResult,
    EpisodeCandidate,
    SFTDatasetBuilder,
)
def read_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("teacher_runs"):
        raise ValueError("teacher_runs must not be empty")
    run_ids = [str(item["run_id"]) for item in config["teacher_runs"]]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("teacher run_ids must be unique")
    return config


def load_candidates(
    config: dict[str, Any], *, config_dir: Path
) -> list[CascadeTeacherCandidate]:
    candidates: list[CascadeTeacherCandidate] = []
    for run in config["teacher_runs"]:
        candidate_file = _resolve(config_dir, run["candidates_file"])
        provenance = TeacherProvenance(
            teacher_id=run["teacher_id"],
            model=run["model"],
            checkpoint=run["checkpoint"],
            tier=run["tier"],
            run_id=run["run_id"],
        )
        for line in candidate_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            candidates.append(
                CascadeTeacherCandidate(
                    provenance=provenance,
                    candidate=TeacherCandidateRecord.model_validate_json(line),
                )
            )
    return candidates


def select_groups(
    candidates: list[CascadeTeacherCandidate],
    *,
    chosen_trajectory_ids: set[str] | None = None,
) -> tuple[list[CascadeSelection], list[dict[str, Any]]]:
    groups: dict[str, list[CascadeTeacherCandidate]] = defaultdict(list)
    for item in candidates:
        groups[item.candidate.task_id].append(item)
    selections: list[CascadeSelection] = []
    failed: list[dict[str, Any]] = []
    for task_id in sorted(groups):
        try:
            selections.append(
                select_cascade_group(
                    groups[task_id],
                    chosen_trajectory_ids=chosen_trajectory_ids,
                )
            )
        except ValueError as exc:
            failed.append(
                {
                    "task_id": task_id,
                    "reason": str(exc),
                    "teacher_ids": sorted(
                        {item.provenance.teacher_id for item in groups[task_id]}
                    ),
                    "candidate_count": len(groups[task_id]),
                }
            )
    return selections, failed


def audit_candidate_trainability(
    candidates: list[CascadeTeacherCandidate],
) -> tuple[set[str], Counter[str]]:
    """Apply the stricter SFT replay/grounding contract before arbitration."""

    eligible = set()
    rejection_codes: Counter[str] = Counter()
    builder = SFTDatasetBuilder()
    for item in candidates:
        goal = _candidate_goal_payload(item)
        constraints = goal.get("hard_constraints") or {}
        if not isinstance(constraints, dict):
            constraints = constraints.model_dump(mode="json")
        result = builder.build(
            [
                EpisodeCandidate(
                    scenario_id=item.candidate.task_id,
                    source="teacher",
                    template_family=item.candidate.family,
                    city=str(constraints.get("destination") or "unknown"),
                    episode=item.candidate.rollout.episode,
                )
            ]
        )
        if result.examples:
            eligible.add(item.candidate.score.trajectory_id)
        else:
            for review in result.reviews:
                rejection_codes.update(review.rejection_codes)
    return eligible, rejection_codes


def build_sft(selections: list[CascadeSelection]) -> tuple[DatasetBuildResult, dict[str, int]]:
    raw = SFTDatasetBuilder().build(
        [
            EpisodeCandidate(
                scenario_id=selection.task_id,
                source="teacher",
                template_family=selection.family,
                city=_destination(selection),
                episode=selection.chosen.candidate.rollout.episode,
            )
            for selection in selections
        ]
    )
    groups: dict[str, list[Any]] = defaultdict(list)
    for example in raw.examples:
        prompt_hash = _hash(
            {
                "messages": [
                    message.model_dump(mode="json", exclude_none=True)
                    for message in example.messages[:-1]
                ],
                "tools": example.tools,
            }
        )
        groups[prompt_hash].append(example)

    selected = []
    duplicate_rows = 0
    conflict_groups = 0
    conflict_rows = 0
    for examples in groups.values():
        responses = {
            _hash(item.messages[-1].model_dump(mode="json", exclude_none=True))
            for item in examples
        }
        if len(responses) > 1:
            conflict_groups += 1
            conflict_rows += len(examples)
            continue
        selected.append(min(examples, key=lambda item: item.example_id))
        duplicate_rows += len(examples) - 1
    selected.sort(key=lambda item: item.example_id)

    selected_ids = {item.example_id for item in selected}
    reviews = []
    for review in raw.reviews:
        count = sum(
            item.example_id in selected_ids and item.trajectory_id == review.trajectory_id
            for item in raw.examples
        )
        reviews.append(review.model_copy(update={"example_count": count}))
    manifest = raw.manifest.model_copy(
        update={
            "dataset_version": "sft-cascade-"
            + _hash([item.example_id for item in selected])[:16],
            "exported_examples": len(selected),
            "split_examples": dict(Counter(item.split for item in selected)),
        }
    )
    return (
        DatasetBuildResult(manifest=manifest, reviews=reviews, examples=selected),
        {
            "duplicates_dropped": duplicate_rows,
            "label_conflict_groups_quarantined": conflict_groups,
            "label_conflict_rows_quarantined": conflict_rows,
        },
    )


def build_preferences(
    selections: list[CascadeSelection],
) -> dict[str, list[dict[str, Any]]]:
    unique: dict[str, dict[str, Any]] = {}
    for selection in selections:
        for pair in selection.preference_pairs:
            unique.setdefault(pair.pair_id, pair.model_dump(mode="json"))
    split_context: dict[str, str] = {}
    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for pair in sorted(unique.values(), key=lambda item: item["pair_id"]):
        context_hash = str(pair["context_hash"])
        split = split_context.setdefault(context_hash, _stable_split(context_hash))
        splits[split].append(pair)
    return splits


def audit_forbidden_prompts(
    selections: list[CascadeSelection],
    forbidden_paths: list[Path],
    *,
    similarity_threshold: float,
) -> tuple[dict[str, Any], set[str]]:
    prompts = [
        (selection.task_id, _original_request(selection)) for selection in selections
    ]
    forbidden = []
    for path in forbidden_paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                forbidden.extend(_prompt_like_strings(json.loads(line)))
    normalized_forbidden = [normalize_text(item) for item in forbidden if item.strip()]
    forbidden_ngrams = [character_ngrams(item) for item in normalized_forbidden]
    exact = set(normalized_forbidden)
    exact_matches = 0
    near_matches = 0
    max_similarity = 0.0
    finding_hashes = []
    contaminated_task_ids = set()
    quarantined_family_counts: Counter[str] = Counter()
    family_by_task = {item.task_id: item.family for item in selections}
    for task_id, prompt in prompts:
        normalized = normalize_text(prompt)
        grams = character_ngrams(normalized)
        best = max(
            (jaccard_similarity(grams, other) for other in forbidden_ngrams),
            default=0.0,
        )
        max_similarity = max(max_similarity, best)
        finding_type = None
        if normalized in exact:
            exact_matches += 1
            finding_type = "exact"
        elif best >= similarity_threshold:
            near_matches += 1
            finding_type = "near_duplicate"
        if finding_type:
            contaminated_task_ids.add(task_id)
            quarantined_family_counts[family_by_task[task_id]] += 1
            finding_hashes.append(
                _hash({"prompt": normalized, "type": finding_type, "score": best})
            )
    audit = {
        "passed": bool(forbidden_paths) and bool(forbidden),
        "registered_files": len(forbidden_paths),
        "forbidden_prompt_fields": len(forbidden),
        "selected_prompts": len(prompts),
        "similarity_threshold": similarity_threshold,
        "source_exact_matches": exact_matches,
        "source_near_duplicate_matches": near_matches,
        "quarantined_tasks": len(contaminated_task_ids),
        "quarantined_family_counts": dict(quarantined_family_counts),
        "retained_exact_matches": 0,
        "retained_near_duplicate_matches": 0,
        "max_similarity": round(max_similarity, 8),
        "quarantined_finding_hashes": sorted(set(finding_hashes)),
    }
    return audit, contaminated_task_ids


def build(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_config(config_path)
    candidates = load_candidates(config, config_dir=config_path.parent)
    trainable_trajectory_ids, candidate_rejections = audit_candidate_trainability(
        candidates
    )
    selections, failed = select_groups(
        candidates,
        chosen_trajectory_ids=trainable_trajectory_ids,
    )
    forbidden_paths = [
        _resolve(config_path.parent, item)
        for item in config.get("forbidden_corpora", [])
    ]
    contamination, contaminated_task_ids = audit_forbidden_prompts(
        selections,
        forbidden_paths,
        similarity_threshold=float(config.get("similarity_threshold", 0.82)),
    )
    selections = [
        item for item in selections if item.task_id not in contaminated_task_ids
    ]
    sft, sft_audit = build_sft(selections)
    preferences = build_preferences(selections)
    contributions = contribution_summary(candidates, selections)
    chosen_counts = Counter(item.chosen.provenance.teacher_id for item in selections)
    family_counts = Counter(item.family for item in selections)
    verifier_success_tasks = {
        item.candidate.task_id
        for item in candidates
        if item.candidate.score.successful
        and item.candidate.task_id not in contaminated_task_ids
    }
    verifier_success_family_counts = Counter(
        next(
            item.candidate.family
            for item in candidates
            if item.candidate.task_id == task_id
        )
        for task_id in verifier_success_tasks
    )
    selection_by_trajectory = {
        item.chosen.candidate.score.trajectory_id: item for item in selections
    }
    sft_trajectories = {item.trajectory_id for item in sft.examples}
    sft_teacher_counts = Counter(
        selection_by_trajectory[trajectory_id].chosen.provenance.teacher_id
        for trajectory_id in sft_trajectories
    )
    sft_family_counts = Counter(
        selection_by_trajectory[trajectory_id].family
        for trajectory_id in sft_trajectories
    )
    student_teacher_ids = {
        item.provenance.teacher_id
        for item in candidates
        if item.provenance.tier == "student_teacher"
    }
    student_chosen = sum(chosen_counts[item] for item in student_teacher_ids)
    student_share = student_chosen / len(selections) if selections else 0.0
    sft_student_chosen = sum(sft_teacher_counts[item] for item in student_teacher_ids)
    sft_student_share = (
        sft_student_chosen / len(sft_trajectories) if sft_trajectories else 0.0
    )
    thresholds = config.get("pilot_thresholds", {})
    minimum_selected = int(thresholds.get("minimum_selected_tasks", 1))
    minimum_verified_success = int(
        thresholds.get("minimum_verified_success_tasks", minimum_selected)
    )
    minimum_per_family = int(thresholds.get("minimum_selected_per_family", 1))
    minimum_sft_examples = int(thresholds.get("minimum_sft_examples", 1))
    minimum_sft_per_family = int(thresholds.get("minimum_sft_per_family", 1))
    minimum_student_share = float(thresholds.get("minimum_student_teacher_chosen_share", 0.0))
    required_families = list(config.get("required_families", []))

    errors = []
    if len(verifier_success_tasks) < minimum_verified_success:
        errors.append(
            "VERIFIER_SUCCESS_TASKS_TOO_SMALL:"
            f"{len(verifier_success_tasks)}<{minimum_verified_success}"
        )
    if len(selections) < minimum_selected:
        errors.append(f"SELECTED_TASKS_TOO_SMALL:{len(selections)}<{minimum_selected}")
    for family in required_families:
        if family_counts[family] < minimum_per_family:
            errors.append(
                f"FAMILY_TOO_SMALL:{family}:{family_counts[family]}<{minimum_per_family}"
            )
        if sft_family_counts[family] < minimum_sft_per_family:
            errors.append(
                f"SFT_FAMILY_TOO_SMALL:{family}:"
                f"{sft_family_counts[family]}<{minimum_sft_per_family}"
            )
    if len(sft.examples) < minimum_sft_examples:
        errors.append(f"SFT_TOO_SMALL:{len(sft.examples)}<{minimum_sft_examples}")
    if sft_student_share < minimum_student_share:
        errors.append(
            "SFT_STUDENT_TEACHER_SHARE_TOO_LOW:"
            f"{sft_student_share:.6f}<{minimum_student_share:.6f}"
        )
    if sft_audit["label_conflict_groups_quarantined"]:
        errors.append("SFT_LABEL_CONFLICTS_QUARANTINED")
    if not contamination["passed"]:
        errors.append("FORBIDDEN_EVALUATION_CONTAMINATION_GATE_FAILED")

    output_dir.mkdir(parents=True, exist_ok=True)
    SFTDatasetBuilder().export(sft, output_dir / "sft")
    for split, rows in preferences.items():
        _write_jsonl(output_dir / "preferences" / f"{split}.jsonl", rows)
    preference_pair_count = sum(len(rows) for rows in preferences.values())
    preference_status = (
        "rejected"
        if errors
        else "empty_no_verified_failure_pairs"
        if preference_pair_count == 0
        else "passed"
    )
    preference_manifest = {
        "schema_version": "cascade-preference-dataset.v1",
        "status": preference_status,
        "dataset_version": "preference-cascade-"
        + _hash(
            {split: [item["pair_id"] for item in rows] for split, rows in preferences.items()}
        )[:16],
        "requires_verifier_success_over_failure": True,
        "split_counts": {split: len(rows) for split, rows in preferences.items()},
        "unique_pairs": preference_pair_count,
        "context_split_overlap": 0,
        "errors": errors,
    }
    (output_dir / "preferences" / "manifest.json").write_text(
        json.dumps(preference_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(output_dir / "failed_groups.jsonl", failed)
    _write_jsonl(
        output_dir / "selections.jsonl",
        [_selection_summary(item) for item in selections],
    )
    source_hashes = {
        item["run_id"]: _sha256(
            _resolve(config_path.parent, item["candidates_file"])
        )
        for item in config["teacher_runs"]
    }
    manifest = {
        "schema_version": "stage32-cascade-distillation-build.v1",
        "status": "passed" if not errors else "rejected",
        "dataset_version": "stage32-cascade-"
        + _hash(
            {
                "selected": [item.task_id for item in selections],
                "sources": source_hashes,
            }
        )[:16],
        "config_sha256": _sha256(config_path),
        "source_candidate_sha256": source_hashes,
        "candidate_rollouts": len(candidates),
        "candidate_tasks": len({item.candidate.task_id for item in candidates}),
        "verifier_success_tasks": len(verifier_success_tasks),
        "verifier_success_family_counts": dict(verifier_success_family_counts),
        "trainable_candidate_rollouts": len(trainable_trajectory_ids),
        "candidate_trainability_rejection_codes": dict(candidate_rejections),
        "selected_tasks": len(selections),
        "failed_groups": len(failed),
        "difficulty_counts": dict(Counter(item.difficulty for item in selections)),
        "family_counts": dict(family_counts),
        "teacher_action_agreement": sum(
            item.teacher_action_agreement for item in selections
        ),
        "teacher_contributions": [
            item.model_dump(mode="json") for item in contributions
        ],
        "chosen_teacher_counts": dict(chosen_counts),
        "student_teacher_chosen_share": round(student_share, 8),
        "sft_chosen_teacher_counts": dict(sft_teacher_counts),
        "sft_student_teacher_chosen_share": round(sft_student_share, 8),
        "sft": {
            "dataset_version": sft.manifest.dataset_version,
            "examples": len(sft.examples),
            "split_counts": sft.manifest.split_examples,
            "family_counts": dict(sft_family_counts),
            "accepted_trajectories": len(sft_trajectories),
            "source_rejected_episodes": sft.manifest.rejected_episodes,
            **sft_audit,
        },
        "preferences": preference_manifest,
        "forbidden_evaluation_contamination": contamination,
        "errors": errors,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _selection_summary(selection: CascadeSelection) -> dict[str, Any]:
    return {
        "schema_version": selection.schema_version,
        "task_id": selection.task_id,
        "family": selection.family,
        "difficulty": selection.difficulty,
        "arbitration_reason": selection.arbitration_reason,
        "chosen_teacher_id": selection.chosen.provenance.teacher_id,
        "chosen_model": selection.chosen.provenance.model,
        "chosen_checkpoint": selection.chosen.provenance.checkpoint,
        "chosen_trajectory_id": selection.chosen.candidate.score.trajectory_id,
        "chosen_score": selection.chosen.candidate.score.model_dump(mode="json"),
        "rejected": [
            {
                "teacher_id": item.provenance.teacher_id,
                "trajectory_id": item.candidate.score.trajectory_id,
                "score": item.candidate.score.model_dump(mode="json"),
            }
            for item in selection.rejected
        ],
        "preference_pair_ids": [item.pair_id for item in selection.preference_pairs],
        "successful_teacher_ids": selection.successful_teacher_ids,
        "teacher_action_agreement": selection.teacher_action_agreement,
    }


def _goal_payload(selection: CascadeSelection) -> dict[str, Any]:
    return _candidate_goal_payload(selection.chosen)


def _candidate_goal_payload(item: CascadeTeacherCandidate) -> dict[str, Any]:
    initial_state = item.candidate.rollout.episode.initial_state
    if not isinstance(initial_state, dict):
        initial_state = initial_state.model_dump(mode="json")
    goal = initial_state.get("goal") or {}
    if not isinstance(goal, dict):
        goal = goal.model_dump(mode="json")
    return goal


def _original_request(selection: CascadeSelection) -> str:
    return str(_goal_payload(selection).get("original_request") or "")


def _destination(selection: CascadeSelection) -> str:
    constraints = _goal_payload(selection).get("hard_constraints") or {}
    if not isinstance(constraints, dict):
        constraints = constraints.model_dump(mode="json")
    return str(constraints.get("destination") or "unknown")


def _prompt_like_strings(value: Any, *, key: str = "") -> list[str]:
    found = []
    if isinstance(value, dict):
        if value.get("role") == "user" and isinstance(value.get("content"), str):
            content = value["content"].strip()
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError:
                decoded = None
            nested = _prompt_like_strings(decoded) if decoded is not None else []
            found.extend(nested or ([content] if len(content) >= 8 else []))
        for child_key, child in value.items():
            if child_key == "content" and value.get("role") == "user":
                continue
            if isinstance(child, str) and child_key in {
                "natural_request",
                "original_request",
                "prompt",
                "request",
                "user_request",
            } and len(child.strip()) >= 8:
                found.append(child)
            else:
                found.extend(_prompt_like_strings(child, key=child_key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_prompt_like_strings(child, key=key))
    return found


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def character_ngrams(text: str, *, size: int = 5) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) <= size:
        return {normalized}
    return {
        normalized[index : index + size]
        for index in range(len(normalized) - size + 1)
    }


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _resolve(parent: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (parent / path).resolve()


def _stable_split(value: str) -> str:
    bucket = int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.config, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
