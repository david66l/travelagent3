"""Dependency-light corpus gates for stateful Agentic GRPO training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic.environment import (
    EnvironmentSnapshot,
    EnvironmentTask,
    environment_fingerprint,
)
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT
from agentic.training import TrainingDependency, check_training_dependencies, load_jsonl
from agentic.trajectory import redact_pii


class GRPOCorpusRow(BaseModel):
    task: EnvironmentTask
    snapshot: EnvironmentSnapshot


class GRPOPreflightReport(BaseModel):
    ready: bool
    train_tasks: int
    validation_tasks: int
    environment_versions: list[str]
    snapshot_versions: list[str]
    dependencies: list[TrainingDependency]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


_FULL_PLAN_TOOLS = {
    "get_weather",
    "search_pois",
    "get_poi_detail",
    "get_route_matrix",
    "solve_itinerary",
    "validate_itinerary",
}


def load_grpo_corpus(path: Path) -> list[GRPOCorpusRow]:
    return [GRPOCorpusRow(**row) for row in load_jsonl(path)]


def to_trl_environment_rows(rows: list[GRPOCorpusRow]) -> list[dict[str, Any]]:
    """Build conversational rows whose reset hook supplies the first policy state."""
    return [
        {
            "prompt": [
                {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
                {"role": "user", "content": ""},
            ],
            "task": row.task.model_dump(mode="json"),
            "snapshot": row.snapshot.model_dump(mode="json"),
            "task_id": row.task.task_id,
            "difficulty": row.task.difficulty,
            "initial_state_fingerprint": environment_fingerprint(row.task, row.snapshot),
        }
        for row in rows
    ]


def preflight_grpo_corpus(
    corpus_dir: Path,
    *,
    minimum_train_tasks: int = 1000,
    require_dependencies: bool = True,
) -> GRPOPreflightReport:
    train = load_grpo_corpus(corpus_dir / "train.jsonl")
    validation = load_grpo_corpus(corpus_dir / "validation.jsonl")
    errors: list[str] = []
    warnings: list[str] = []
    if len(train) < minimum_train_tasks:
        errors.append(f"TRAIN_TASKS_BELOW_MINIMUM:{len(train)}<{minimum_train_tasks}")
    if not validation:
        errors.append("VALIDATION_TASKS_EMPTY")

    train_ids = [row.task.task_id for row in train]
    validation_ids = [row.task.task_id for row in validation]
    if len(train_ids) != len(set(train_ids)):
        errors.append("DUPLICATE_TRAIN_TASK_ID")
    if len(validation_ids) != len(set(validation_ids)):
        errors.append("DUPLICATE_VALIDATION_TASK_ID")
    if set(train_ids) & set(validation_ids):
        errors.append("TASK_ID_SPLIT_OVERLAP")

    train_fingerprints = {environment_fingerprint(row.task, row.snapshot) for row in train}
    validation_fingerprints = {
        environment_fingerprint(row.task, row.snapshot) for row in validation
    }
    if train_fingerprints & validation_fingerprints:
        errors.append("INITIAL_STATE_SPLIT_OVERLAP")

    for split, rows in (("train", train), ("validation", validation)):
        for row in rows:
            prefix = f"{split}:{row.task.task_id}"
            payload = row.model_dump(mode="json")
            if redact_pii(payload) != payload:
                errors.append(f"PII_DETECTED:{prefix}")
            if row.task.missing_slots:
                continue
            if row.task.feasibility_report.get("feasible", True) is False:
                continue
            available = set(row.snapshot.tool_responses)
            missing = sorted(_FULL_PLAN_TOOLS - available)
            if missing:
                errors.append(f"SNAPSHOT_TOOLS_MISSING:{prefix}:{','.join(missing)}")
            if any(not row.snapshot.tool_responses.get(name) for name in _FULL_PLAN_TOOLS):
                errors.append(f"SNAPSHOT_RESPONSES_EMPTY:{prefix}")
            if not row.snapshot.hidden_test_facts:
                warnings.append(f"HIDDEN_TEST_FACTS_EMPTY:{prefix}")

    dependencies = check_training_dependencies(extra_names=("jmespath",))
    missing_dependencies = [item.name for item in dependencies if not item.installed]
    if require_dependencies and missing_dependencies:
        errors.append("TRAINING_DEPENDENCIES_MISSING:" + ",".join(missing_dependencies))

    all_rows = [*train, *validation]
    return GRPOPreflightReport(
        ready=not errors,
        train_tasks=len(train),
        validation_tasks=len(validation),
        environment_versions=sorted({row.snapshot.environment_version for row in all_rows}),
        snapshot_versions=sorted({row.snapshot.snapshot_version for row in all_rows}),
        dependencies=dependencies,
        errors=sorted(set(errors)),
        warnings=sorted(set(warnings)),
    )


__all__ = [
    "GRPOCorpusRow",
    "GRPOPreflightReport",
    "load_grpo_corpus",
    "preflight_grpo_corpus",
    "to_trl_environment_rows",
]
