"""Dependency-light preflight helpers for real Agent Policy training jobs."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from agentic.sft_dataset import DatasetManifest, SFTExample


class TrainingDependency(BaseModel):
    name: str
    installed: bool
    version: str | None = None


class SFTPreflightReport(BaseModel):
    ready: bool
    dataset_version: str
    train_examples: int
    validation_examples: int
    test_examples: int
    dependencies: list[TrainingDependency]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def to_conversational_prompt_completion(
    rows: list[dict[str, Any]],
) -> list[dict[str, list[dict[str, str]]]]:
    """Convert audited examples to TRL completion-only conversational rows."""
    converted: list[dict[str, list[dict[str, str]]]] = []
    for index, row in enumerate(rows):
        example = SFTExample(**row)
        messages = [item.model_dump() for item in example.messages]
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            raise ValueError(f"row {index} must end with one assistant decision")
        prompt = messages[:-1]
        completion = [messages[-1]]
        if not any(item["role"] == "user" for item in prompt):
            raise ValueError(f"row {index} has no user policy context")
        converted.append({"prompt": prompt, "completion": completion})
    return converted


def check_training_dependencies() -> list[TrainingDependency]:
    names = (
        "torch",
        "transformers",
        "datasets",
        "trl",
        "peft",
        "bitsandbytes",
        "accelerate",
    )
    dependencies: list[TrainingDependency] = []
    for name in names:
        installed = importlib.util.find_spec(name) is not None
        version = None
        if installed:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        dependencies.append(TrainingDependency(name=name, installed=installed, version=version))
    return dependencies


def preflight_sft_dataset(
    dataset_dir: Path,
    *,
    minimum_train_examples: int = 3000,
    require_dependencies: bool = True,
) -> SFTPreflightReport:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"dataset manifest missing: {manifest_path}")
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    split_rows = {
        split: load_jsonl(dataset_dir / f"{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    errors: list[str] = []
    warnings: list[str] = []
    counts = {split: len(rows) for split, rows in split_rows.items()}
    if manifest.split_group_overlap:
        errors.append("DATASET_SPLIT_GROUP_OVERLAP")
    if manifest.rejected_episodes:
        warnings.append("DATASET_CONTAINS_REJECTED_CANDIDATES_IN_REVIEW_REPORT")
    if counts["train"] < minimum_train_examples:
        errors.append(f"TRAIN_EXAMPLES_BELOW_MINIMUM:{counts['train']}<{minimum_train_examples}")
    if not counts["validation"]:
        errors.append("VALIDATION_SPLIT_EMPTY")
    if not counts["test"]:
        errors.append("TEST_SPLIT_EMPTY")
    for split, rows in split_rows.items():
        for row in rows:
            example = SFTExample(**row)
            if example.split != split:
                errors.append(f"SPLIT_LABEL_MISMATCH:{split}:{example.example_id}")
    dependencies = check_training_dependencies()
    missing = [item.name for item in dependencies if not item.installed]
    if require_dependencies and missing:
        errors.append("TRAINING_DEPENDENCIES_MISSING:" + ",".join(missing))
    return SFTPreflightReport(
        ready=not errors,
        dataset_version=manifest.dataset_version,
        train_examples=counts["train"],
        validation_examples=counts["validation"],
        test_examples=counts["test"],
        dependencies=dependencies,
        errors=errors,
        warnings=warnings,
    )
