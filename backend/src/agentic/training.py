"""Dependency-light preflight helpers for real Agent Policy training jobs."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pydantic import BaseModel, Field

from agentic.sft_dataset import DatasetManifest, SFTExample


class TrainingDependency(BaseModel):
    name: str
    installed: bool
    version: str | None = None
    incompatible_requirements: list[str] = Field(default_factory=list)


class SFTPreflightReport(BaseModel):
    ready: bool
    dataset_version: str
    train_examples: int
    validation_examples: int
    test_examples: int
    unique_model_visible_payloads: int
    dependencies: list[TrainingDependency]
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SFTModelPreflightReport(BaseModel):
    ready: bool
    rows_checked: int
    max_sequence_tokens: int
    p95_sequence_tokens: int
    over_max_length: int
    tool_envelope_rows: int
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SFTTerminationBoundaryPreflightReport(BaseModel):
    ready: bool
    rows_checked: int
    boundary_rows: int
    termination_token_id: int | None
    missing_termination_rows: int
    multiple_termination_rows: int
    malformed_boundary_rows: int
    errors: list[str] = Field(default_factory=list)


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
) -> list[dict[str, Any]]:
    """Convert audited examples to TRL completion-only conversational rows."""
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        example = SFTExample(**row)
        # Empty argument objects are meaningful for zero-argument tools.  Using
        # exclude_defaults=True removes ``arguments: {}``, which makes Qwen3's
        # native tool-call template receive an Undefined value and fail JSON
        # serialization during SFT tokenization.
        messages = [item.model_dump(exclude_none=True) for item in example.messages]
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            raise ValueError(f"row {index} must end with one assistant decision")
        prompt = messages[:-1]
        completion = [messages[-1]]
        if not any(item["role"] == "user" for item in prompt):
            raise ValueError(f"row {index} has no user policy context")
        if not example.tools:
            raise ValueError(f"row {index} has no policy action schemas")
        converted.append({"prompt": prompt, "completion": completion, "tools": example.tools})
    return converted


def select_sft_smoke_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select a deterministic action-stratified subset for bounded smoke runs."""
    if limit <= 0 or limit >= len(rows):
        return rows
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        example = SFTExample(**row)
        final = example.messages[-1]
        action = final.tool_calls[0].function.name if final.tool_calls else "no_tool_call"
        groups[action].append(row)
    for action in groups:
        groups[action].sort(key=lambda row: str(row.get("example_id") or ""))
    selected: list[dict[str, Any]] = []
    actions = sorted(groups)
    cursor = 0
    while len(selected) < limit and actions:
        next_actions: list[str] = []
        for action in actions:
            bucket = groups[action]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                next_actions.append(action)
                if len(selected) >= limit:
                    break
        actions = next_actions
        cursor += 1
    return selected


def _chat_template_token_ids(encoded: Any) -> list[int]:
    """Normalize Transformers chat-template results without importing Transformers."""
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return list(encoded)


def preflight_sft_model(
    dataset_dir: Path,
    tokenizer: Any,
    *,
    max_length: int = 2048,
) -> SFTModelPreflightReport:
    """Verify the model's native template can losslessly encode every SFT row."""
    errors: list[str] = []
    warnings: list[str] = []
    lengths: list[int] = []
    tool_envelope_rows = 0
    prefix_mismatches = 0
    empty_completions = 0
    template_failures = 0

    if not getattr(tokenizer, "chat_template", None):
        return SFTModelPreflightReport(
            ready=False,
            rows_checked=0,
            max_sequence_tokens=0,
            p95_sequence_tokens=0,
            over_max_length=0,
            tool_envelope_rows=0,
            errors=["MODEL_CHAT_TEMPLATE_MISSING"],
        )

    for split in ("train", "validation", "test"):
        rows = to_conversational_prompt_completion(load_jsonl(dataset_dir / f"{split}.jsonl"))
        for row in rows:
            prompt = row["prompt"]
            messages = [*prompt, *row["completion"]]
            template_kwargs = {
                "tools": row["tools"],
                "enable_thinking": False,
            }
            try:
                prompt_ids = _chat_template_token_ids(
                    tokenizer.apply_chat_template(
                        prompt,
                        tokenize=True,
                        add_generation_prompt=True,
                        **template_kwargs,
                    )
                )
                full_ids = _chat_template_token_ids(
                    tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=False,
                        **template_kwargs,
                    )
                )
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    **template_kwargs,
                )
            except Exception:  # pragma: no cover - exact exception is tokenizer-specific
                template_failures += 1
                continue
            lengths.append(len(full_ids))
            if full_ids[: len(prompt_ids)] != prompt_ids:
                prefix_mismatches += 1
            if len(full_ids) <= len(prompt_ids):
                empty_completions += 1
            if "<tool_call>" in rendered and "</tool_call>" in rendered:
                tool_envelope_rows += 1

    rows_checked = len(lengths)
    if template_failures:
        errors.append(f"MODEL_CHAT_TEMPLATE_FAILURES:{template_failures}")
    if prefix_mismatches:
        errors.append(f"MODEL_COMPLETION_PREFIX_MISMATCHES:{prefix_mismatches}")
    if empty_completions:
        errors.append(f"MODEL_EMPTY_COMPLETIONS:{empty_completions}")
    if rows_checked and tool_envelope_rows != rows_checked:
        errors.append(f"MODEL_TOOL_ENVELOPE_MISSING:{rows_checked - tool_envelope_rows}")
    over_max_length = sum(length > max_length for length in lengths)
    if over_max_length:
        errors.append(f"MODEL_SEQUENCES_OVER_MAX_LENGTH:{over_max_length}")
    if not lengths:
        errors.append("MODEL_NO_ROWS_ENCODED")
        max_tokens = 0
        p95_tokens = 0
    else:
        ordered = sorted(lengths)
        max_tokens = ordered[-1]
        p95_tokens = ordered[int((len(ordered) - 1) * 0.95)]
        if max_tokens > int(max_length * 0.9):
            warnings.append(f"MODEL_SEQUENCE_LENGTH_HEADROOM_LOW:{max_tokens}/{max_length}")
    return SFTModelPreflightReport(
        ready=not errors,
        rows_checked=rows_checked,
        max_sequence_tokens=max_tokens,
        p95_sequence_tokens=p95_tokens,
        over_max_length=over_max_length,
        tool_envelope_rows=tool_envelope_rows,
        errors=errors,
        warnings=warnings,
    )


def preflight_sft_termination_boundaries(
    dataset_dir: Path,
    tokenizer: Any,
) -> SFTTerminationBoundaryPreflightReport:
    """Verify every supervised tool call teaches one immediate termination token."""
    termination_token_id = getattr(tokenizer, "eos_token_id", None)
    errors: list[str] = []
    if termination_token_id is None:
        return SFTTerminationBoundaryPreflightReport(
            ready=False,
            rows_checked=0,
            boundary_rows=0,
            termination_token_id=None,
            missing_termination_rows=0,
            multiple_termination_rows=0,
            malformed_boundary_rows=0,
            errors=["MODEL_TERMINATION_TOKEN_MISSING"],
        )

    rows_checked = 0
    boundary_rows = 0
    missing = 0
    multiple = 0
    malformed = 0
    for split in ("train", "validation", "test"):
        rows = to_conversational_prompt_completion(load_jsonl(dataset_dir / f"{split}.jsonl"))
        for row in rows:
            rows_checked += 1
            kwargs = {"tools": row["tools"], "enable_thinking": False}
            try:
                prompt_ids = _chat_template_token_ids(
                    tokenizer.apply_chat_template(
                        row["prompt"],
                        tokenize=True,
                        add_generation_prompt=True,
                        **kwargs,
                    )
                )
                full_ids = _chat_template_token_ids(
                    tokenizer.apply_chat_template(
                        [*row["prompt"], *row["completion"]],
                        tokenize=True,
                        add_generation_prompt=False,
                        **kwargs,
                    )
                )
            except Exception:  # pragma: no cover - tokenizer-specific exception
                malformed += 1
                continue
            if full_ids[: len(prompt_ids)] != prompt_ids:
                malformed += 1
                continue
            completion_ids = full_ids[len(prompt_ids) :]
            positions = [
                index
                for index, token_id in enumerate(completion_ids)
                if token_id == termination_token_id
            ]
            if not positions:
                missing += 1
                continue
            if len(positions) > 1:
                multiple += 1
                continue
            boundary_index = positions[0]
            try:
                before = tokenizer.decode(
                    completion_ids[:boundary_index], skip_special_tokens=False
                ).rstrip()
                after = tokenizer.decode(
                    completion_ids[boundary_index + 1 :], skip_special_tokens=False
                )
            except Exception:  # pragma: no cover - tokenizer-specific exception
                malformed += 1
                continue
            if not before.endswith("</tool_call>") or after.strip():
                malformed += 1
                continue
            boundary_rows += 1

    if missing:
        errors.append(f"SFT_TERMINATION_TOKEN_MISSING:{missing}")
    if multiple:
        errors.append(f"SFT_MULTIPLE_TERMINATION_TOKENS:{multiple}")
    if malformed:
        errors.append(f"SFT_MALFORMED_TERMINATION_BOUNDARY:{malformed}")
    if boundary_rows != rows_checked:
        errors.append(f"SFT_BOUNDARY_COVERAGE:{boundary_rows}/{rows_checked}")
    return SFTTerminationBoundaryPreflightReport(
        ready=not errors,
        rows_checked=rows_checked,
        boundary_rows=boundary_rows,
        termination_token_id=termination_token_id,
        missing_termination_rows=missing,
        multiple_termination_rows=multiple,
        malformed_boundary_rows=malformed,
        errors=errors,
    )


def check_training_dependencies(*, extra_names: tuple[str, ...] = ()) -> list[TrainingDependency]:
    names = (
        "torch",
        "transformers",
        "datasets",
        "trl",
        "peft",
        "bitsandbytes",
        "accelerate",
        *extra_names,
    )
    dependencies: list[TrainingDependency] = []
    installed_versions: dict[str, str] = {}
    for name in names:
        installed = importlib.util.find_spec(name) is not None
        version = None
        if installed:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        if version is not None:
            installed_versions[canonicalize_name(name)] = version
        dependencies.append(TrainingDependency(name=name, installed=installed, version=version))
    for dependency in dependencies:
        if not dependency.installed:
            continue
        try:
            declared = importlib.metadata.requires(dependency.name) or []
        except importlib.metadata.PackageNotFoundError:
            continue
        for raw_requirement in declared:
            try:
                requirement = Requirement(raw_requirement)
            except ValueError:
                continue
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            installed_version = installed_versions.get(canonicalize_name(requirement.name))
            if (
                installed_version is not None
                and requirement.specifier
                and not requirement.specifier.contains(installed_version, prereleases=True)
            ):
                dependency.incompatible_requirements.append(
                    f"{requirement.name}{requirement.specifier} (installed {installed_version})"
                )
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
    model_visible_payloads: set[str] = set()
    duplicate_payloads = 0
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
            payload = json.dumps(
                {
                    "messages": [message.model_dump(mode="json") for message in example.messages],
                    "tools": example.tools,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            digest = hashlib.sha256(payload.encode()).hexdigest()
            if digest in model_visible_payloads:
                duplicate_payloads += 1
            else:
                model_visible_payloads.add(digest)
    if duplicate_payloads:
        errors.append(f"MODEL_VISIBLE_PAYLOAD_DUPLICATES:{duplicate_payloads}")
    dependencies = check_training_dependencies()
    missing = [item.name for item in dependencies if not item.installed]
    if require_dependencies and missing:
        errors.append("TRAINING_DEPENDENCIES_MISSING:" + ",".join(missing))
    conflicts = [
        f"{item.name}->{requirement}"
        for item in dependencies
        for requirement in item.incompatible_requirements
    ]
    if require_dependencies and conflicts:
        errors.append("TRAINING_DEPENDENCIES_INCOMPATIBLE:" + ",".join(conflicts))
    return SFTPreflightReport(
        ready=not errors,
        dataset_version=manifest.dataset_version,
        train_examples=counts["train"],
        validation_examples=counts["validation"],
        test_examples=counts["test"],
        unique_model_visible_payloads=len(model_visible_payloads),
        dependencies=dependencies,
        errors=errors,
        warnings=warnings,
    )
