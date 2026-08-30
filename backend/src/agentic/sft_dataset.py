"""Audit, split and export real Agent Loop episodes for policy SFT."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic.loop import NO_TOOL_ACTIONS
from agentic.policy import AGENT_TOOL_POLICY_SYSTEM_PROMPT, policy_prompt_payload
from agentic.policy_actions import policy_action_schemas, validate_policy_arguments
from agentic.trajectory import AgentEpisode, EpisodeReplayVerifier
from tools.tool_definitions import TOOL_NAME_TO_MODEL


SFT_DATASET_SCHEMA_VERSION = "agent-policy-sft.v2"
_PHONE = re.compile(r"(?<![A-Za-z0-9])(?:\+?86[- ]?)?1[3-9]\d{9}(?![A-Za-z0-9])")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ID_CARD = re.compile(r"(?<![A-Za-z0-9])\d{17}[\dXx](?![A-Za-z0-9])")
_PROTECTED_ARGUMENTS = {
    "constraints",
    "facts",
    "itinerary",
    "pois",
    "dist_matrix",
    "tc_matrix",
    "amap_minutes",
}
_GENERATIVE_ARGUMENTS = {
    "abort": {"reason"},
    "ask_user": {"question"},
    "propose_tradeoff": {"options", "reason"},
    "retrieve_city_knowledge": {"topic"},
    "retry_solve": {"reason"},
    "search_current_info": {"query"},
    "search_pois": {"keywords"},
}
_SCHEMA_ENUM_ARGUMENTS = {
    "info_type",
    "mode",
    "strategy",
}

EpisodeSource = Literal["teacher", "shadow", "production", "synthetic"]
DatasetSplit = Literal["train", "validation", "test"]
QualityLabel = Literal["validated_plan", "clarification", "safe_termination"]


class EpisodeCandidate(BaseModel):
    scenario_id: str = Field(min_length=1)
    source: EpisodeSource
    template_family: str = Field(min_length=1)
    city: str = Field(min_length=1)
    episode: AgentEpisode
    user_partition_key: str | None = None
    training_authorized: bool = True
    contains_production_data: bool = False


class SFTToolFunction(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class SFTToolCall(BaseModel):
    type: Literal["function"] = "function"
    function: SFTToolFunction


class SFTMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[SFTToolCall] = Field(default_factory=list)


class SFTExample(BaseModel):
    schema_version: str = SFT_DATASET_SCHEMA_VERSION
    example_id: str
    scenario_id: str
    trajectory_id: str
    step_index: int = Field(ge=0)
    split: DatasetSplit
    quality_label: QualityLabel
    source: EpisodeSource
    environment_version: str
    policy_name: str
    policy_version: str
    messages: list[SFTMessage]
    tools: list[dict[str, Any]]


class EpisodeReview(BaseModel):
    scenario_id: str
    trajectory_id: str
    accepted: bool
    quality_label: QualityLabel | None = None
    rejection_codes: list[str] = Field(default_factory=list)
    split: DatasetSplit | None = None
    example_count: int = Field(default=0, ge=0)


class DatasetManifest(BaseModel):
    schema_version: str = SFT_DATASET_SCHEMA_VERSION
    dataset_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    candidate_episodes: int
    accepted_episodes: int
    rejected_episodes: int
    exported_examples: int
    split_examples: dict[str, int]
    source_episodes: dict[str, int]
    quality_episodes: dict[str, int]
    rejection_codes: dict[str, int]
    environment_versions: list[str]
    policy_versions: list[str]
    split_group_overlap: bool
    excluded_policy_steps: int = 0
    excluded_duplicate_policy_steps: int = 0
    shared_trajectory_snapshots: int = 0


class DatasetBuildResult(BaseModel):
    manifest: DatasetManifest
    reviews: list[EpisodeReview]
    examples: list[SFTExample]


class SFTDatasetBuilder:
    """Three-level deterministic cleaning for policy action examples."""

    def __init__(self, *, max_steps: int = 16, max_exact_retries: int = 1) -> None:
        self.max_steps = max_steps
        self.max_exact_retries = max_exact_retries
        self.replay = EpisodeReplayVerifier()

    def build(self, candidates: list[EpisodeCandidate | dict[str, Any]]) -> DatasetBuildResult:
        parsed = [
            item if isinstance(item, EpisodeCandidate) else EpisodeCandidate(**item)
            for item in candidates
        ]
        self._assert_unique_ids(parsed)
        reviews: list[EpisodeReview] = []
        examples: list[SFTExample] = []
        split_groups: dict[str, set[str]] = {"train": set(), "validation": set(), "test": set()}

        for candidate in parsed:
            rejection_codes = self._review(candidate)
            quality_label = self._quality_label(candidate.episode)
            if rejection_codes or quality_label is None:
                reviews.append(
                    EpisodeReview(
                        scenario_id=candidate.scenario_id,
                        trajectory_id=candidate.episode.trajectory_id,
                        accepted=False,
                        rejection_codes=sorted(set(rejection_codes or ["L3_UNUSABLE_OUTCOME"])),
                    )
                )
                continue

            group = self._split_group(candidate)
            split = self._split_for_group(group)
            split_groups[split].add(group)
            episode_examples = self._examples(candidate, split, quality_label)
            if not episode_examples:
                reviews.append(
                    EpisodeReview(
                        scenario_id=candidate.scenario_id,
                        trajectory_id=candidate.episode.trajectory_id,
                        accepted=False,
                        rejection_codes=["L3_NO_VERIFIED_POLICY_DECISION"],
                    )
                )
                continue
            examples.extend(episode_examples)
            reviews.append(
                EpisodeReview(
                    scenario_id=candidate.scenario_id,
                    trajectory_id=candidate.episode.trajectory_id,
                    accepted=True,
                    quality_label=quality_label,
                    split=split,
                    example_count=len(episode_examples),
                )
            )

        overlap = bool(
            (split_groups["train"] & split_groups["validation"])
            or (split_groups["train"] & split_groups["test"])
            or (split_groups["validation"] & split_groups["test"])
        )
        manifest = self._manifest(parsed, reviews, examples, overlap)
        return DatasetBuildResult(manifest=manifest, reviews=reviews, examples=examples)

    def export(self, result: DatasetBuildResult, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "validation", "test"):
            rows = [item for item in result.examples if item.split == split]
            self._write_jsonl(output_dir / f"{split}.jsonl", rows)
        self._write_jsonl(output_dir / "reviews.jsonl", result.reviews)
        (output_dir / "manifest.json").write_text(
            result.manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )

    def _review(self, candidate: EpisodeCandidate) -> list[str]:
        episode = candidate.episode
        errors: list[str] = []
        errors.extend(f"L1_REPLAY:{error}" for error in self.replay.verify(episode))
        if candidate.contains_production_data and not candidate.training_authorized:
            errors.append("L1_PRODUCTION_DATA_NOT_AUTHORIZED")
        if candidate.contains_production_data and not candidate.user_partition_key:
            errors.append("L1_PRODUCTION_USER_PARTITION_MISSING")
        if not episode.steps:
            errors.append("L1_EMPTY_EPISODE")
        if len(episode.steps) > self.max_steps:
            errors.append("L1_STEP_BUDGET_EXCEEDED")
        if episode.status == "running" or not episode.termination_reason:
            errors.append("L1_TERMINATION_MISSING")
        if episode.completed_at is None or episode.content_hash is None:
            errors.append("L1_UNFINALIZED_EPISODE")
        if _contains_pii(episode.model_dump(mode="json")):
            errors.append("L1_PII_DETECTED")
        if _contains_unicode_replacement(episode.model_dump(mode="json")):
            errors.append("L1_TEXT_ENCODING_CORRUPT")

        signatures: dict[str, list[bool]] = {}
        for step in episode.steps:
            if step.action.decision_source == "controller":
                continue
            action = step.action.action
            if action not in step.context.allowed_actions:
                errors.append("L2_ACTION_NOT_ALLOWED")
            errors.extend(self._argument_errors(action, step.action.arguments))
            if action not in NO_TOOL_ACTIONS and not _arguments_grounded(
                action, step.action.arguments, step.context.model_dump(mode="json")
            ):
                errors.append("L2_ARGUMENT_NOT_GROUNDED")
            if action not in NO_TOOL_ACTIONS:
                if not step.observations:
                    errors.append("L2_TOOL_OBSERVATION_MISSING")
                for observation in step.observations:
                    if not observation.tool_call_id:
                        errors.append("L2_TOOL_CALL_ID_MISSING")
                    if observation.schema_version != episode.observation_schema_version:
                        errors.append("L2_OBSERVATION_SCHEMA_MISMATCH")
                signature = _canonical(
                    {
                        "goal_version": step.context.goal_version,
                        "plan_version": step.context.plan_version,
                        "action": action,
                        "arguments": step.action.arguments,
                    }
                )
                outcomes = signatures.setdefault(signature, [])
                outcomes.append(any(observation.ok for observation in step.observations))

        for outcomes in signatures.values():
            if len(outcomes) > self.max_exact_retries + 1:
                errors.append("L2_EXCESSIVE_IDENTICAL_RETRIES")

        if self._quality_label(episode) is None:
            errors.append("L3_UNUSABLE_OUTCOME")
        return errors

    @staticmethod
    def _argument_errors(action: str, arguments: dict[str, Any]) -> list[str]:
        if set(arguments) & _PROTECTED_ARGUMENTS:
            return ["L2_POLICY_SUPPLIED_PROTECTED_ARGUMENT"]
        try:
            validate_policy_arguments(action, arguments)
        except ValueError:
            if action not in TOOL_NAME_TO_MODEL and action not in NO_TOOL_ACTIONS:
                return ["L2_UNKNOWN_ACTION"]
            return ["L2_POLICY_ARGUMENT_INVALID"]
        return []

    @staticmethod
    def _quality_label(episode: AgentEpisode) -> QualityLabel | None:
        final = episode.final_state or {}
        artifacts = list((final.get("artifacts") or {}).values())
        reports = [item for item in artifacts if item.get("artifact_type") == "validation_report"]
        hard_pass = bool(reports and (reports[-1].get("payload") or {}).get("hard_pass"))
        has_plan = any(
            item.get("artifact_type") in {"solver_result", "itinerary_draft"} for item in artifacts
        )
        if (
            hard_pass
            and has_plan
            and episode.termination_reason
            in {
                "awaiting_user",
                "validated_finish",
            }
        ):
            return "validated_plan"

        goal = final.get("goal") or episode.initial_state.get("goal") or {}
        missing = goal.get("missing_information") or []
        last_action = episode.steps[-1].action if episode.steps else None
        if (
            episode.termination_reason == "awaiting_user"
            and missing
            and last_action
            and last_action.action == "ask_user"
            and str(last_action.arguments.get("question") or "").strip()
        ):
            return "clarification"

        capability = (goal.get("capability") or {}).get("status")
        if (
            capability in {"infeasible", "unsafe", "missing_tool"}
            and last_action
            and last_action.action in {"abort", "propose_tradeoff"}
        ):
            return "safe_termination"
        if (
            episode.termination_reason == "awaiting_user"
            and last_action
            and last_action.action == "propose_tradeoff"
            and _tradeoff_context_grounded(episode.steps[-1].context)
        ):
            return "safe_termination"
        return None

    @staticmethod
    def _split_group(candidate: EpisodeCandidate) -> str:
        if candidate.contains_production_data:
            assert candidate.user_partition_key is not None
            return "production-user:" + _sha256(candidate.user_partition_key)[:20]
        return (
            f"scenario:{candidate.template_family.strip().lower()}|{candidate.city.strip().lower()}"
        )

    @staticmethod
    def _split_for_group(group: str) -> DatasetSplit:
        bucket = int(_sha256(group)[:8], 16) % 100
        if bucket < 70:
            return "train"
        if bucket < 85:
            return "validation"
        return "test"

    @staticmethod
    def _examples(
        candidate: EpisodeCandidate,
        split: DatasetSplit,
        quality_label: QualityLabel,
    ) -> list[SFTExample]:
        result: list[SFTExample] = []
        seen_successful_calls: set[str] = set()
        for step in candidate.episode.steps:
            if step.action.decision_source == "controller":
                continue
            if not _step_verified_success(step):
                continue
            signature = _policy_step_signature(step)
            if step.action.action not in NO_TOOL_ACTIONS:
                if signature in seen_successful_calls:
                    continue
                seen_successful_calls.add(signature)
            context_json = json.dumps(
                policy_prompt_payload(step.context),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            result.append(
                SFTExample(
                    example_id=(
                        f"{candidate.scenario_id}:"
                        f"{(candidate.episode.content_hash or 'unfinalized')[:12]}:"
                        f"{step.step_index}"
                    ),
                    scenario_id=candidate.scenario_id,
                    trajectory_id=candidate.episode.trajectory_id,
                    step_index=step.step_index,
                    split=split,
                    quality_label=quality_label,
                    source=candidate.source,
                    environment_version=candidate.episode.environment_version,
                    policy_name=candidate.episode.policy_name,
                    policy_version=candidate.episode.policy_version,
                    messages=[
                        SFTMessage(role="system", content=AGENT_TOOL_POLICY_SYSTEM_PROMPT),
                        SFTMessage(role="user", content=context_json),
                        SFTMessage(
                            role="assistant",
                            tool_calls=[
                                SFTToolCall(
                                    function=SFTToolFunction(
                                        name=step.action.action,
                                        arguments=step.action.arguments,
                                    )
                                )
                            ],
                        ),
                    ],
                    tools=policy_action_schemas(step.context.allowed_actions),
                )
            )
        return result

    @staticmethod
    def _assert_unique_ids(candidates: list[EpisodeCandidate]) -> None:
        scenario_ids = [item.scenario_id for item in candidates]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("duplicate scenario_id in candidate set")
        episode_snapshot_ids = [
            (item.episode.trajectory_id, item.episode.content_hash) for item in candidates
        ]
        if len(episode_snapshot_ids) != len(set(episode_snapshot_ids)):
            raise ValueError("duplicate episode snapshot in candidate set")

    @staticmethod
    def _manifest(
        candidates: list[EpisodeCandidate],
        reviews: list[EpisodeReview],
        examples: list[SFTExample],
        overlap: bool,
    ) -> DatasetManifest:
        accepted = [review for review in reviews if review.accepted]
        rejected = [review for review in reviews if not review.accepted]
        version_payload = {
            "candidates": sorted(
                (item.scenario_id, item.episode.content_hash) for item in candidates
            ),
            # Dataset identity must also change when the model-visible projection,
            # tool schema, or target changes while the source episode stays fixed.
            # Hashing IDs alone made context-compaction releases indistinguishable.
            "examples": sorted(
                (
                    item.example_id,
                    _sha256(
                        _canonical(
                            {
                                "messages": [
                                    message.model_dump(mode="json")
                                    for message in item.messages
                                ],
                                "tools": item.tools,
                            }
                        )
                    ),
                )
                for item in examples
            ),
        }
        return DatasetManifest(
            dataset_version="sft-" + _sha256(_canonical(version_payload))[:16],
            candidate_episodes=len(candidates),
            accepted_episodes=len(accepted),
            rejected_episodes=len(rejected),
            exported_examples=len(examples),
            split_examples=dict(Counter(item.split for item in examples)),
            source_episodes=dict(
                Counter(
                    candidate.source
                    for candidate, review in zip(candidates, reviews, strict=True)
                    if review.accepted
                )
            ),
            quality_episodes=dict(
                Counter(review.quality_label for review in accepted if review.quality_label)
            ),
            rejection_codes=dict(
                Counter(code for review in rejected for code in review.rejection_codes)
            ),
            environment_versions=sorted({item.episode.environment_version for item in candidates}),
            policy_versions=sorted(
                {f"{item.episode.policy_name}:{item.episode.policy_version}" for item in candidates}
            ),
            split_group_overlap=overlap,
            excluded_policy_steps=sum(
                1
                for candidate in candidates
                for step in candidate.episode.steps
                if step.action.decision_source != "controller" and not _step_verified_success(step)
            ),
            excluded_duplicate_policy_steps=sum(
                _duplicate_verified_policy_steps(candidate.episode)
                for candidate in candidates
            ),
            shared_trajectory_snapshots=(
                len(candidates)
                - len({item.episode.trajectory_id for item in candidates})
            ),
        )

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[BaseModel]) -> None:
        content = "\n".join(row.model_dump_json() for row in rows)
        path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def _trusted_hydrated_fields(action: str) -> set[str]:
    mapping = {
        "get_weather": {"city"},
        "search_pois": {"city"},
        "get_poi_detail": {"poi_name", "city"},
        "get_route_matrix": {"pois", "constraints", "amap_minutes"},
        "solve_itinerary": {
            "pois",
            "constraints",
            "dist_matrix",
            "tc_matrix",
            "amap_minutes",
        },
        "validate_itinerary": {"itinerary", "constraints", "facts"},
    }
    return mapping.get(action, set())


def _step_verified_success(step: Any) -> bool:
    """Only imitate decisions that the environment actually verified."""
    task_status = str((step.verification or {}).get("task_status") or "")
    error_code = (step.verification or {}).get("error_code")
    if task_status:
        if task_status == "blocked" and step.action.action in {
            "ask_user",
            "finish",
        }:
            return True
        if task_status == "blocked" and step.action.action == "propose_tradeoff":
            return _tradeoff_context_grounded(step.context)
        if (
            task_status == "failed"
            and step.action.action == "abort"
            and step.context.capability.get("status") in {"infeasible", "unsafe", "missing_tool"}
            and step.context.capability.get("actionable_alternatives") is False
        ):
            # ``abort`` is represented as a failed task transition so the
            # loop cannot continue, but it is a verified successful policy
            # decision when the controller proves that no alternative exists.
            return True
        if task_status in {"succeeded", "skipped"}:
            return not error_code
        if task_status == "ready" and step.action.action not in NO_TOOL_ACTIONS:
            # ReAct research actions can succeed while leaving the enclosing
            # evidence-gathering task ready for another tool decision. The
            # action-level observations, not terminal task status, are the
            # verifier signal for these intermediate decisions.
            return (
                not error_code
                and bool(step.observations)
                and all(observation.ok for observation in step.observations)
            )
        return False
    # Backward-compatible teacher fixtures may use an explicit boolean.
    return (step.verification or {}).get("passed") is True


def _tradeoff_context_grounded(context: Any) -> bool:
    capability = context.capability or {}
    if capability.get("status") in {"infeasible", "unsafe", "missing_tool"}:
        return True
    current = context.current_subtask or {}
    attempts = current.get("action_attempt_counts") or {}
    verifier_attempts = int(attempts.get("finalize_research") or 0) + int(
        attempts.get("validate_itinerary") or 0
    )
    evidence_attempts = sum(
        int(attempts.get(action) or 0)
        for action in {
            "get_weather",
            "retrieve_city_knowledge",
            "search_current_info",
            "search_pois",
            "search_transport",
        }
    )
    return verifier_attempts > 0 and evidence_attempts > 0


def _policy_step_signature(step: Any) -> str:
    return _canonical(
        {
            "goal_version": step.context.goal_version,
            "plan_version": step.context.plan_version,
            "action": step.action.action,
            "arguments": step.action.arguments,
        }
    )


def _duplicate_verified_policy_steps(episode: AgentEpisode) -> int:
    seen: set[str] = set()
    duplicates = 0
    for step in episode.steps:
        if (
            step.action.decision_source == "controller"
            or step.action.action in NO_TOOL_ACTIONS
            or not _step_verified_success(step)
        ):
            continue
        signature = _policy_step_signature(step)
        if signature in seen:
            duplicates += 1
        else:
            seen.add(signature)
    return duplicates


def _arguments_grounded(action: str, arguments: dict[str, Any], context: dict[str, Any]) -> bool:
    if not arguments:
        return True
    grounded = _canonical(context).casefold()
    for name, value in arguments.items():
        if (
            name in _trusted_hydrated_fields(action)
            or name in _GENERATIVE_ARGUMENTS.get(action, set())
            or name in _SCHEMA_ENUM_ARGUMENTS
        ):
            continue
        for leaf in _argument_leaves(value):
            normalized = str(leaf).strip().casefold()
            if normalized and normalized not in grounded:
                return False
    return True


def _argument_leaves(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _argument_leaves(item)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _argument_leaves(item)]
    return [value]


def _contains_pii(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_pii(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_pii(item) for item in value)
    if isinstance(value, str):
        return bool(_PHONE.search(value) or _EMAIL.search(value) or _ID_CARD.search(value))
    return False


def _contains_unicode_replacement(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_unicode_replacement(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unicode_replacement(item) for item in value)
    return isinstance(value, str) and "\ufffd" in value


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
