"""Fail-closed governance for post-training promotion evaluation.

This module validates the public protocol and hashed lineage/access metadata. It
must never open promotion or sealed case payloads as part of a protocol audit.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


SCHEMA_VERSION = "travelagent-posttraining-promotion-protocol.v1"
ACCESS_SCHEMA_VERSION = "travelagent-promotion-access-event.v1"
LINEAGE_SCHEMA_VERSION = "travelagent-eval-lineage.v1"
READINESS_SCHEMA_VERSION = "travelagent-promotion-readiness-state.v1"

TRAIN_ELIGIBLE_SPLITS = frozenset(
    {
        "train",
        "train-hard-donor",
        "train-shadow",
        "router-train",
        "router-calibration",
    }
)
FORBIDDEN_DONOR_SPLITS = frozenset(
    {
        "hard-challenge-h003-h004",
        "internal-dev",
        "promotion-val-v1",
        "sealed-160",
    }
)
REQUIRED_BASELINE_MATRIX = frozenset(
    {"H-001+H-005", "H-004+H-005", "H-006+H-005"}
)
REQUIRED_LINEAGE_FIELDS = frozenset(
    {
        "source_state_id",
        "source_hash",
        "template_lineage",
        "tool_snapshot_hash",
        "generator_model",
        "generator_prompt_hash",
        "parent_trajectory_id",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class MetricGates(BaseModel):
    full_success_mean_4_min: float = Field(ge=0, le=1)
    h001_paired_ci_lower_min: float
    hard_gain_vs_h001_point_min: float
    hard_gain_vs_h001_ci_lower_min: float
    hard_noninferiority_vs_h004_ci_lower_min: float
    easy_noninferiority_vs_h001_ci_lower_min: float
    pass4_noninferiority_vs_h001_ci_lower_min: float
    contract_success_min: float = Field(ge=0, le=1)
    unauthorized_actions_max: int = Field(ge=0)


class StatisticalContract(BaseModel):
    primary_unit: Literal["source_state_id"]
    rollouts_per_state: int = Field(ge=1)
    paired_method: Literal["source-state-clustered-bootstrap"]
    confidence_level: float = Field(gt=0, lt=1)
    binary_test: Literal["mcnemar"]
    multiple_comparison_correction: Literal["holm"]
    power_analysis_required_before_freeze: bool
    provisional_source_states: int = Field(ge=1)
    repeated_rollouts_increase_independent_n: Literal[False]


class DataGovernance(BaseModel):
    development_splits: list[str]
    promotion_split: Literal["promotion-val-v1"]
    sealed_split: Literal["sealed-160"]
    required_lineage_fields: list[str]
    forbidden_training_sources: list[str]
    source_equal_weighting: bool
    max_success_targets_per_source: int = Field(ge=1)
    source_mix_tolerance_pp: float = Field(ge=0)
    supervised_token_mix_tolerance_pp: float = Field(ge=0)
    causal_visibility_audit_required: bool


class AccessPolicy(BaseModel):
    promotion_campaigns_per_experiment_family: int = Field(ge=0)
    sealed_campaigns_total: int = Field(ge=0)
    append_only_hash_chain_required: bool
    configuration_frozen_before_access: bool
    promotion_failure_reusable_for_tuning: Literal[False]
    sealed_failure_can_revive_same_candidate: Literal[False]


class EvaluationContract(BaseModel):
    same_stack_baselines: list[str]
    raw_model_and_assembled_system_reported_separately: bool
    policy_temperature: float = Field(ge=0)
    inference_seeds: list[int] = Field(min_length=1)
    selection_uses_promotion_results: Literal[False]
    llm_judge_is_promotion_authority: Literal[False]


class PromotionReadiness(BaseModel):
    schema_version: Literal[READINESS_SCHEMA_VERSION]
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    protocol_locked: bool
    promotion_dataset_authored: bool
    power_analysis_completed: bool
    lineage_audit_passed: bool
    h006_gpu_training_authorized: bool
    blocking_reasons: list[str]

    @model_validator(mode="after")
    def authorization_is_fail_closed(self) -> "PromotionReadiness":
        prerequisites = (
            self.protocol_locked
            and self.promotion_dataset_authored
            and self.power_analysis_completed
            and self.lineage_audit_passed
        )
        if self.h006_gpu_training_authorized and not prerequisites:
            raise ValueError("H-006 cannot be authorized before all prerequisites pass")
        return self


class PromotionProtocol(BaseModel):
    schema_version: Literal[SCHEMA_VERSION]
    protocol_id: Literal["promotion-val-v1"]
    status: Literal["frozen"]
    decision_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    metrics: MetricGates
    statistics: StatisticalContract
    data_governance: DataGovernance
    access_policy: AccessPolicy
    evaluation_contract: EvaluationContract


class LineageRecord(BaseModel):
    schema_version: Literal[LINEAGE_SCHEMA_VERSION] = LINEAGE_SCHEMA_VERSION
    record_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    source_state_id: str = Field(min_length=1)
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    template_lineage: str = Field(min_length=1)
    tool_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    generator_model: str = Field(min_length=1)
    generator_prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_trajectory_id: str = Field(min_length=1)


class PromotionAccessEvent(BaseModel):
    schema_version: Literal[ACCESS_SCHEMA_VERSION] = ACCESS_SCHEMA_VERSION
    sequence: int = Field(ge=0)
    event_type: Literal["campaign_opened", "campaign_closed"]
    experiment_family: str = Field(pattern=r"^[A-Z][A-Z0-9-]+$")
    split: Literal["promotion-val-v1"]
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_run_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    actor: str = Field(min_length=1)
    reason: str = Field(min_length=3)
    previous_event_sha256: str
    event_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def previous_hash_is_valid(self) -> "PromotionAccessEvent":
        if self.previous_event_sha256 != "GENESIS" and not _SHA256_RE.fullmatch(
            self.previous_event_sha256
        ):
            raise ValueError("previous_event_sha256 must be GENESIS or a sha256")
        return self


def access_event_hash(event: PromotionAccessEvent | dict[str, Any]) -> str:
    payload = event.model_dump(mode="json") if isinstance(event, PromotionAccessEvent) else dict(event)
    payload.pop("event_sha256", None)
    return canonical_hash(payload)


def audit_protocol(protocol: PromotionProtocol) -> dict[str, Any]:
    metrics = protocol.metrics
    statistics = protocol.statistics
    governance = protocol.data_governance
    access = protocol.access_policy
    evaluation = protocol.evaluation_contract
    gates = {
        "protocol_status_frozen": protocol.status == "frozen",
        "four_rollouts_per_state": statistics.rollouts_per_state == 4,
        "independent_unit_is_source_state": statistics.primary_unit == "source_state_id",
        "clustered_paired_statistics": (
            statistics.paired_method == "source-state-clustered-bootstrap"
            and statistics.confidence_level == 0.95
            and statistics.binary_test == "mcnemar"
            and statistics.multiple_comparison_correction == "holm"
            and not statistics.repeated_rollouts_increase_independent_n
        ),
        "power_analysis_required": statistics.power_analysis_required_before_freeze,
        "full_success_gate_090": metrics.full_success_mean_4_min == 0.90,
        "paired_gain_ci_positive": metrics.h001_paired_ci_lower_min == 0.0,
        "hard_gain_gate": (
            metrics.hard_gain_vs_h001_point_min == 0.10
            and metrics.hard_gain_vs_h001_ci_lower_min == 0.0
        ),
        "noninferiority_margin_2pp": (
            metrics.hard_noninferiority_vs_h004_ci_lower_min == -0.02
            and metrics.easy_noninferiority_vs_h001_ci_lower_min == -0.02
            and metrics.pass4_noninferiority_vs_h001_ci_lower_min == -0.02
        ),
        "hard_safety_gates": (
            metrics.contract_success_min == 1.0 and metrics.unauthorized_actions_max == 0
        ),
        "same_stack_baseline_matrix": set(evaluation.same_stack_baselines)
        == REQUIRED_BASELINE_MATRIX,
        "raw_and_assembled_separate": evaluation.raw_model_and_assembled_system_reported_separately,
        "selection_does_not_use_promotion": not evaluation.selection_uses_promotion_results,
        "judge_not_authority": not evaluation.llm_judge_is_promotion_authority,
        "required_lineage_fields": REQUIRED_LINEAGE_FIELDS.issubset(
            governance.required_lineage_fields
        ),
        "forbidden_donors_registered": FORBIDDEN_DONOR_SPLITS.issubset(
            governance.forbidden_training_sources
        ),
        "source_weighting_locked": (
            governance.source_equal_weighting
            and governance.max_success_targets_per_source <= 2
            and governance.source_mix_tolerance_pp <= 5
            and governance.supervised_token_mix_tolerance_pp <= 5
            and governance.causal_visibility_audit_required
        ),
        "development_isolation": (
            governance.promotion_split not in governance.development_splits
            and governance.sealed_split not in governance.development_splits
        ),
        "one_promotion_campaign": access.promotion_campaigns_per_experiment_family == 1,
        "one_sealed_campaign": access.sealed_campaigns_total == 1,
        "append_only_access_log": access.append_only_hash_chain_required,
        "freeze_before_access": access.configuration_frozen_before_access,
    }
    return {
        "schema_version": "travelagent-posttraining-promotion-protocol-audit.v1",
        "passed": all(gates.values()),
        "gates": gates,
        "protocol_sha256": canonical_hash(protocol.model_dump(mode="json")),
    }


def audit_readiness(
    readiness: PromotionReadiness, *, protocol_file_sha256: str
) -> dict[str, Any]:
    prerequisites = {
        "protocol_hash_matches": readiness.protocol_sha256 == protocol_file_sha256,
        "protocol_locked": readiness.protocol_locked,
        "promotion_dataset_authored": readiness.promotion_dataset_authored,
        "power_analysis_completed": readiness.power_analysis_completed,
        "lineage_audit_passed": readiness.lineage_audit_passed,
    }
    dataset_ready = all(prerequisites.values())
    return {
        "schema_version": "travelagent-promotion-readiness-audit.v1",
        "passed": prerequisites["protocol_hash_matches"]
        and (not readiness.h006_gpu_training_authorized or dataset_ready),
        "prerequisites": prerequisites,
        "dataset_ready": dataset_ready,
        "h006_gpu_training_authorized": bool(
            dataset_ready and readiness.h006_gpu_training_authorized
        ),
        "blocking_reasons": readiness.blocking_reasons,
    }


def _cross_role_overlaps(
    records: list[LineageRecord], key_fn
) -> list[dict[str, Any]]:
    roles: dict[str, set[str]] = defaultdict(set)
    record_ids: dict[str, list[str]] = defaultdict(list)
    for record in records:
        key = key_fn(record)
        roles[key].add(record.split)
        record_ids[key].append(record.record_id)
    findings = []
    for key, splits in roles.items():
        if not (splits & TRAIN_ELIGIBLE_SPLITS and splits & FORBIDDEN_DONOR_SPLITS):
            continue
        findings.append(
            {
                "lineage_key_sha256": canonical_hash(key),
                "splits": sorted(splits),
                "record_id_hashes": sorted(canonical_hash(item) for item in record_ids[key]),
            }
        )
    return sorted(findings, key=lambda item: item["lineage_key_sha256"])


def audit_lineage_isolation(records: list[LineageRecord]) -> dict[str, Any]:
    duplicate_record_ids = [
        record_id for record_id, count in Counter(row.record_id for row in records).items() if count > 1
    ]
    unknown_splits = sorted(
        {row.split for row in records} - TRAIN_ELIGIBLE_SPLITS - FORBIDDEN_DONOR_SPLITS
    )
    source_id_overlaps = _cross_role_overlaps(records, lambda row: row.source_state_id)
    source_hash_overlaps = _cross_role_overlaps(records, lambda row: row.source_hash)
    lineage_overlaps = _cross_role_overlaps(
        records,
        lambda row: "|".join(
            (row.source_state_id, row.template_lineage, row.tool_snapshot_hash)
        ),
    )
    gates = {
        "records_registered": bool(records),
        "unique_record_ids": not duplicate_record_ids,
        "known_splits": not unknown_splits,
        "source_state_ids_isolated": not source_id_overlaps,
        "source_hashes_isolated": not source_hash_overlaps,
        "lineage_tuples_isolated": not lineage_overlaps,
    }
    return {
        "schema_version": "travelagent-eval-lineage-audit.v1",
        "passed": all(gates.values()),
        "gates": gates,
        "counts": {
            "records": len(records),
            "splits": dict(sorted(Counter(row.split for row in records).items())),
        },
        "duplicate_record_id_hashes": sorted(canonical_hash(item) for item in duplicate_record_ids),
        "unknown_splits": unknown_splits,
        "source_state_id_overlaps": source_id_overlaps,
        "source_hash_overlaps": source_hash_overlaps,
        "lineage_tuple_overlaps": lineage_overlaps,
        "privacy_note": "Overlap findings expose only hashes, never protected case content.",
    }


def audit_access_log(
    events: list[PromotionAccessEvent], *, max_campaigns_per_family: int = 1
) -> dict[str, Any]:
    errors: list[str] = []
    previous = "GENESIS"
    open_families: set[str] = set()
    opened_counts: Counter[str] = Counter()
    for expected_sequence, event in enumerate(events):
        if event.sequence != expected_sequence:
            errors.append(f"sequence:{event.sequence}:expected:{expected_sequence}")
        if event.previous_event_sha256 != previous:
            errors.append(f"previous_hash_mismatch:{event.sequence}")
        expected_hash = access_event_hash(event)
        if event.event_sha256 != expected_hash:
            errors.append(f"event_hash_mismatch:{event.sequence}")
        if event.event_type == "campaign_opened":
            opened_counts[event.experiment_family] += 1
            if event.experiment_family in open_families:
                errors.append(f"campaign_already_open:{event.experiment_family}")
            open_families.add(event.experiment_family)
        else:
            if event.experiment_family not in open_families:
                errors.append(f"campaign_close_without_open:{event.experiment_family}")
            open_families.discard(event.experiment_family)
        previous = event.event_sha256
    over_budget = sorted(
        family for family, count in opened_counts.items() if count > max_campaigns_per_family
    )
    if over_budget:
        errors.extend(f"campaign_budget_exceeded:{family}" for family in over_budget)
    return {
        "schema_version": "travelagent-promotion-access-log-audit.v1",
        "passed": not errors,
        "events": len(events),
        "campaigns_opened": dict(sorted(opened_counts.items())),
        "currently_open_families": sorted(open_families),
        "errors": errors,
        "head_sha256": previous,
    }


__all__ = [
    "ACCESS_SCHEMA_VERSION",
    "FORBIDDEN_DONOR_SPLITS",
    "LINEAGE_SCHEMA_VERSION",
    "READINESS_SCHEMA_VERSION",
    "PromotionAccessEvent",
    "PromotionProtocol",
    "PromotionReadiness",
    "LineageRecord",
    "SCHEMA_VERSION",
    "TRAIN_ELIGIBLE_SPLITS",
    "access_event_hash",
    "audit_access_log",
    "audit_lineage_isolation",
    "audit_protocol",
    "audit_readiness",
    "canonical_hash",
]
