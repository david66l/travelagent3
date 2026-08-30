"""Contracts and leakage audit for the independent TravelAgent benchmark."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


SCHEMA_VERSION = "travel-agent-external-benchmark.v1"


class BenchmarkSource(StrEnum):
    AUTHORIZED_REAL_OR_SIMULATED = "authorized_real_or_simulated"
    HUMAN_ORIGINAL_CONSTRAINT = "human_original_constraint"
    TOOL_FAILURE = "tool_failure"
    LONG_CONTEXT_REPLAN = "long_context_replan"


class BenchmarkSplit(StrEnum):
    DEV = "dev"
    SEALED_TEST = "sealed_test"


class ExpectedTermination(StrEnum):
    PLAN = "plan"
    CLARIFICATION = "clarification"
    TRADEOFF = "tradeoff"
    SAFE_ABORT = "safe_abort"


class Provenance(BaseModel):
    authoring_method: Literal["authorized_real", "human_original", "simulated"]
    permission_basis: str = Field(min_length=3)
    deidentified: bool
    template_independent: bool
    author_group: str = Field(min_length=1)


class GroupKeys(BaseModel):
    request_family: str = Field(min_length=1)
    city_cluster: str = Field(min_length=1)
    date_pattern: str = Field(min_length=1)
    constraint_combo: str = Field(min_length=1)
    failure_template: str = "none"


class HardConstraint(BaseModel):
    constraint_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    description: str = Field(min_length=3)
    verifier: str = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


class ArgumentRule(BaseModel):
    action: str = Field(min_length=1)
    required: dict[str, Any] = Field(default_factory=dict)
    forbidden_keys: list[str] = Field(default_factory=list)


class FaultInjection(BaseModel):
    fault_type: Literal["empty_result", "timeout", "rate_limit", "stale_data", "invalid_argument"]
    trigger_step: int = Field(ge=1)
    recoverable: bool


class OutcomeRubric(BaseModel):
    success: str = Field(min_length=3)
    partial_success: str = Field(min_length=3)
    failure: str = Field(min_length=3)
    safe_termination: str = Field(min_length=3)


class IndependentAnnotation(BaseModel):
    annotator_id: str = Field(min_length=1)
    primary_action: str = Field(min_length=1)
    allowed_actions: list[str] = Field(min_length=1)
    hard_constraint_labels: dict[str, bool] = Field(default_factory=dict)
    notes: str | None = None


class Adjudication(BaseModel):
    adjudicator_id: str = Field(min_length=1)
    primary_action: str = Field(min_length=1)
    reason: str = Field(min_length=3)


class ForbiddenCorpusDocument(BaseModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class ExternalBenchmarkCase(BaseModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    case_id: str = Field(pattern=r"^ext-v1-[a-z0-9-]+$")
    split: BenchmarkSplit
    source: BenchmarkSource
    difficulty: Literal["L1", "L2", "L3", "L4"]
    provenance: Provenance
    group_keys: GroupKeys
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_fixture: dict[str, Any] = Field(default_factory=dict)
    fault_injection: FaultInjection | None = None
    allowed_actions: list[str] = Field(min_length=1)
    argument_rules: list[ArgumentRule] = Field(default_factory=list)
    hard_constraints: list[HardConstraint] = Field(min_length=1)
    acceptable_clarification_categories: list[str] = Field(default_factory=list)
    acceptable_tradeoff_categories: list[str] = Field(default_factory=list)
    expected_termination: ExpectedTermination
    outcome_rubric: OutcomeRubric
    max_steps: int = Field(ge=1, le=32)
    annotations: list[IndependentAnnotation] = Field(default_factory=list)
    adjudication: Adjudication | None = None

    @model_validator(mode="after")
    def validate_provenance_and_fault(self) -> "ExternalBenchmarkCase":
        if not self.provenance.deidentified:
            raise ValueError("benchmark cases must be deidentified")
        if self.source == BenchmarkSource.TOOL_FAILURE and self.fault_injection is None:
            raise ValueError("tool_failure cases require fault_injection")
        if self.source != BenchmarkSource.TOOL_FAILURE and self.fault_injection is not None:
            raise ValueError("fault_injection is only valid for tool_failure cases")
        if self.expected_termination == ExpectedTermination.CLARIFICATION:
            if not self.acceptable_clarification_categories:
                raise ValueError("clarification cases require acceptable categories")
        if self.expected_termination == ExpectedTermination.TRADEOFF:
            if not self.acceptable_tradeoff_categories:
                raise ValueError("tradeoff cases require acceptable categories")
        return self


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\W_]+", "", normalized, flags=re.UNICODE)


def normalized_prompt(case: ExternalBenchmarkCase) -> str:
    user_text = "\n".join(
        str(message.get("content") or "")
        for message in case.messages
        if message.get("role") == "user"
    )
    return normalize_text(user_text)


def normalized_prompt_hash(case: ExternalBenchmarkCase) -> str:
    return hashlib.sha256(normalized_prompt(case).encode("utf-8")).hexdigest()


def tool_payload_hash(case: ExternalBenchmarkCase) -> str:
    return canonical_hash(case.tool_fixture)


def constraint_fingerprint(case: ExternalBenchmarkCase) -> str:
    constraints = [
        {
            "kind": item.kind,
            "verifier": item.verifier,
            "params": item.params,
        }
        for item in case.hard_constraints
    ]
    return canonical_hash(sorted(constraints, key=canonical_hash))


def group_signature(case: ExternalBenchmarkCase) -> str:
    return canonical_hash(case.group_keys.model_dump(mode="json"))


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("Cohen's kappa requires paired non-empty labels")
    total = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b, strict=True)) / total
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    expected = sum(
        (count_a[label] / total) * (count_b[label] / total) for label in set(count_a) | set(count_b)
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1.0 - expected), 8)


def _overlap(dev: list[ExternalBenchmarkCase], test: list[ExternalBenchmarkCase], fn) -> int:
    return len({fn(case) for case in dev} & {fn(case) for case in test})


def audit_split_isolation(
    dev: list[ExternalBenchmarkCase], test: list[ExternalBenchmarkCase]
) -> dict[str, Any]:
    overlap = {
        "case_ids": len({case.case_id for case in dev} & {case.case_id for case in test}),
        "normalized_prompts": _overlap(dev, test, normalized_prompt_hash),
        "tool_payloads": _overlap(dev, test, tool_payload_hash),
        "constraint_fingerprints": _overlap(dev, test, constraint_fingerprint),
        "group_signatures": _overlap(dev, test, group_signature),
    }
    return {"passed": not any(overlap.values()), "overlap_counts": overlap}


def character_ngrams(text: str, *, size: int = 5) -> set[str]:
    normalized = normalize_text(text)
    if not normalized:
        return set()
    if len(normalized) <= size:
        return {normalized}
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def audit_training_contamination(
    cases: list[ExternalBenchmarkCase],
    documents: list[ForbiddenCorpusDocument],
    *,
    similarity_threshold: float = 0.82,
) -> dict[str, Any]:
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in (0, 1]")
    normalized_documents = [normalize_text(document.text) for document in documents]
    document_ngrams = [character_ngrams(text) for text in normalized_documents]
    exact_index: dict[str, list[int]] = {}
    ngram_index: dict[str, set[int]] = {}
    for index, normalized in enumerate(normalized_documents):
        if not normalized:
            continue
        exact_index.setdefault(normalized, []).append(index)
        for ngram in document_ngrams[index]:
            ngram_index.setdefault(ngram, set()).add(index)

    exact_matches = 0
    near_matches = 0
    dev_findings = []
    sealed_finding_hashes = []
    max_similarity = 0.0
    for case in cases:
        prompt = normalized_prompt(case)
        grams = character_ngrams(prompt)
        exact = exact_index.get(prompt, [])
        candidate_indices: set[int] = set()
        for gram in grams:
            candidate_indices.update(ngram_index.get(gram, set()))
        best_index = None
        best_similarity = 0.0
        for index in candidate_indices:
            similarity = jaccard_similarity(grams, document_ngrams[index])
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index
        max_similarity = max(max_similarity, best_similarity)
        finding_type = None
        if exact:
            exact_matches += 1
            finding_type = "exact"
            best_index = exact[0]
            best_similarity = 1.0
        elif best_index is not None and best_similarity >= similarity_threshold:
            near_matches += 1
            finding_type = "near_duplicate"
        if finding_type is None or best_index is None:
            continue
        finding = {
            "type": finding_type,
            "similarity": round(best_similarity, 8),
            "document_id": documents[best_index].document_id,
        }
        if case.split == BenchmarkSplit.DEV:
            dev_findings.append({"case_id": case.case_id, **finding})
        else:
            sealed_finding_hashes.append(canonical_hash({"case_id": case.case_id, **finding}))
    return {
        "passed": exact_matches == 0 and near_matches == 0,
        "documents": len(documents),
        "cases": len(cases),
        "similarity_threshold": similarity_threshold,
        "exact_matches": exact_matches,
        "near_duplicate_matches": near_matches,
        "max_similarity": round(max_similarity, 8),
        "dev_findings": dev_findings,
        "sealed_finding_hashes": sealed_finding_hashes,
        "privacy_note": "Sealed findings expose only canonical hashes, never prompts or case IDs.",
    }


def annotation_agreement(test: list[ExternalBenchmarkCase]) -> dict[str, Any]:
    labels_a = []
    labels_b = []
    missing_double_annotation = 0
    unresolved_conflicts = 0
    for case in test:
        if len(case.annotations) < 2:
            missing_double_annotation += 1
            continue
        first, second = case.annotations[:2]
        labels_a.append(first.primary_action)
        labels_b.append(second.primary_action)
        if first.primary_action != second.primary_action and case.adjudication is None:
            unresolved_conflicts += 1
    kappa = cohens_kappa(labels_a, labels_b) if labels_a else None
    return {
        "passed": (
            missing_double_annotation == 0
            and unresolved_conflicts == 0
            and kappa is not None
            and kappa >= 0.75
        ),
        "paired_cases": len(labels_a),
        "missing_double_annotation": missing_double_annotation,
        "unresolved_conflicts": unresolved_conflicts,
        "primary_action_kappa": kappa,
        "target_kappa": 0.75,
    }


def audit_external_benchmark(
    cases: list[ExternalBenchmarkCase], *, sealed_access_events: int
) -> dict[str, Any]:
    dev = [case for case in cases if case.split == BenchmarkSplit.DEV]
    test = [case for case in cases if case.split == BenchmarkSplit.SEALED_TEST]
    source_counts = Counter(case.source.value for case in cases)
    expected_sources = {
        BenchmarkSource.AUTHORIZED_REAL_OR_SIMULATED.value: 200,
        BenchmarkSource.HUMAN_ORIGINAL_CONSTRAINT.value: 150,
        BenchmarkSource.TOOL_FAILURE.value: 100,
        BenchmarkSource.LONG_CONTEXT_REPLAN.value: 50,
    }
    original_count = sum(
        case.provenance.authoring_method in {"authorized_real", "human_original"}
        and case.provenance.template_independent
        for case in cases
    )
    isolation = audit_split_isolation(dev, test)
    agreement = annotation_agreement(test)
    gates = {
        "total_500": len(cases) == 500,
        "dev_100": len(dev) == 100,
        "sealed_test_400": len(test) == 400,
        "source_composition_exact": dict(source_counts) == expected_sources,
        "human_or_authorized_original_at_least_60_percent": original_count >= 300,
        "unique_case_ids": len({case.case_id for case in cases}) == len(cases),
        "split_isolation": isolation["passed"],
        "double_annotation_and_kappa": agreement["passed"],
        "sealed_test_unaccessed": sealed_access_events == 0,
    }
    return {
        "schema_version": "travel-agent-external-benchmark-audit.v1",
        "passed": all(gates.values()),
        "gates": gates,
        "counts": {
            "total": len(cases),
            "dev": len(dev),
            "sealed_test": len(test),
            "sources": dict(source_counts),
            "original": original_count,
        },
        "split_isolation": isolation,
        "annotation_agreement": agreement,
        "sealed_access_events": sealed_access_events,
        "dataset_content_sha256": canonical_hash(
            sorted(canonical_hash(case.model_dump(mode="json")) for case in cases)
        ),
    }


__all__ = [
    "Adjudication",
    "ArgumentRule",
    "BenchmarkSource",
    "BenchmarkSplit",
    "ExpectedTermination",
    "ExternalBenchmarkCase",
    "FaultInjection",
    "ForbiddenCorpusDocument",
    "GroupKeys",
    "HardConstraint",
    "IndependentAnnotation",
    "OutcomeRubric",
    "Provenance",
    "annotation_agreement",
    "audit_external_benchmark",
    "audit_split_isolation",
    "audit_training_contamination",
    "character_ngrams",
    "cohens_kappa",
    "constraint_fingerprint",
    "group_signature",
    "normalized_prompt_hash",
    "normalize_text",
    "tool_payload_hash",
]
