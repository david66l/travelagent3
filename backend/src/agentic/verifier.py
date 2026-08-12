"""Programmatic subtask verification; model claims are never sufficient."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic.state import ArtifactRecord, FactRecord, TaskNode
from agentic.observations import ObservationEnvelope


class VerificationResult(BaseModel):
    passed: bool
    evidence_refs: list[str] = Field(default_factory=list)
    failure_codes: list[str] = Field(default_factory=list)


class SubtaskVerifier:
    def verify(
        self,
        task: TaskNode,
        *,
        facts: dict[str, FactRecord] | None = None,
        artifacts: dict[str, ArtifactRecord] | None = None,
        observations: list[ObservationEnvelope | dict[str, Any]] | None = None,
    ) -> VerificationResult:
        facts = facts or {}
        artifacts = artifacts or {}
        parsed_observations = [
            item if isinstance(item, ObservationEnvelope) else ObservationEnvelope(**item)
            for item in (observations or [])
        ]
        criteria = task.success_criteria
        evidence: list[str] = []
        failures: list[str] = []

        required_fact_keys = set(task.required_facts) | set(
            criteria.get("required_fact_keys") or []
        )
        facts_by_key = {fact.key: fact for fact in facts.values()}
        for key in sorted(required_fact_keys):
            fact = facts_by_key.get(key)
            if fact is None:
                failures.append(f"MISSING_FACT:{key}")
            else:
                evidence.extend([fact.fact_id, fact.observation_ref])

        required_artifact_types = set(criteria.get("required_artifact_types") or [])
        artifacts_by_type: dict[str, list[ArtifactRecord]] = {}
        for artifact in artifacts.values():
            artifacts_by_type.setdefault(artifact.artifact_type, []).append(artifact)
        for artifact_type in sorted(required_artifact_types):
            matches = artifacts_by_type.get(artifact_type, [])
            if not matches:
                failures.append(f"MISSING_ARTIFACT:{artifact_type}")
            else:
                evidence.extend(item.artifact_id for item in matches)

        successful = [observation for observation in parsed_observations if observation.ok]
        minimum = int(criteria.get("min_successful_observations") or 0)
        if len(successful) < minimum:
            failures.append("INSUFFICIENT_SUCCESSFUL_OBSERVATIONS")
        evidence.extend(
            observation.tool_call_id
            for observation in successful
            if observation.tool_call_id is not None
        )

        if criteria.get("require_hard_pass"):
            reports = artifacts_by_type.get("validation_report", [])
            if not any(report.payload.get("hard_pass") is True for report in reports):
                failures.append("HARD_CONSTRAINT_FAILED")

        if not criteria and not task.required_facts:
            failures.append("MISSING_SUCCESS_CRITERIA")

        return VerificationResult(
            passed=not failures,
            evidence_refs=list(dict.fromkeys(evidence)),
            failure_codes=failures,
        )
