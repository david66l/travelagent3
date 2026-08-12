"""Tests for evidence-based subtask verification."""

from agentic.observations import ObservationEnvelope
from agentic.state import ArtifactRecord, FactRecord, TaskNode
from agentic.verifier import SubtaskVerifier


def test_verifier_rejects_model_only_success_claim():
    task = TaskNode(
        task_id="validate",
        goal="validate",
        success_criteria={
            "required_artifact_types": ["validation_report"],
            "require_hard_pass": True,
        },
    )

    result = SubtaskVerifier().verify(task)
    assert result.passed is False
    assert "MISSING_ARTIFACT:validation_report" in result.failure_codes


def test_verifier_accepts_facts_observations_and_hard_pass_artifact():
    task = TaskNode(
        task_id="validate",
        goal="validate",
        required_facts=("candidate_poi_ids",),
        success_criteria={
            "required_artifact_types": ["validation_report"],
            "require_hard_pass": True,
            "min_successful_observations": 1,
        },
    )
    facts = {
        "fact-1": FactRecord(
            fact_id="fact-1",
            key="candidate_poi_ids",
            value=["poi-1"],
            observation_ref="obs-1",
            goal_version=1,
            plan_version=1,
            source="database",
            confidence=1,
        )
    }
    artifacts = {
        "report-1": ArtifactRecord(
            artifact_id="report-1",
            artifact_type="validation_report",
            payload={"hard_pass": True},
            evidence_refs=["obs-1"],
            goal_version=1,
            plan_version=1,
        )
    }
    observations = [
        ObservationEnvelope(
            ok=True,
            tool="validate_itinerary",
            data={"hard_pass": True},
            source="built_in",
            confidence=1,
            tool_call_id="obs-2",
        )
    ]

    result = SubtaskVerifier().verify(
        task, facts=facts, artifacts=artifacts, observations=observations
    )
    assert result.passed is True
    assert set(result.evidence_refs) >= {"fact-1", "obs-1", "obs-2", "report-1"}


def test_verifier_rejects_validation_artifact_with_hard_failure():
    task = TaskNode(
        task_id="validate",
        goal="validate",
        success_criteria={
            "required_artifact_types": ["validation_report"],
            "require_hard_pass": True,
        },
    )
    artifacts = {
        "report-1": ArtifactRecord(
            artifact_id="report-1",
            artifact_type="validation_report",
            payload={"hard_pass": False},
            goal_version=1,
            plan_version=1,
        )
    }

    result = SubtaskVerifier().verify(task, artifacts=artifacts)
    assert result.passed is False
    assert "HARD_CONSTRAINT_FAILED" in result.failure_codes
