from copy import deepcopy

import pytest

from scripts.audit_h006_rollout_quality import audit_records, target_quality_findings


def _record(reason="文化馆入场比票面固定时刻早27分钟。"):
    raw = {"reason": reason}
    return {
        "source_state_id": "source-1",
        "target_action": "retry_solve",
        "gates": dict.fromkeys(
            (
                "selection_eligible",
                "raw_model_semantic_success",
                "assembled_system_success",
                "deterministic_verifier_success",
                "model_contract_compliant",
                "no_controller_override",
                "no_policy_rejection",
                "exactly_one_sampled_decision",
            ),
            True,
        ),
        "actions": [
            {
                "step_index": 1,
                "action": "retry_solve",
                "model_raw_arguments": raw.copy(),
                "inference_metrics": {"model": "frozen-h001", "completion_tokens": 30},
            }
        ],
        "candidate": {
            "rollout": {
                "episode": {
                    "steps": [
                        {
                            "step_index": 1,
                            "action": {
                                "action": "retry_solve",
                                "decision_source": "policy",
                                "model_arguments": raw.copy(),
                            },
                        }
                    ]
                }
            }
        },
    }


def test_accepts_grounded_detail_with_sampled_provenance():
    assert target_quality_findings(_record()) == []


@pytest.mark.parametrize(
    "reason,code",
    [
        (
            "Grounded failure detail; the system adds the fixed bounded-retry rationale",
            "REASON_SCHEMA_DESCRIPTION_ECHO",
        ),
        (
            "现有校验事实无法修复，应基于现有约束重新调度，文化馆入场早27分钟。",
            "RETRY_REASON_AMBIGUOUS_OR_CONTRADICTORY",
        ),
        (
            "当前已锁定约束并需用户调整优先级，文化馆入场早27分钟。",
            "RETRY_REASON_AMBIGUOUS_OR_CONTRADICTORY",
        ),
    ],
)
def test_quarantines_bad_reason_without_mutating_record(reason, code):
    record = _record(reason)
    original = deepcopy(record)
    assert code in target_quality_findings(record)
    assert record == original


def test_fail_closed_on_uninstrumented_or_mismatching_target():
    record = _record()
    record["actions"][0]["inference_metrics"] = None
    record["actions"][0]["model_raw_arguments"] = {"reason": "a rewritten answer"}
    findings = target_quality_findings(record)
    assert "SAMPLED_INFERENCE_PROVENANCE_MISSING" in findings
    assert "SAMPLED_RAW_ARGUMENTS_MISMATCH" in findings


def test_positive_summary_cannot_override_failed_component_gate():
    record = _record()
    record["gates"]["raw_model_semantic_success"] = False
    assert "UPSTREAM_COMPONENT_GATE_REJECTED" in target_quality_findings(record)


def test_report_counts_sources_not_four_samples_as_independent():
    accepted = _record()
    rejected = _record("Grounded failure detail")
    rejected["source_state_id"] = "source-2"
    report = audit_records([accepted, deepcopy(accepted), rejected])
    assert report["upstream_successful_sources"] == 2
    assert report["post_audit_successful_sources"] == 1
    assert report["post_audit_eligible_rollouts"] == 2
    assert report["post_audit_sources_per_target"] == {"retry_solve": 1}
