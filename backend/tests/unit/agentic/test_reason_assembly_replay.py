from __future__ import annotations

import json

from scripts.replay_reason_assembly import replay_row


def _contract(target_action: str) -> dict[str, object]:
    options = ["删除明孝陵", "把明孝陵调整到第2天"]
    return {
        "target_action": target_action,
        "grounding_phrases": ["第1天", "重叠45分钟"],
        "expected_arguments": {},
        "controller_arguments": {},
        "supervised_options": options if target_action == "propose_tradeoff" else [],
        "prompt_messages": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "policy_state": {
                            "relevant_artifacts": [
                                {
                                    "artifact_type": "validation_report",
                                    "violations": [
                                        {
                                            "message": (
                                                "南京4日行程第1天，中山陵尚未结束就与"
                                                "明孝陵重叠45分钟。"
                                            )
                                        }
                                    ],
                                }
                            ]
                        }
                    },
                    ensure_ascii=False,
                ),
            }
        ],
    }


def _historical_row(reason: str) -> dict[str, object]:
    options = ["删除明孝陵", "把明孝陵调整到第2天"]
    return {
        "task_id": "task-1",
        "source_task_id": "source-1",
        "sample_index": 0,
        "rollout_seed": 7,
        "target_action": "propose_tradeoff",
        "observed_action": "propose_tradeoff",
        "model_raw_arguments": {"reason": reason},
        "observed_arguments": {"reason": reason, "options": options},
        "model_contract_compliant": True,
        "controller_override_attempt": False,
        "controller_hydration_exact": True,
        "reward": 0.8,
        "checks": {
            "single_policy_call": True,
            "no_policy_call_rejections": True,
            "execution_valid": True,
            "decision_step_valid": False,
        },
        "generation_audit": {"raw_output_sha256": "historical-output-hash"},
    }


def test_replay_recovers_connector_without_changing_decision_or_evidence() -> None:
    reason = "南京4日行程第1天，中山陵尚未结束就与明孝陵重叠45分钟。"

    replayed = replay_row(_historical_row(reason), _contract("propose_tradeoff"))

    assert replayed["raw_model_legacy_success"] is False
    assert replayed["raw_model_contract_success"] is True
    assert replayed["assembled_system_success"] is True
    assert replayed["assembly_succeeded"] is True
    assert replayed["invariants"]["action_unchanged"] is True
    assert replayed["invariants"]["non_reason_arguments_unchanged"] is True
    assert replayed["invariants"]["evidence_metrics_unchanged"] is True
    assert replayed["invariants"]["historical_raw_output_unchanged"] is True
    assert replayed["invariants"]["source_row_unchanged"] is True
    assert replayed["invariants"]["tool_trace_projection_unchanged"] is True
    assert replayed["invariants"]["tool_execution_performed"] is False
    assert (
        replayed["invariants"]["full_tool_trace_status"]
        == "unverifiable_not_present_in_source_artifact"
    )
    assert replayed["historical_raw_output_sha256"] == "historical-output-hash"


def test_replay_rejects_private_internal_wording() -> None:
    reason = "内部动作要求第1天重叠45分钟后让用户选择。"

    replayed = replay_row(_historical_row(reason), _contract("propose_tradeoff"))

    assert replayed["assembly_succeeded"] is False
    assert replayed["assembly_error"] == "REPAIR_REASON_PRIVATE_CONTENT"
    assert replayed["assembled_system_success"] is False
