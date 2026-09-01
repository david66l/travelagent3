import json
from types import SimpleNamespace

from scripts.audit_verifier_reason_quality import audit


def _row():
    contract = {
        "target_action": "retry_solve",
        "grounding_phrases": ["宽窄巷子尚未结束", "人民公园重叠45分钟"],
        "prompt_messages": [
            {
                "role": "tool",
                "content": json.dumps(
                    {
                        "policy_state": {
                            "relevant_artifacts": [
                                {
                                    "artifact_type": "validation_report",
                                    "violations": [
                                        {
                                            "message": (
                                                "成都3日行程第1天，宽窄巷子尚未结束就与"
                                                "人民公园重叠45分钟。"
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
    return SimpleNamespace(
        task=SimpleNamespace(task_id="task-1"),
        snapshot=SimpleNamespace(
            hidden_test_facts={"grpo_decision_state": contract}
        ),
    )


def test_offline_audit_rejects_shortcuts_and_accepts_complete_reason():
    rollout = {
        "task_id": "task-1",
        "observed_action": "retry_solve",
        "model_raw_arguments": {
            "reason": (
                "成都3日行程第1天，宽窄巷子尚未结束就与人民公园重叠45分钟。"
                "该问题可在现有约束内通过调整顺序进行一次有界重算。"
            )
        },
        "model_contract_compliant": True,
        "controller_override_attempt": False,
    }

    report = audit([_row()], [rollout])

    assert report["summary"]["strict_full_success_rate"] == 1.0
    assert report["adversarial"]["cases"] == 5
    assert report["adversarial"]["false_pass_rate"] == 0.0
    assert report["adversarial"]["passed"] is True
