from scripts.generate_h006_local_donor_rollouts import (
    decision_selection_gate,
    normal_selection_gate,
)


def test_decision_gate_requires_raw_semantics_not_only_system_success():
    gates = decision_selection_gate(
        audit_metrics={
            "decision_model_semantic_valid": False,
            "decision_system_step_valid": True,
            "decision_execution_valid": True,
            "decision_no_policy_call_rejections": True,
        },
        policy_errors=[],
        actions=[
            {
                "model_contract_compliant": True,
                "controller_override_attempt": False,
            }
        ],
    )

    assert gates["assembled_system_success"] is True
    assert gates["raw_model_semantic_success"] is False
    assert gates["selection_eligible"] is False


def test_decision_gate_accepts_only_one_clean_sampled_action():
    metrics = {
        "decision_model_semantic_valid": True,
        "decision_system_step_valid": True,
        "decision_execution_valid": True,
        "decision_no_policy_call_rejections": True,
    }
    action = {
        "model_contract_compliant": True,
        "controller_override_attempt": False,
    }

    assert decision_selection_gate(
        audit_metrics=metrics,
        policy_errors=[],
        actions=[action],
    )["selection_eligible"] is True
    assert decision_selection_gate(
        audit_metrics=metrics,
        policy_errors=[],
        actions=[action, action],
    )["selection_eligible"] is False


def test_normal_gate_requires_replay_verifier_and_contract():
    accepted = normal_selection_gate(
        successful=True,
        hard_pass=True,
        replay_errors=[],
        policy_errors=[],
        actions=[
            {
                "model_contract_compliant": True,
                "controller_override_attempt": False,
            }
        ],
    )
    rejected = normal_selection_gate(
        successful=True,
        hard_pass=True,
        replay_errors=["future observation"],
        policy_errors=[],
        actions=[
            {
                "model_contract_compliant": True,
                "controller_override_attempt": False,
            }
        ],
    )

    assert accepted["selection_eligible"] is True
    assert rejected["selection_eligible"] is False
