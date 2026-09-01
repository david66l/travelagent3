from copy import deepcopy

from scripts.select_tradeoff_grounding_sft_candidate import (
    candidate_gates,
    paired_retention_gates,
    select_candidate,
)


def _route(
    rows,
    *,
    full,
    action,
    grounding,
    supported=None,
    coverage=None,
    minus_one=0,
):
    return {
        "rows": rows,
        "full_success_count": full,
        "full_success_rate": full / rows,
        "system_full_success_count": full,
        "system_full_success_rate": full / rows,
        "model_contract_compliance_rate": 1.0,
        "controller_override_attempt_rate": 0.0,
        "controller_hydration_exact_rate": 1.0 if supported is not None else None,
        "parse_schema_cardinality_valid_rate": 1.0,
        "action_accuracy": action / rows,
        "reason_grounding_rate": grounding / rows,
        "tradeoff_options_supported_rate": (
            supported / rows if supported is not None else None
        ),
        "tradeoff_required_option_coverage_rate": (
            coverage / rows if coverage is not None else None
        ),
        "reward_minus_one_rate": minus_one / rows,
        "policy_output_error_rate": 0.0,
        "policy_call_rejection_rate": 0.0,
        "execution_invalid_rate": 0.0,
        "mean_reward": 0.0,
    }


def _baseline():
    return {
        "splits": {
            "internal_dev": {
                "parse_schema_cardinality_valid_rate": 1.0,
                "policy_output_error_rate": 0.0,
                "action_accuracy": 35 / 50,
                "reason_grounding_rate": 34 / 50,
                "full_success_count": 13,
                "system_full_success_count": 13,
                "model_contract_compliance_rate": 1.0,
                "controller_override_attempt_rate": 0.0,
                "controller_hydration_exact_rate": 1.0,
                "reward_minus_one_rate": 15 / 50,
                "targets": {
                    "abort": _route(10, full=10, action=10, grounding=10),
                    "propose_tradeoff": _route(
                        30,
                        full=0,
                        action=22,
                        grounding=21,
                        supported=0,
                        coverage=21,
                        minus_one=8,
                    ),
                    "retry_solve": _route(10, full=3, action=3, grounding=3, minus_one=7),
                },
            }
        }
    }


def _passing_candidate():
    return {
        "splits": {
            "internal_dev": {
                "parse_schema_cardinality_valid_rate": 1.0,
                "policy_output_error_rate": 0.0,
                "action_accuracy": 45 / 50,
                "reason_grounding_rate": 45 / 50,
                "full_success_count": 44,
                "system_full_success_count": 44,
                "model_contract_compliance_rate": 1.0,
                "controller_override_attempt_rate": 0.0,
                "controller_hydration_exact_rate": 1.0,
                "reward_minus_one_rate": 2 / 50,
                "targets": {
                    "abort": _route(10, full=10, action=10, grounding=10),
                    "propose_tradeoff": _route(
                        30,
                        full=26,
                        action=27,
                        grounding=27,
                        supported=30,
                        coverage=30,
                        minus_one=2,
                    ),
                    "retry_solve": _route(10, full=8, action=8, grounding=8),
                },
            }
        }
    }


def test_candidate_threshold_boundaries_pass_with_no_retention_regression():
    gates = candidate_gates(_passing_candidate(), baseline=_baseline())

    assert all(item["passed"] for item in gates)


def test_option_support_below_30_of_30_fails_closed():
    candidate = _passing_candidate()
    candidate["splits"]["internal_dev"]["targets"]["propose_tradeoff"][
        "tradeoff_options_supported_rate"
    ] = 29 / 30

    gates = candidate_gates(candidate, baseline=_baseline())

    failed = {item["name"] for item in gates if not item["passed"]}
    assert "tradeoff_options_supported" in failed


def test_controller_override_or_inexact_hydration_cannot_select_candidate():
    candidate = _passing_candidate()
    tradeoff = candidate["splits"]["internal_dev"]["targets"]["propose_tradeoff"]
    tradeoff["controller_override_attempt_rate"] = 1 / 30
    tradeoff["controller_hydration_exact_rate"] = 29 / 30

    gates = candidate_gates(candidate, baseline=_baseline())

    failed = {item["name"] for item in gates if not item["passed"]}
    assert {
        "tradeoff_controller_override_attempts",
        "tradeoff_controller_hydration_exact",
    } <= failed


def test_selector_uses_earliest_passing_checkpoint_and_never_best_average():
    baseline = _baseline()
    epoch1 = _passing_candidate()
    epoch2 = deepcopy(epoch1)
    epoch2["splits"]["internal_dev"]["full_success_count"] = 50

    selected, gates = select_candidate(
        {"baseline": baseline, "epoch1": epoch1, "epoch2": epoch2}
    )

    assert not all(item["passed"] for item in gates["baseline"])
    assert all(item["passed"] for item in gates["epoch1"])
    assert selected == "epoch1"


def test_tradeoff_partial_execution_failures_cannot_pass_full_candidate_gates():
    candidate = _passing_candidate()
    candidate["splits"]["internal_dev"]["targets"]["propose_tradeoff"][
        "execution_invalid_rate"
    ] = 4 / 30

    gates = candidate_gates(candidate, baseline=_baseline())

    failed = {item["name"] for item in gates if not item["passed"]}
    assert {"tradeoff_execution_invalid", "overall_execution_invalid"} <= failed


def test_taskwise_retention_rejects_replacing_old_successes_with_new_ones():
    baseline = [
        {
            "task_id": f"retry-{index}",
            "target_action": "retry_solve",
            "reward": 1.0 if index < 3 else -1.0,
            "checks": {"decision_step_valid": index < 3},
            "model_contract_compliant": True,
            "controller_override_attempt": False,
        }
        for index in range(10)
    ]
    candidate = [
        {
            **row,
            "reward": 1.0 if 3 <= index < 8 else -1.0,
            "checks": {"decision_step_valid": 3 <= index < 8},
        }
        for index, row in enumerate(baseline)
    ]

    gates = paired_retention_gates(baseline, candidate)

    retry_gate = next(
        item for item in gates if item["name"] == "retry_solve_taskwise_full_success_retention"
    )
    assert retry_gate["passed"] is False
    assert retry_gate["observed"] == ["retry-0", "retry-1", "retry-2"]
