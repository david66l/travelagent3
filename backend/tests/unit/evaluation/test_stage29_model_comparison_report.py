from scripts.build_stage29_model_comparison_report import (
    bootstrap_rate_ci,
    exact_mcnemar_p,
    route_to_teacher,
)


def test_policy_visible_router_uses_capability_and_recovery_state():
    assert route_to_teacher({"capability": {"status": "infeasible"}}) is True
    assert route_to_teacher({"capability": {"status": "unsafe"}}) is True
    assert (
        route_to_teacher(
            {
                "capability": {"status": "missing_tool"},
                "failure_summary": [{"retryable": True, "retry_budget_remaining": 1}],
            }
        )
        is False
    )
    assert (
        route_to_teacher(
            {
                "capability": {"status": "missing_tool"},
                "failure_summary": [{"retryable": False, "retry_budget_remaining": 0}],
            }
        )
        is True
    )


def test_bootstrap_and_mcnemar_are_deterministic():
    values = [True, True, False, True]

    assert bootstrap_rate_ci(values, samples=200) == bootstrap_rate_ci(values, samples=200)
    result = exact_mcnemar_p([True, True, True, False], [False, True, False, False])
    assert result["candidate_only_success"] == 2
    assert result["baseline_only_success"] == 0
    assert result["exact_two_sided_p"] == 0.5
