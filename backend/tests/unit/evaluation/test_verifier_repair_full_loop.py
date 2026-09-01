from scripts.evaluate_verifier_repair_full_loop import score_downstream_rollout


def test_retry_success_requires_fresh_solve_and_validation_after_retry():
    success, checks = score_downstream_rollout(
        target_action="retry_solve",
        episode_actions=[
            "solve_itinerary",
            "validate_itinerary",
            "retry_solve",
            "solve_itinerary",
            "validate_itinerary",
        ],
        gate_status="passed",
        audit_metrics={"hard_pass": True},
    )
    missing_downstream, _ = score_downstream_rollout(
        target_action="retry_solve",
        episode_actions=["solve_itinerary", "validate_itinerary", "retry_solve"],
        gate_status="passed",
        audit_metrics={"hard_pass": True},
    )

    assert success is True
    assert all(checks.values())
    assert missing_downstream is False


def test_tradeoff_and_abort_use_termination_contract_instead_of_hard_pass():
    tradeoff, _ = score_downstream_rollout(
        target_action="propose_tradeoff",
        episode_actions=["validate_itinerary", "propose_tradeoff"],
        gate_status="passed",
        audit_metrics={"hard_pass": False, "needs_user_action_mismatch": False},
    )
    abort, _ = score_downstream_rollout(
        target_action="abort",
        episode_actions=["validate_itinerary", "abort"],
        gate_status="passed",
        audit_metrics={
            "hard_pass": False,
            "termination_action_mismatch": False,
            "capability_termination_mismatch": False,
        },
    )

    assert tradeoff is True
    assert abort is True


def test_success_rejects_target_that_only_appears_after_a_wrong_repair_action():
    success, checks = score_downstream_rollout(
        target_action="retry_solve",
        episode_actions=[
            "validate_itinerary",
            "search",
            "retry_solve",
            "solve_itinerary",
            "validate_itinerary",
        ],
        gate_status="passed",
        audit_metrics={"hard_pass": True},
    )

    assert success is False
    assert checks["first_repair_action_matches_target"] is False
