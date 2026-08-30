from scripts.build_stage3_fullchain_evidence import build_report


def test_two_axis_report_keeps_planning_and_rl_denominators_separate():
    planning = {
        "paired_tasks": 30,
        "model": "teacher",
        "pure_agent": {"hard_pass_rate": 0.1, "mean_total_tokens": 1000},
        "verified_planner": {"hard_pass_rate": 1.0, "mean_total_tokens": 100},
        "paired_delta": {"verified_token_reduction_vs_pure_percent": 90.0},
    }
    runtime = {
        "paired_tasks": 30,
        "execution_mode": "controller_first",
        "real_agent_runtime": {
            "hard_pass_rate": 1.0,
            "mean_total_tokens": 120,
            "mean_model_calls": 1,
            "mean_tool_calls": 9,
        },
    }
    rl_gain = {
        "tasks": 32,
        "paired_rollouts": 128,
        "baseline_success_rate": 0.86,
        "candidate_success_rate": 0.92,
        "absolute_gain": 0.06,
        "relative_error_reduction": 0.42,
        "exact_mcnemar_two_sided_p": 0.01,
        "task_cluster_bootstrap_95ci": [0.01, 0.11],
        "gate": {"passed": True},
    }

    report = build_report(planning, runtime, rl_gain)

    assert report["gate"]["passed"] is True
    assert report["planning_axis"]["paired_tasks"] == 30
    assert report["post_training_recovery_axis"]["paired_rollouts"] == 128
    assert "three-arm ranking" in report["methodology"]
