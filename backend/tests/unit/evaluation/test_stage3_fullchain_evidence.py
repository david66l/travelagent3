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
        "execution_mode": "react",
        "real_agent_runtime": {
            "hard_pass_rate": 1.0,
            "mean_total_tokens": 120,
            "mean_model_calls": 1,
            "mean_tool_calls": 9,
        },
    }
    rl_gain = {
        "tasks": 32,
        "independent_source_clusters": 32,
        "paired_rollouts": 128,
        "baseline_success_rate": 0.86,
        "candidate_success_rate": 0.92,
        "absolute_gain": 0.06,
        "relative_error_reduction": 0.42,
        "exact_mcnemar_two_sided_p": 0.01,
        "source_cluster_randomization_two_sided_p": 0.02,
        "source_cluster_bootstrap_95ci": [0.01, 0.11],
        "gate": {"passed": True},
    }
    downstream_repair = {
        "schema_version": "verifier-repair-downstream-gain.v1",
        "paired_tasks": 32,
        "specialist_route_activations": 32,
        "eligible_specialist_decisions": 32,
        "specialist_activation_rate": 1.0,
        "scope_violations": 0,
        "baseline_hard_pass_rate": 0.75,
        "candidate_hard_pass_rate": 0.84,
        "gate": {"passed": True},
    }

    report = build_report(planning, runtime, rl_gain, downstream_repair)

    assert report["gate"]["passed"] is True
    assert report["planning_axis"]["paired_tasks"] == 30
    assert report["post_training_decision_axis"]["paired_rollouts"] == 128
    assert report["downstream_repair_axis"]["specialist_route_activations"] == 32
    assert "three-arm ranking" in report["methodology"]


def test_two_axis_report_rejects_missing_downstream_repair_evidence():
    planning = {
        "paired_tasks": 30,
        "model": "teacher",
        "pure_agent": {"hard_pass_rate": 0.1, "mean_total_tokens": 1000},
        "verified_planner": {"hard_pass_rate": 1.0, "mean_total_tokens": 100},
        "paired_delta": {"verified_token_reduction_vs_pure_percent": 90.0},
    }
    runtime = {
        "paired_tasks": 30,
        "execution_mode": "react",
        "real_agent_runtime": {
            "hard_pass_rate": 1.0,
            "mean_total_tokens": 120,
            "mean_model_calls": 1,
            "mean_tool_calls": 9,
        },
    }
    rl_gain = {
        "tasks": 16,
        "independent_source_clusters": 16,
        "paired_rollouts": 192,
        "baseline_success_rate": 0.8,
        "candidate_success_rate": 0.85,
        "absolute_gain": 0.05,
        "relative_error_reduction": 0.25,
        "exact_mcnemar_two_sided_p": 0.01,
        "source_cluster_randomization_two_sided_p": 0.02,
        "source_cluster_bootstrap_95ci": [0.01, 0.09],
        "gate": {"passed": True},
    }

    report = build_report(planning, runtime, rl_gain)

    assert report["gate"]["passed"] is False
    assert "DOWNSTREAM_REPAIR_EVIDENCE_MISSING" in report["gate"]["errors"]
