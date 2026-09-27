from scripts.compare_native_react_hard_arms import build_comparison


def _report(model: str, outcomes: list[bool]) -> dict:
    case_ids = [f"case-{index}" for index in range(len(outcomes))]
    records = []
    for index, (case_id, passed) in enumerate(zip(case_ids, outcomes, strict=True)):
        records.append(
            {
                "case_id": case_id,
                "passed": passed,
                "failures": [] if passed else ["FAIL"],
                "total_tokens": 100 + index,
                "policy_calls": 1,
                "tool_calls": 2,
                "latency_ms": 20,
                "benchmark_metadata": {"family": f"family-{index % 2}"},
                "actions": [
                    {
                        "source": "policy",
                        "action": "get_poi_detail",
                    }
                ],
            }
        )
    return {
        "benchmark": {
            "split": "dev",
            "frozen_file_sha256": "file",
            "dataset_sha256": "dataset",
            "selected_case_ids": case_ids,
        },
        "runtime": {
            "policy_model": model,
            "policy_protocol": "native_tool",
            "base_seed": 7,
            "temperature": 0.0,
        },
        "summary": {
            "cluster_bootstrap_95ci": [0.0, 1.0],
            "verifier_final_hard_pass_rate": 1.0,
            "bounded_recovery_rate": 1.0,
            "mean_tokens": 101.5,
            "mean_policy_calls": 1.0,
            "mean_tool_calls": 2.0,
            "mean_latency_ms": 20.0,
        },
        "records": records,
    }


def test_paired_comparison_reports_case_migrations() -> None:
    report = build_comparison(
        {
            "base": _report("base", [True, False, True, False]),
            "candidate": _report("candidate", [True, True, False, True]),
        },
        baseline="base",
        samples=100,
        seed=7,
    )

    paired = report["paired_vs_baseline"]["candidate"]
    assert paired["candidate_only_success"] == 2
    assert paired["baseline_only_success"] == 1
    assert paired["absolute_delta_pp"] == 25.0
    assert report["arms"]["candidate"]["policy_calls"] == 4
