import json

from scripts.build_stage3_rl_gain_report import _variant, build_report


def _write_report(path, outcomes, seed=7):
    path.mkdir()
    rows = []
    for index, success in enumerate(outcomes):
        rows.append(
            {
                "task_id": f"task-{index // 4:03d}-cross-tool-{index // 4:03d}",
                "sample_index": index % 4,
                "rollout_seed": seed * 10000 + index,
                "gate_status": "passed" if success else "failed",
                "audit_metrics": {"hard_pass": success},
            }
        )
    (path / "rollouts.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


def test_stage3_rl_gain_report_requires_real_paired_gain(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_report(baseline, [False] * 20 + [True] * 20)
    _write_report(candidate, [True] * 16 + [False] * 4 + [True] * 20)

    report = build_report(
        [baseline],
        [candidate],
        minimum_pairs=40,
        minimum_gain=0.20,
        minimum_candidate_success=0.80,
        maximum_p_value=0.05,
        bootstrap_samples=1000,
    )

    assert report["baseline_success_rate"] == 0.5
    assert report["candidate_success_rate"] == 0.9
    assert report["absolute_gain"] == 0.4
    assert report["paired_outcomes"]["candidate_only_success"] == 16
    assert report["paired_outcomes"]["baseline_only_success"] == 0
    assert report["gate"]["passed"] is True


def test_stage3_rl_gain_report_identifies_decision_loop_strata():
    assert (
        _variant("task-decision-loop-change-arguments-diagnostic-00578")
        == "change_arguments/diagnostic_evidence"
    )
    assert (
        _variant("task-decision-loop-retry-same-explicit-00577")
        == "retry_same_arguments/explicit_instruction"
    )
    assert (
        _variant(
            "opaque-task-id",
            {
                "decision_loop": {
                    "scenario": "change_arguments",
                    "evidence_style": "diagnostic_evidence",
                    "target_position": 1,
                }
            },
        )
        == "change_arguments/diagnostic_evidence/position-1"
    )
