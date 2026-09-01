import json

import pytest

from scripts.build_stage3_rl_gain_report import _success, _variant, build_report


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
    _write_report(baseline, [False] * 48 + [True] * 16)
    _write_report(candidate, [True] * 40 + [False] * 8 + [True] * 16)

    report = build_report(
        [baseline],
        [candidate],
        minimum_pairs=64,
        minimum_gain=0.20,
        minimum_candidate_success=0.80,
        maximum_p_value=0.05,
        bootstrap_samples=1000,
    )

    assert report["baseline_success_rate"] == 0.25
    assert report["candidate_success_rate"] == 0.875
    assert report["absolute_gain"] == 0.625
    assert report["paired_outcomes"]["candidate_only_success"] == 40
    assert report["paired_outcomes"]["baseline_only_success"] == 0
    assert report["independent_source_clusters"] == 16
    assert report["source_cluster_randomization_two_sided_p"] < 0.05
    assert report["gate"]["passed"] is True


def test_stage3_rl_gain_report_clusters_synthetic_variants_by_source_state(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_report(baseline, [False] * 8)
    _write_report(candidate, [True] * 8)
    for report_dir in (baseline, candidate):
        rows = [
            json.loads(line)
            for line in (report_dir / "rollouts.jsonl").read_text().splitlines()
        ]
        for row in rows:
            row["verifier_repair"] = {
                "source_task_id": "shared-source-task",
                "source_snapshot_version": "snapshot-v1",
            }
        (report_dir / "rollouts.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )

    report = build_report(
        [baseline],
        [candidate],
        minimum_pairs=8,
        minimum_gain=0.1,
        minimum_candidate_success=0.5,
        maximum_p_value=0.05,
        bootstrap_samples=100,
    )

    assert report["tasks"] == 2
    assert report["independent_source_clusters"] == 1
    assert report["exact_mcnemar_two_sided_p"] < 0.05
    assert report["source_cluster_randomization_two_sided_p"] == 1.0
    assert "SOURCE_CLUSTER_SIGNIFICANCE_NOT_REACHED" in report["gate"]["errors"]
    assert "INSUFFICIENT_INDEPENDENT_SOURCE_CLUSTERS" in report["gate"]["errors"]


def test_stage3_rl_gain_report_rejects_mismatched_source_metadata(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_report(baseline, [False] * 4)
    _write_report(candidate, [True] * 4)
    for report_dir, source_task_id in (
        (baseline, "source-a"),
        (candidate, "source-b"),
    ):
        rows = [
            json.loads(line)
            for line in (report_dir / "rollouts.jsonl").read_text().splitlines()
        ]
        for row in rows:
            row["verifier_repair"] = {
                "source_task_id": source_task_id,
                "source_snapshot_version": "snapshot-v1",
            }
        (report_dir / "rollouts.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )

    with pytest.raises(ValueError, match="source metadata differs"):
        build_report(
            [baseline],
            [candidate],
            minimum_pairs=4,
            bootstrap_samples=100,
        )


def test_stage3_rl_gain_report_identifies_decision_loop_strata():
    assert (
        _variant(
            "opaque-verifier-task",
            {"verifier_repair": {"target_action": "retry_solve"}},
        )
        == "verifier_repair/retry_solve"
    )
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


def test_decision_state_success_uses_the_declared_decision_gate():
    assert _success(
        {
            "gate_status": "passed",
            "audit_metrics": {
                "decision_state_training": True,
                "decision_step_valid": True,
                "hard_pass": False,
            },
        }
    )
