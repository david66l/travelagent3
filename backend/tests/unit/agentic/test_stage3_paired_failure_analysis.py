import json
from pathlib import Path

from scripts.analyze_stage3_paired_failures import build_report


def _row(task_id: str, error_code: str) -> dict:
    return {
        "task_id": task_id,
        "sample_index": 0,
        "rollout_seed": 7,
        "gate_status": "task_failed",
        "termination_reason": "rollout_truncated",
        "audit_metrics": {"hard_pass": False, "duplicate_calls": 0},
        "policy_errors": (
            [{"code": error_code}] if error_code == "POLICY_ARGUMENT_INVALID" else []
        ),
        "actions": (
            [{"error_code": error_code}] if error_code != "POLICY_ARGUMENT_INVALID" else []
        ),
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def _success(task_id: str) -> dict:
    row = _row(task_id, "")
    row["gate_status"] = "passed"
    row["audit_metrics"]["hard_pass"] = True
    row["actions"] = []
    return row


def test_report_separates_harness_and_model_failures(tmp_path: Path):
    rows = [
        _row("snapshot", "SNAPSHOT_ARGUMENT_MISMATCH"),
        _row("schema", "POLICY_ARGUMENT_INVALID"),
    ]
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(baseline, rows)
    _write(candidate, rows)

    report = build_report(
        baseline,
        candidate,
        snapshot_mismatch_owner="evaluation_harness_fault",
    )

    assert report["shared_failure_count"] == 2
    assert report["root_cause_counts"] == {
        "policy_argument_schema_violation": 1,
        "snapshot_argument_contract_mismatch": 1,
    }
    assert report["fault_owner_counts"] == {
        "evaluation_harness_fault": 1,
        "model_fault": 1,
    }
    actions = report["recommended_actions"]["by_root_cause"]
    assert "constrained decoding" in actions["policy_argument_schema_violation"]
    assert "repair SFT" in actions["snapshot_argument_contract_mismatch"]


def test_report_attributes_candidate_only_improvement_to_baseline_failure(tmp_path: Path):
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write(baseline, [_row("fixed", "POLICY_ARGUMENT_INVALID")])
    _write(candidate, [_success("fixed")])

    report = build_report(baseline, candidate)

    assert report["candidate_only_improvement_count"] == 1
    assert report["candidate_only_improvement_causes"] == {"policy_argument_schema_violation": 1}
    assert report["baseline_only_regression_count"] == 0
