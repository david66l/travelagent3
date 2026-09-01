from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_verifier_repair_downstream_report import build_report


def _write_rollouts(
    report_dir: Path,
    *,
    successes: list[bool],
    targets: list[str] | None = None,
    specialist: bool = False,
    protocol_overrides: dict[str, object] | None = None,
) -> None:
    report_dir.mkdir(parents=True)
    targets = targets or ["retry_solve"] * len(successes)
    rows = []
    for index, (success, target) in enumerate(zip(successes, targets, strict=True)):
        rows.append(
            {
                "task_id": f"task-{index}",
                "source_task_id": f"source-{index}",
                "source_snapshot_version": "snapshot-v1",
                "target_action": target,
                "sample_index": 0,
                "rollout_seed": 1000 + index,
                "success": success,
                "repair_review_reached": True,
                "specialist_requested": specialist,
                "specialist_executed": specialist,
                "specialist_fallback": False,
                "scope_violation": False,
                "policy_errors": [],
                "latency_ms": 10.0,
                "completion_tokens": 20,
            }
        )
    (report_dir / "rollouts.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    protocol = {
        "schema_version": "verifier-repair-full-loop-audit.v1",
        "topology": (
            "sft-generalist-plus-verifier-repair-specialist"
            if specialist
            else "sft-generalist"
        ),
        "execution_mode": "react",
        "structured_decoding_mode": "native",
        "generalist": "sft",
        "generalist_adapter_sha256": "sft-sha",
        "specialist": "rl" if specialist else None,
        "specialist_adapter_sha256": "rl-sha" if specialist else None,
        "corpus_sha256": "corpus-sha",
        "seed": 42,
        "temperature": 1.0,
        "group_size": 1,
        "max_new_tokens": 192,
        "max_tool_calling_iterations": 16,
        "load_in_4bit": True,
        "tasks_per_target": len(successes),
        "target_offset": 0,
        "tasks": len(successes),
        "rollouts": len(successes),
    }
    protocol.update(protocol_overrides or {})
    (report_dir / "report.json").write_text(
        json.dumps(protocol),
        encoding="utf-8",
    )


def test_downstream_report_passes_only_with_source_level_gain_and_real_route(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    targets = ["retry_solve", "propose_tradeoff", "abort"] * 3
    _write_rollouts(baseline_dir, successes=[False] * 9, targets=targets)
    _write_rollouts(
        candidate_dir,
        successes=[True] * 9,
        targets=targets,
        specialist=True,
    )

    report = build_report(
        baseline_dir,
        candidate_dir,
        minimum_pairs=9,
        minimum_pairs_per_target=3,
        minimum_source_clusters=9,
        minimum_gain=0.03,
        minimum_candidate_success=0.80,
        bootstrap_samples=1000,
    )

    assert report["gate"]["passed"] is True
    assert report["independent_source_clusters"] == 9
    assert report["specialist_activation_rate"] == 1.0
    assert report["source_cluster_bootstrap_95ci"][0] > 0


def test_downstream_report_rejects_per_target_regression(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    targets = ["retry_solve"] * 4 + ["abort"] * 4
    _write_rollouts(
        baseline_dir,
        successes=[False] * 4 + [True] * 4,
        targets=targets,
    )
    _write_rollouts(
        candidate_dir,
        successes=[True] * 4 + [False] * 4,
        targets=targets,
        specialist=True,
    )

    report = build_report(
        baseline_dir,
        candidate_dir,
        minimum_pairs=8,
        minimum_pairs_per_target=0,
        minimum_source_clusters=8,
        minimum_gain=0.0,
        minimum_candidate_success=0.0,
        maximum_p_value=1.0,
        bootstrap_samples=1000,
    )

    assert report["gate"]["passed"] is False
    assert "TARGET_ACTION_REGRESSION" in report["gate"]["errors"]


def test_downstream_report_rejects_source_provenance_mismatch(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_rollouts(baseline_dir, successes=[False])
    _write_rollouts(candidate_dir, successes=[True], specialist=True)
    candidate_path = candidate_dir / "rollouts.jsonl"
    row = json.loads(candidate_path.read_text(encoding="utf-8"))
    row["source_snapshot_version"] = "different-snapshot"
    candidate_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source metadata differs"):
        build_report(
            baseline_dir,
            candidate_dir,
            minimum_pairs=1,
            minimum_pairs_per_target=0,
            minimum_source_clusters=1,
        )


def test_downstream_report_rejects_mixed_evaluation_protocols(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    _write_rollouts(baseline_dir, successes=[False])
    _write_rollouts(
        candidate_dir,
        successes=[True],
        specialist=True,
        protocol_overrides={"temperature": 1.2},
    )

    with pytest.raises(ValueError, match="protocols differ"):
        build_report(
            baseline_dir,
            candidate_dir,
            minimum_pairs=1,
            minimum_pairs_per_target=0,
            minimum_source_clusters=1,
        )


def test_downstream_report_requires_all_targets_and_minimum_samples(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    candidate_dir = tmp_path / "candidate"
    targets = ["retry_solve", "propose_tradeoff", "abort"]
    _write_rollouts(baseline_dir, successes=[False] * 3, targets=targets)
    _write_rollouts(
        candidate_dir,
        successes=[True] * 3,
        targets=targets,
        specialist=True,
    )

    report = build_report(
        baseline_dir,
        candidate_dir,
        minimum_pairs=3,
        minimum_pairs_per_target=2,
        minimum_source_clusters=3,
        minimum_gain=0.0,
        minimum_candidate_success=0.0,
        maximum_p_value=1.0,
        bootstrap_samples=1000,
    )

    assert report["gate"]["passed"] is False
    assert "TARGET_ACTION_SAMPLE_TOO_SMALL" in report["gate"]["errors"]
