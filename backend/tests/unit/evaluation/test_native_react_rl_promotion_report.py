import json

import pytest

from scripts.build_native_react_rl_promotion_report import _paired


def _write_rollouts(root, name, outcomes):
    path = root / name
    path.mkdir()
    rows = []
    for task_id, samples in outcomes.items():
        rows.extend(
            {
                "task_id": task_id,
                "sample_index": sample_index,
                "gate_status": "passed" if passed else "task_failed",
            }
            for sample_index, passed in enumerate(samples)
        )
    (path / "rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_paired_reports_independent_task_statistics(tmp_path):
    before = {
        "task-1": [False, False, True, True, True, True, True, True],
        "task-2": [False, False, True, True, True, True, True, True],
        "task-3": [False, True, True, True, True, True, True, True],
        "task-4": [False, True, True, True, True, True, True, True],
        "task-5": [True] * 8,
        "task-6": [True] * 8,
    }
    after = {task_id: [True] * 8 for task_id in before}
    _write_rollouts(tmp_path, "before", before)
    _write_rollouts(tmp_path, "after", after)

    result = _paired(tmp_path, "before", "after")

    assert result["paired_rollouts"] == 48
    assert result["paired_improvements"] == 6
    assert result["rollout_level_exact_mcnemar_p"] == pytest.approx(0.03125)
    assert result["independent_tasks"] == 6
    assert result["task_level_improvements"] == 4
    assert result["task_level_regressions"] == 0
    assert result["task_level_ties"] == 2
    assert result["task_level_exact_sign_test_p"] == pytest.approx(0.125)
    assert result["task_level_significant_at_0_05"] is False


def test_paired_reports_reject_missing_samples(tmp_path):
    _write_rollouts(tmp_path, "before", {"task-1": [True, False]})
    _write_rollouts(tmp_path, "after", {"task-1": [True]})

    with pytest.raises(ValueError, match="paired rollout keys differ"):
        _paired(tmp_path, "before", "after")
