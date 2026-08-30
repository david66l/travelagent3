import json

from scripts.compare_curriculum_audits import compare


def _report(checkpoint: str, search: float, clarification: float):
    decisions = []
    for task_id, success, fingerprint in (
        ("search-1", search, "search-state"),
        ("clarify-1", clarification, "clarify-state"),
    ):
        decisions.append(
            {
                "task_id": task_id,
                "initial_state_fingerprint": fingerprint,
                "group_size": 4,
                "success_rate": success,
                "mean_reward": success * 1.9 - 0.96,
            }
        )
    return {
        "checkpoint": checkpoint,
        "corpus_file": "fixed-validation.jsonl",
        "seed": 44,
        "seed_protocol": "sha256-task-sample-v1",
        "family_offset": 2,
        "temperature": 0.8,
        "quantization": "nf4-double-quant",
        "group_size": 4,
        "families": {"search": 1, "clarification": 1},
        "decisions": decisions,
    }


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_promotion_gate_rejects_family_regression(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write(before, _report("sft", 1.0, 1.0))
    _write(after, _report("grpo", 1.0, 0.5))

    result = compare(before, after)

    assert result["promoted"] is False
    assert result["success_rate_delta"] == -0.25
    assert any(
        error.startswith("FAMILY_SUCCESS_REGRESSION:clarification")
        for error in result["gate_errors"]
    )


def test_promotion_gate_accepts_non_regressing_checkpoint(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    _write(before, _report("sft", 0.75, 0.75))
    _write(after, _report("grpo", 1.0, 0.75))

    result = compare(before, after)

    assert result["promoted"] is True
    assert result["success_rate_delta"] == 0.125


def test_promotion_gate_rejects_unknown_or_protected_policy_arguments(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before_report = _report("sft", 0.75, 0.75)
    after_report = _report("recovery-sft", 0.75, 0.75)
    after_report["behavior_gate"] = {
        "unknown_argument_error_rate": 0.125,
        "protected_argument_error_rate": 0.125,
    }
    _write(before, before_report)
    _write(after, after_report)

    result = compare(before, after)

    assert result["promoted"] is False
    assert result["after_behavior_gate"] == after_report["behavior_gate"]
    assert any(error.startswith("UNKNOWN_ARGUMENT_ERROR_RATE") for error in result["gate_errors"])
    assert any(error.startswith("PROTECTED_ARGUMENT_ERROR_RATE") for error in result["gate_errors"])
