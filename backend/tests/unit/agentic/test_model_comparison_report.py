import json

import pytest

from scripts.build_model_comparison_report import build


def _arm(path, checkpoint, success):
    path.mkdir()
    report = {
        "checkpoint": checkpoint,
        "tasks": 1,
        "corpus_file": "validation.jsonl",
        "seed": 44,
        "seed_protocol": "sha256-task-sample-v1",
        "temperature": 0.8,
        "quantization": "nf4-double-quant",
        "group_size": 1,
        "family_offset": 2,
    }
    row = {
        "task_id": "task-1",
        "family": "tradeoff",
        "sample_index": 0,
        "rollout_seed": 123,
        "gate_status": "passed" if success else "task_failed",
        "reward": 0.8 if success else -0.9,
        "reward_config_version": "hierarchical-b0.v1",
        "reward_components": {
            key: float(success)
            for key in (
                "task",
                "constraint",
                "format",
                "tool",
                "grounding",
                "efficiency",
                "quality",
            )
        },
        "termination_reason": "done" if success else "rollout_truncated",
        "actions": [],
    }
    (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (path / "rollouts.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_comparison_requires_same_contract_and_computes_delta(tmp_path):
    base = tmp_path / "base"
    grpo = tmp_path / "grpo"
    _arm(base, "base", False)
    _arm(grpo, "grpo", True)

    result = build([("Base", base), ("SFT+GRPO", grpo)])

    assert result["arms"][0]["success_rate"] == 0
    assert result["arms"][1]["success_rate"] == 1
    assert result["arms"][1]["delta_vs_base"]["success_rate"] == 1


def test_comparison_rejects_mixed_protocol(tmp_path):
    base = tmp_path / "base"
    other = tmp_path / "other"
    _arm(base, "base", False)
    _arm(other, "other", True)
    report_path = other / "report.json"
    report = json.loads(report_path.read_text())
    report["seed"] = 45
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="contract mismatch"):
        build([("Base", base), ("Other", other)])
