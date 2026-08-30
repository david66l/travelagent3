import json

import pytest

from scripts.build_stage3_multiturn_rl_corpus import build as build_source
from scripts.build_stage3_variance_grpo_curriculum import build


def test_variance_curriculum_keeps_only_exact_mixed_outcome_tasks(tmp_path):
    source = tmp_path / "source"
    build_source(source, start_index=76000, train_count=8, validation_count=4, test_count=4)
    rows = [
        json.loads(line)
        for line in (source / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    decisions = []
    rates = [0.0, 0.125, 0.75, 0.875, 1.0, 0.5, 0.95, 0.05]
    for row, rate in zip(rows, rates, strict=True):
        decisions.append(
            {
                "task_id": row["task"]["task_id"],
                "success_rate": rate,
                "zero_variance": rate in {0.0, 1.0},
                "route": "grpo_update" if 0.25 <= rate <= 0.8 else "evaluation",
            }
        )
    report = tmp_path / "audit.json"
    report.write_text(json.dumps({"decisions": decisions}), encoding="utf-8")

    manifest = build(source, report, tmp_path / "output")

    assert manifest["counts"] == {"train": 4, "validation": 4}
    assert [item["success_rate"] for item in manifest["selected_tasks"]] == [
        0.125,
        0.75,
        0.875,
        0.5,
    ]


def test_variance_curriculum_rejects_invalid_success_band(tmp_path):
    with pytest.raises(ValueError, match="success-rate bounds"):
        build(tmp_path, tmp_path / "missing.json", tmp_path / "out", minimum_success_rate=0.9)


def test_variance_curriculum_can_expand_one_audited_seed_to_its_stratum(tmp_path):
    source = tmp_path / "source"
    build_source(source, start_index=77000, train_count=8, validation_count=4, test_count=4)
    rows = [
        json.loads(line)
        for line in (source / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    report = tmp_path / "audit.json"
    report.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "task_id": rows[0]["task"]["task_id"],
                        "success_rate": 0.125,
                        "zero_variance": False,
                        "route": "sft_repair",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = build(
        source,
        report,
        tmp_path / "output",
        minimum_success_rate=0.1,
        maximum_success_rate=0.2,
        expand_to_strata=True,
    )

    assert manifest["expanded_to_strata"] is True
    assert manifest["counts"] == {"train": 2, "validation": 4}
