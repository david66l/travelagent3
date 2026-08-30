import json

from scripts.build_stage3_multiturn_rl_corpus import build as build_source
from scripts.build_stage3_routed_rl_curriculum import build


def test_stage3_routing_expands_only_audited_learnable_strata(tmp_path):
    source = tmp_path / "source"
    build_source(
        source,
        start_index=80000,
        train_count=800,
        validation_count=100,
        test_count=8,
    )
    train_rows = [
        json.loads(line)
        for line in (source / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    audited = train_rows[:8]
    decisions = []
    for index, row in enumerate(audited):
        decisions.append(
            {
                "task_id": row["task"]["task_id"],
                "route": "grpo_update" if index in {0, 1, 3} else "evaluation",
            }
        )
    report = tmp_path / "audit.json"
    report.write_text(json.dumps({"decisions": decisions}), encoding="utf-8")

    output = tmp_path / "routed"
    manifest = build(source, report, output)

    assert manifest["audited_update_tasks"] == 3
    assert manifest["counts"]["train"] == 600
    assert manifest["counts"]["validation"] == 75
    assert len(manifest["learnable_strata"]) == 3
    assert manifest["preflight"]["ready"] is True
