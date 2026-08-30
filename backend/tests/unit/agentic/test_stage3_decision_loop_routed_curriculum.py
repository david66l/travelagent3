import importlib.util
import json
from pathlib import Path

from agentic.grpo_training import load_grpo_corpus


SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_script("build_stage3_decision_loop_curriculum")
ROUTER = _load_script("build_stage3_decision_loop_routed_curriculum")


def test_routes_only_audited_learnable_semantic_stratum(tmp_path: Path):
    source = tmp_path / "source"
    BUILDER.build(
        source,
        start_index=74000,
        train_count=16,
        validation_count=8,
        test_count=8,
    )
    source_rows = load_grpo_corpus(source / "train.jsonl")
    seed = next(
        row
        for row in source_rows
        if row.snapshot.hidden_test_facts["decision_loop_curriculum"]
        == {
            **row.snapshot.hidden_test_facts["decision_loop_curriculum"],
            "scenario": "change_arguments",
            "evidence_style": "diagnostic_evidence",
        }
    )
    report_path = tmp_path / "audit.json"
    report_path.write_text(
        json.dumps(
            {
                "decisions": [
                    {
                        "task_id": seed.task.task_id,
                        "route": "grpo_update",
                        "zero_variance": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "routed"
    manifest = ROUTER.build(source, report_path, output, minimum_train_tasks=1)

    assert manifest["learnable_strata"] == [
        {
            "scenario": "change_arguments",
            "evidence_style": "diagnostic_evidence",
        }
    ]
    assert manifest["counts"] == {"train": 4, "validation": 2}
    for row in load_grpo_corpus(output / "train.jsonl"):
        metadata = row.snapshot.hidden_test_facts["decision_loop_curriculum"]
        assert metadata["scenario"] == "change_arguments"
        assert metadata["evidence_style"] == "diagnostic_evidence"
