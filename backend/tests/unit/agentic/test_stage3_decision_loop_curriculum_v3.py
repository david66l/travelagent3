import importlib.util
import json
from pathlib import Path

from agentic.grpo_training import load_grpo_corpus, preflight_grpo_corpus


SCRIPT = (
    Path(__file__).resolve().parents[4] / "scripts" / "build_stage3_decision_loop_curriculum_v3.py"
)
SPEC = importlib.util.spec_from_file_location("stage3_decision_loop_curriculum_v3", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v3_orthogonalizes_city_template_and_target_position(tmp_path: Path):
    manifest = MODULE.build(
        tmp_path,
        start_index=101000,
        train_count=64,
        validation_count=32,
        test_count=32,
    )

    for split in ("train", "validation", "test"):
        for stratum, facts in manifest["coverage"][split].items():
            assert len(facts["cities"]) == 4
            assert len(facts["template_ids"]) >= 2
            if stratum.startswith("change_arguments/"):
                assert set(facts["target_positions"]) == {"0", "1"}

    rows = load_grpo_corpus(tmp_path / "train.jsonl")
    diagnostic_change = [
        row
        for row in rows
        if row.snapshot.hidden_test_facts["decision_loop_curriculum"]["scenario"]
        == "change_arguments"
        and row.snapshot.hidden_test_facts["decision_loop_curriculum"]["evidence_style"]
        == "diagnostic_evidence"
    ]
    assert {row.task.slots["destination"] for row in diagnostic_change} == {
        "北京",
        "南京",
        "广州",
        "成都",
    }


def test_v3_is_preflight_compatible_and_split_templates_are_disjoint(tmp_path: Path):
    MODULE.build(
        tmp_path,
        start_index=102000,
        train_count=64,
        validation_count=32,
        test_count=32,
    )
    report = preflight_grpo_corpus(
        tmp_path,
        minimum_train_tasks=64,
        require_dependencies=False,
    )
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert report.ready is True
    assert report.errors == []
    assert manifest["template_family_overlap"] == []
