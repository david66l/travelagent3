import importlib.util
import json
from pathlib import Path

import pytest
from agentic.grpo_training import load_grpo_corpus


SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_script("build_stage3_decision_loop_curriculum_v3")
ROUTER = _load_script("build_stage3_decision_loop_state_routed_curriculum")


def _decision(task_id: str, *, selected: bool) -> dict:
    return {
        "task_id": task_id,
        "route": "grpo_update" if selected else "evaluation",
        "eligible_for_update": selected,
        "zero_variance": not selected,
        "success_rate": 0.5 if selected else 1.0,
    }


def test_routes_exact_audited_states_without_semantic_stratum_expansion(tmp_path: Path):
    source = tmp_path / "source"
    BUILDER.build(source, start_index=96000, train_count=32, validation_count=32, test_count=32)
    source_rows = load_grpo_corpus(source / "train.jsonl")
    selected_ids = {row.task.task_id for row in source_rows[:5]}
    report_path = tmp_path / "audit.json"
    report_path.write_text(
        json.dumps(
            {
                "decisions": [
                    _decision(row.task.task_id, selected=row.task.task_id in selected_ids)
                    for row in source_rows
                ]
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "routed"
    manifest = ROUTER.build(source, report_path, output, minimum_train_tasks=5)

    routed_ids = {row.task.task_id for row in load_grpo_corpus(output / "train.jsonl")}
    assert routed_ids == selected_ids
    assert manifest["selection_unit"] == "exact audited environment state"
    assert manifest["counts"] == {"train": 5, "validation": 32}
    assert sum(manifest["coverage"]["train"]["target_position"].values()) == 5


def test_rejects_too_few_learnable_exact_states(tmp_path: Path):
    source = tmp_path / "source"
    BUILDER.build(source, start_index=97000, train_count=32, validation_count=32, test_count=32)
    source_rows = load_grpo_corpus(source / "train.jsonl")
    report_path = tmp_path / "audit.json"
    report_path.write_text(
        json.dumps({"decisions": [_decision(source_rows[0].task.task_id, selected=True)]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact-state routed train split is too small"):
        ROUTER.build(source, report_path, tmp_path / "routed", minimum_train_tasks=2)


def test_combines_independent_audits_and_deduplicates_learnable_states(tmp_path: Path):
    source = tmp_path / "source"
    BUILDER.build(source, start_index=98000, train_count=32, validation_count=32, test_count=32)
    source_rows = load_grpo_corpus(source / "train.jsonl")
    first, second = source_rows[:2]
    report_a = tmp_path / "audit-a.json"
    report_b = tmp_path / "audit-b.json"
    report_a.write_text(
        json.dumps({"decisions": [_decision(first.task.task_id, selected=True)]}),
        encoding="utf-8",
    )
    report_b.write_text(
        json.dumps(
            {
                "decisions": [
                    _decision(first.task.task_id, selected=True),
                    _decision(second.task.task_id, selected=True),
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = ROUTER.build(
        source,
        [report_a, report_b],
        tmp_path / "routed",
        minimum_train_tasks=2,
    )

    assert manifest["counts"]["train"] == 2
    assert manifest["audited_decisions"] == 3
    assert manifest["audited_unique_tasks"] == 2
    assert len(manifest["audit_reports"]) == 2
