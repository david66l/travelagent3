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


SOURCE_BUILDER = _load_script("build_stage3_decision_loop_curriculum_v3")
ROUTED_BUILDER = _load_script("build_routed_grpo_curriculum")


def test_only_audited_evaluation_tasks_can_be_anti_regression_anchors(tmp_path: Path):
    source = tmp_path / "source"
    SOURCE_BUILDER.build(
        source,
        start_index=99000,
        train_count=32,
        validation_count=32,
        test_count=32,
    )
    source_rows = load_grpo_corpus(source / "train.jsonl")
    update, evaluation, sft_repair, reject = source_rows[:4]
    routes = {
        update.task.task_id: "grpo_update",
        evaluation.task.task_id: "evaluation",
        sft_repair.task.task_id: "sft_repair",
        reject.task.task_id: "reject",
    }
    report = tmp_path / "audit.json"
    report.write_text(
        json.dumps(
            {
                "decisions": [
                    {"task_id": task_id, "route": route}
                    for task_id, route in routes.items()
                ]
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "routed"
    manifest = ROUTED_BUILDER.build(
        source,
        report,
        output,
        support_per_variant=0,
        anchor_count=8,
    )

    train_ids = {row.task.task_id for row in load_grpo_corpus(output / "train.jsonl")}
    assert train_ids == {update.task.task_id, evaluation.task.task_id}
    assert sft_repair.task.task_id not in train_ids
    assert reject.task.task_id not in train_ids
    assert manifest["train_update_tasks"] == 1
    assert manifest["train_anchors"] == 1
    assert (
        manifest["selection_policy"]["anti_regression_support"]
        == "audited evaluation tasks only"
    )


def test_accepts_jsonl_group_decisions_from_offline_reroute(tmp_path: Path):
    source = tmp_path / "source"
    SOURCE_BUILDER.build(
        source,
        start_index=99100,
        train_count=32,
        validation_count=32,
        test_count=32,
    )
    update, evaluation = load_grpo_corpus(source / "train.jsonl")[:2]
    decisions = tmp_path / "group_decisions.jsonl"
    decisions.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"task_id": update.task.task_id, "route": "grpo_update"},
                {"task_id": evaluation.task.task_id, "route": "evaluation"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "routed"
    manifest = ROUTED_BUILDER.build(
        source,
        decisions,
        output,
        support_per_variant=0,
        anchor_count=1,
    )

    assert manifest["counts"]["train"] == 2
    assert manifest["audit_routes"] == {"grpo_update": 1, "evaluation": 1}
