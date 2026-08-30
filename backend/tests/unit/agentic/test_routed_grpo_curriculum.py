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


def test_reports_verifier_repair_action_distribution(tmp_path: Path):
    source = tmp_path / "source"
    SOURCE_BUILDER.build(
        source,
        start_index=99200,
        train_count=32,
        validation_count=32,
        test_count=32,
    )
    rows = load_grpo_corpus(source / "train.jsonl")
    update, evaluation = rows[:2]
    update.snapshot.hidden_test_facts["grpo_decision_state"] = {
        "target_action": "propose_tradeoff"
    }
    evaluation.snapshot.hidden_test_facts["grpo_decision_state"] = {
        "target_action": "abort"
    }
    (source / "train.jsonl").write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json"), ensure_ascii=False)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "audit.json"
    report.write_text(
        json.dumps(
            {
                "decisions": [
                    {"task_id": update.task.task_id, "route": "grpo_update"},
                    {"task_id": evaluation.task.task_id, "route": "evaluation"},
                ]
            }
        ),
        encoding="utf-8",
    )

    manifest = ROUTED_BUILDER.build(
        source,
        report,
        tmp_path / "routed",
        support_per_variant=0,
        anchor_count=1,
    )

    assert manifest["train_update_actions"] == {"propose_tradeoff": 1}
    assert manifest["train_anchor_actions"] == {"abort": 1}
    assert manifest["train_actions"] == {"abort": 1, "propose_tradeoff": 1}
