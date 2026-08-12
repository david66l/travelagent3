"""Tests for Agentic GRPO corpus gates and TRL row conversion."""

import json

from agentic.grpo_training import (
    GRPOCorpusRow,
    preflight_grpo_corpus,
    to_trl_environment_rows,
)
from tests.unit.agentic.test_environment import _snapshot, _task


def _write(path, rows):
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_environment_rows_keep_snapshot_out_of_model_prompt():
    row = GRPOCorpusRow(task=_task(), snapshot=_snapshot())

    converted = to_trl_environment_rows([row])[0]

    assert [message["role"] for message in converted["prompt"]] == ["system", "user"]
    assert converted["prompt"][-1]["content"] == ""
    assert "hidden_test_facts" not in json.dumps(converted["prompt"])
    assert converted["snapshot"]["hidden_test_facts"] == {"closed_pois": []}
    assert converted["initial_state_fingerprint"]


def test_preflight_accepts_complete_non_overlapping_snapshot_corpus(tmp_path):
    train_task = _task()
    validation_task = _task().model_copy(update={"task_id": "validation-task", "seed": 99})
    _write(
        tmp_path / "train.jsonl",
        [
            {
                "task": train_task.model_dump(mode="json"),
                "snapshot": _snapshot().model_dump(mode="json"),
            }
        ],
    )
    validation_snapshot = _snapshot().model_copy(update={"state_id": "validation-state"})
    _write(
        tmp_path / "validation.jsonl",
        [
            {
                "task": validation_task.model_dump(mode="json"),
                "snapshot": validation_snapshot.model_dump(mode="json"),
            }
        ],
    )

    report = preflight_grpo_corpus(tmp_path, minimum_train_tasks=1, require_dependencies=False)

    assert report.ready is True
    assert report.train_tasks == 1
    assert report.validation_tasks == 1


def test_preflight_blocks_split_leakage_missing_tools_and_pii(tmp_path):
    task = _task()
    task.user_request = "Call 13812345678"
    snapshot = _snapshot()
    snapshot.tool_responses.pop("validate_itinerary")
    row = {"task": task.model_dump(mode="json"), "snapshot": snapshot.model_dump(mode="json")}
    _write(tmp_path / "train.jsonl", [row])
    _write(tmp_path / "validation.jsonl", [row])

    report = preflight_grpo_corpus(tmp_path, minimum_train_tasks=2, require_dependencies=False)

    assert report.ready is False
    assert "TASK_ID_SPLIT_OVERLAP" in report.errors
    assert "INITIAL_STATE_SPLIT_OVERLAP" in report.errors
    assert any(error.startswith("PII_DETECTED") for error in report.errors)
    assert any(error.startswith("SNAPSHOT_TOOLS_MISSING") for error in report.errors)
    assert any(error.startswith("TRAIN_TASKS_BELOW_MINIMUM") for error in report.errors)
