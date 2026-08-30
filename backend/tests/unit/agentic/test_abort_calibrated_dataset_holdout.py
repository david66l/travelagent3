import json

from scripts.build_abort_calibrated_training_datasets import (
    _audit_grpo_holdout_pairs,
)


def _write_holdout(path):
    row = {
        "task": {
            "task_id": "holdout-1",
            "user_request": "红色预警期间不能前往封闭景区。",
        }
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def test_grpo_holdout_audit_accepts_distinct_training_requests(tmp_path):
    holdout = tmp_path / "test.jsonl"
    _write_holdout(holdout)

    report = _audit_grpo_holdout_pairs(
        [("train-1", "指定场馆闭馆，日期和场馆都不能调整。")],
        holdout,
    )

    assert report["passed"] is True
    assert report["training_requests_missing"] == 0
    assert report["task_id_overlap"] == 0
    assert report["exact_normalized_request_overlap"] == 0


def test_grpo_holdout_audit_rejects_id_or_normalized_request_overlap(tmp_path):
    holdout = tmp_path / "test.jsonl"
    _write_holdout(holdout)

    report = _audit_grpo_holdout_pairs(
        [
            ("holdout-1", "另一条请求"),
            ("train-2", "红色预警期间，不能前往封闭景区！"),
        ],
        holdout,
    )

    assert report["passed"] is False
    assert report["task_id_overlap"] == 1
    assert report["exact_normalized_request_overlap"] == 1


def test_grpo_holdout_audit_tracks_missing_minimal_context_requests(tmp_path):
    holdout = tmp_path / "test.jsonl"
    _write_holdout(holdout)

    report = _audit_grpo_holdout_pairs([("minimal-context", "")], holdout)

    assert report["passed"] is True
    assert report["training_pairs"] == 1
    assert report["training_requests_present"] == 0
    assert report["training_requests_missing"] == 1
