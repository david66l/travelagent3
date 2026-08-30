import json

from scripts import build_balanced_preferences as balanced


def _row(index: int, family: str, split: str) -> dict:
    return {
        "schema_version": "teacher-preference-pair.v1",
        "pair_id": f"pair-{split}-{family}-{index}",
        "task_id": f"task-{split}-{family}-{index}",
        "family": family,
        "context_hash": f"context-{split}-{family}-{index}",
        "messages": [{"role": "user", "content": "go"}],
        "tools": [],
        "chosen": {"role": "assistant", "content": "good"},
        "rejected": {"role": "assistant", "content": "bad"},
        "reason_codes": ["VERIFIER_SUCCESS_OVER_FAILURE"],
    }


def test_balanced_preferences_caps_majority_and_preserves_frozen_splits(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "requires_verifier_success_over_failure": True,
                "frozen_holdout_payload_overlap": 0,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(balanced, "DEFAULT_TRAIN_LIMITS", {"search": 1, "tradeoff": 10})
    train = [_row(0, "search", "train"), _row(1, "search", "train"), _row(0, "tradeoff", "train")]
    for split, rows in {
        "train": train,
        "validation": [_row(0, "search", "validation")],
        "test": [_row(0, "tradeoff", "test")],
    }.items():
        (source / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )

    result = balanced.build(source, output)

    assert result["selected_train_family_counts"] == {"search": 1, "tradeoff": 1}
    assert result["split_counts"] == {"train": 2, "validation": 1, "test": 1}
    assert result["context_split_overlap"] == 0
    assert result["frozen_holdout_payload_overlap"] == 0
