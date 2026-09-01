import json

from scripts.derive_balanced_verifier_repair_smoke import derive


def _row(index: int, target: str, pattern: str, split: str) -> dict:
    return {
        "task": {"task_id": f"{split}-{index}"},
        "snapshot": {
            "hidden_test_facts": {
                "grpo_decision_state": {
                    "target_action": target,
                    "flexibility_pattern": pattern,
                    "source_task_id": f"source-{split}-{index}",
                    "source_snapshot_version": f"snapshot-{split}-{index}",
                }
            }
        },
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_derivation_balances_actions_and_contract_patterns_without_test_access(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "smoke"
    source.mkdir()
    train = [
        _row(0, "retry_solve", "P0", "train"),
        _row(1, "retry_solve", "P1", "train"),
        _row(2, "retry_solve", "P2", "train"),
        _row(3, "propose_tradeoff", "P0", "train"),
        _row(4, "propose_tradeoff", "P2", "train"),
        _row(5, "propose_tradeoff", "P1", "train"),
        _row(6, "abort", "P1", "train"),
        _row(7, "abort", "P0", "train"),
        _row(8, "abort", "P2", "train"),
    ]
    validation = [
        _row(index, target, pattern, "validation")
        for index, (target, pattern) in enumerate(
            (target, pattern)
            for pattern in ("P0", "P1", "P2", "P0")
            for target in ("retry_solve", "propose_tradeoff", "abort")
        )
    ]
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "react-verifier-repair-corpus.v2",
                "decision_schema_version": "react-verifier-repair-decision.v2",
                "constraint_flexibility_schema_version": "constraint-flexibility.v1",
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(source / "train.jsonl", train)
    _write_jsonl(source / "validation.jsonl", validation)
    (source / "test.jsonl").write_text("not-json-and-must-not-be-read\n", encoding="utf-8")

    report = derive(source_dir=source, output_dir=output)

    assert report["reporting_eligibility"] == "exploratory_smoke_only"
    assert report["counts"] == {"train": 3, "validation": 12}
    assert report["target_counts"]["train"] == {
        "abort": 1,
        "propose_tradeoff": 1,
        "retry_solve": 1,
    }
    assert report["target_counts"]["validation"] == {
        "abort": 4,
        "propose_tradeoff": 4,
        "retry_solve": 4,
    }
    assert report["flexibility_pattern_counts"]["train"] == {
        "P0": 1,
        "P1": 1,
        "P2": 1,
    }
    assert not (output / "test.jsonl").exists()
