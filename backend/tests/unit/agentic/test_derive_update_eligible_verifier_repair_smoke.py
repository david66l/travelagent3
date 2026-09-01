import json

from scripts.derive_update_eligible_verifier_repair_smoke import derive


def _row(task_id: str, target: str) -> dict:
    return {
        "task": {"task_id": task_id},
        "snapshot": {
            "hidden_test_facts": {
                "grpo_decision_state": {
                    "target_action": target,
                    "flexibility_pattern": f"pattern-{target}",
                }
            }
        },
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _decision(task_id: str, reward_std: float, rewards: list[float]) -> dict:
    mean = sum(rewards) / len(rewards)
    return {
        "task_id": task_id,
        "initial_state_fingerprint": f"fingerprint-{task_id}",
        "group_size": 4,
        "reward_std": reward_std,
        "zero_variance": reward_std == 0,
        "eligible_for_update": reward_std > 0,
        "route": "grpo_update" if reward_std > 0 else "evaluation",
        "advantages": [
            {
                "reward": reward,
                "standardized_advantage": reward - mean,
            }
            for reward in rewards
        ],
    }


def test_derive_selects_first_eligible_non_abort_in_source_order(tmp_path):
    source = tmp_path / "source"
    audit_dir = tmp_path / "audit"
    output = tmp_path / "output"
    source.mkdir()
    audit_dir.mkdir()
    rows = [
        _row("retry-first", "retry_solve"),
        _row("tradeoff-larger-std", "propose_tradeoff"),
        _row("abort-anchor", "abort"),
    ]
    _write_jsonl(source / "train.jsonl", rows)
    _write_jsonl(source / "validation.jsonl", [_row("validation", "abort")])
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
    decisions = [
        _decision("retry-first", 0.5, [1, 1, -1, -1]),
        _decision("tradeoff-larger-std", 1.0, [1, 1, -1, -1]),
        _decision("abort-anchor", 0.0, [1, 1, 1, 1]),
    ]
    (audit_dir / "report.json").write_text(
        json.dumps({"decisions": decisions}), encoding="utf-8"
    )
    _write_jsonl(
        audit_dir / "rollouts.jsonl",
        [
            {
                "task_id": task_id,
                "policy_errors": [],
                "actions": [{"action": "valid", "error_code": None}],
            }
            for task_id in ("retry-first", "tradeoff-larger-std", "abort-anchor")
            for _ in range(4)
        ],
    )

    manifest = derive(
        source_dir=source,
        audit_report=audit_dir / "report.json",
        output_dir=output,
        target_action="retry_solve",
    )

    assert manifest["selected"]["task_id"] == "retry-first"
    assert manifest["selected"]["target_action"] == "retry_solve"
    assert manifest["preregistered_target_action"] == "retry_solve"
    assert manifest["reporting_eligibility"] == "diagnostic_smoke_only"
    assert manifest["counts"] == {"train": 1, "validation": 1}
    assert manifest["frozen_test_in_training"] is False
    assert _read_selected(output / "train.jsonl") == "retry-first"


def _read_selected(path) -> str:
    return json.loads(path.read_text(encoding="utf-8"))["task"]["task_id"]


def test_derive_rejects_a_different_target_instead_of_falling_back(tmp_path):
    source = tmp_path / "source"
    audit_dir = tmp_path / "audit"
    output = tmp_path / "output"
    source.mkdir()
    audit_dir.mkdir()
    _write_jsonl(source / "train.jsonl", [_row("retry-only", "retry_solve")])
    _write_jsonl(source / "validation.jsonl", [_row("validation", "abort")])
    (source / "manifest.json").write_text("{}", encoding="utf-8")
    (audit_dir / "report.json").write_text(
        json.dumps({"decisions": [_decision("retry-only", 0.5, [1, 1, -1, -1])]}),
        encoding="utf-8",
    )
    _write_jsonl(
        audit_dir / "rollouts.jsonl",
        [
            {
                "task_id": "retry-only",
                "policy_errors": [],
                "actions": [{"action": "retry_solve", "error_code": None}],
            }
            for _ in range(4)
        ],
    )

    try:
        derive(
            source_dir=source,
            audit_report=audit_dir / "report.json",
            output_dir=output,
            target_action="propose_tradeoff",
        )
    except ValueError as exc:
        assert "propose_tradeoff" in str(exc)
    else:
        raise AssertionError("derive must not fall back to another target action")
