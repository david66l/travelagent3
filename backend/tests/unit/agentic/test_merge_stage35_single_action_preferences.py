import json
from pathlib import Path

from scripts.merge_stage35_single_action_preferences import merge


def _call() -> dict:
    return {
        "type": "function",
        "function": {"name": "abort", "arguments": {"reason": "unsafe"}},
    }


def _pair(split: str, source: str) -> dict:
    call = _call()
    contract = source == "contract"
    return {
        "schema_version": "teacher-preference-pair.v1",
        "pair_id": f"{source}-{split}",
        "task_id": f"task-{source}-{split}",
        "family": "single_action_abort" if contract else "abort",
        "context_hash": f"context-{source}-{split}",
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "request"},
        ],
        "tools": [{"type": "function", "function": {"name": "abort"}}],
        "chosen": {"role": "assistant", "content": None, "tool_calls": [call]},
        "rejected": {
            "role": "assistant",
            "content": None,
            "tool_calls": [call, call]
            if contract
            else [
                {
                    "type": "function",
                    "function": {"name": "ask_user", "arguments": {}},
                }
            ],
        },
        "chosen_trajectory_id": f"chosen-{source}-{split}",
        "rejected_trajectory_id": f"rejected-{source}-{split}",
        "reason_codes": [
            "SINGLE_ACTION_CONTRACT_OVER_DUPLICATE_CALL"
            if contract
            else "VERIFIER_SUCCESS_OVER_FAILURE"
        ],
        "reward_margin": 1.0,
    }


def _write(path: Path, row: dict) -> None:
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_merge_preserves_two_explicit_evidence_classes(tmp_path: Path):
    base = tmp_path / "base"
    target = tmp_path / "target"
    preferences = target / "preferences"
    output = tmp_path / "output"
    base.mkdir()
    preferences.mkdir(parents=True)
    (base / "manifest.json").write_text(
        json.dumps({"status": "passed", "requires_verifier_success_over_failure": True}),
        encoding="utf-8",
    )
    (target / "manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "accepted_frozen_exact_overlap": 0,
                "accepted_frozen_near_overlap": 0,
            }
        ),
        encoding="utf-8",
    )
    for split in ("train", "validation", "test"):
        _write(base / f"{split}.jsonl", _pair(split, "verifier"))
        _write(preferences / f"{split}.jsonl", _pair(split, "contract"))

    manifest = merge(base, target, output)

    assert manifest["status"] == "passed"
    assert manifest["split_counts"] == {"train": 2, "validation": 2, "test": 2}
    assert manifest["evidence_counts"] == {
        "verifier_success_over_failure": 3,
        "deterministic_single_action_contract": 3,
    }
    assert manifest["context_split_overlap"] == 0
