import json

from scripts.validate_checkpoint_candidate import validate


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path):
    checkpoint = "artifacts/grpo-candidate"
    gate = {
        "promoted": True,
        "after_checkpoint": checkpoint,
        "gate_errors": [],
        "paired_contract": {"seed_protocol": "paired-v1"},
    }
    _write(tmp_path / "small.json", gate)
    _write(tmp_path / "holdout.json", gate)
    _write(
        tmp_path / "comparison.json",
        {
            "contract": {"seed_protocol": "paired-v1"},
            "arms": [
                {
                    "checkpoint": checkpoint,
                    "tasks": 4,
                    "success_rate": 0.75,
                    "mean_reward": 0.5,
                }
            ],
        },
    )
    _write(
        tmp_path / "app-smoke.json",
        {
            "checkpoint": checkpoint,
            "agent_status": "awaiting_confirmation",
            "hard_pass": True,
            "itinerary_days": 1,
        },
    )
    manifest = {
        "status": "offline_qualified_online_unreleased",
        "checkpoint": checkpoint,
        "credit_assignment_claim": "trajectory-level only",
        "offline_evaluation": {
            "report": "comparison.json",
            "paired_seed_protocol": "paired-v1",
            "tasks": 4,
            "success_rate": 0.75,
            "mean_reward": 0.5,
        },
        "promotion_evidence": {
            "small_gate": "small.json",
            "holdout_gate": "holdout.json",
        },
        "app_branch_smoke": {
            "report": "app-smoke.json",
            "status": "awaiting_confirmation",
            "hard_pass": True,
            "itinerary_days": 1,
        },
    }
    path = tmp_path / "candidate.json"
    _write(path, manifest)
    return path


def test_candidate_accepts_matching_promoted_evidence(tmp_path):
    path = _fixture(tmp_path)

    result = validate(path, repo_root=tmp_path)

    assert result["valid"] is True
    assert result["errors"] == []


def test_candidate_rejects_failed_gate(tmp_path):
    path = _fixture(tmp_path)
    gate_path = tmp_path / "holdout.json"
    gate = json.loads(gate_path.read_text())
    gate["promoted"] = False
    gate["gate_errors"] = ["regression"]
    _write(gate_path, gate)

    result = validate(path, repo_root=tmp_path)

    assert result["valid"] is False
    assert any("did not promote" in error for error in result["errors"])
