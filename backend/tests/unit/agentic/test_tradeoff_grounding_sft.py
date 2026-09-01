import json
from pathlib import Path

import pytest

from agentic.sft_dataset import SFTExample
from agentic.grpo_training import load_grpo_corpus
from scripts.build_tradeoff_grounding_sft import build


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_TRAIN = (
    REPO_ROOT
    / "ml/agentic/datasets/native-react-verifier-repair-grpo-rl-challenge-v5/train.jsonl"
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def v5_source_train(tmp_path) -> Path:
    """Provide a non-production schema fixture for the v5-only builder tests.

    The checked-in historical corpus is intentionally v3 and must remain
    rejected in production.  Its prompt/state payload is still adequate for
    testing source partitioning once the fixture is explicitly upgraded; real
    production corpora are rebuilt from raw snapshots by the v5 builder.
    """
    rows = _read_jsonl(SOURCE_TRAIN)
    for row in rows:
        contract = row["snapshot"]["hidden_test_facts"]["grpo_decision_state"]
        contract["schema_version"] = "react-verifier-repair-decision.v5"
        if contract.get("target_action") == "retry_solve":
            contract["controller_arguments"] = dict(
                contract.get("expected_arguments") or {"strategy": "greedy"}
            )
            contract["expected_arguments"] = {}
        else:
            contract["controller_arguments"] = {}
    path = tmp_path / "train.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_builder_uses_only_train_and_keeps_source_groups_disjoint(tmp_path, v5_source_train):
    report = build(v5_source_train, tmp_path)

    assert report["status"] == "passed"
    assert report["official_validation_or_test_used"] is False
    assert report["source_files_read"] == [v5_source_train.as_posix()]
    assert report["source_state_counts"] == {
        "optimization": 40,
        "internal_dev": 10,
        "train_shadow": 10,
    }
    assert report["split_counts"] == {"train": 180, "validation": 50, "test": 50}
    assert report["quality_gate"]["source_overlap_zero"] is True
    assert report["quality_gate"]["source_validation_or_test_files_read"] == 0
    assert report["matched_boundary_pair_count"] >= 30
    internal_dev = load_grpo_corpus(tmp_path / "internal_dev_grpo.jsonl")
    train_shadow = load_grpo_corpus(tmp_path / "train_shadow_grpo.jsonl")
    internal_sources = {
        row.snapshot.hidden_test_facts["grpo_decision_state"]["source_task_id"]
        for row in internal_dev
    }
    shadow_sources = {
        row.snapshot.hidden_test_facts["grpo_decision_state"]["source_task_id"]
        for row in train_shadow
    }
    assert len(internal_dev) == len(train_shadow) == 50
    assert internal_sources.isdisjoint(shadow_sources)


def test_balanced_actions_fail_closed_without_strict_source_profile(tmp_path, v5_source_train):
    with pytest.raises(ValueError, match="strict-counterfactual"):
        build(v5_source_train, tmp_path, balanced_actions=True)


def test_builder_enforces_model_owned_grounding_and_controller_authority(
    tmp_path, v5_source_train
):
    report = build(v5_source_train, tmp_path)
    quality = report["quality_gate"]

    assert quality["exactly_one_tool_call_rate"] == 1.0
    assert quality["schema_valid_rate"] == 1.0
    assert quality["reason_grounding_rate"] == 1.0
    assert quality["reason_quality_rate"] == 1.0
    assert quality["reason_rationale_first_rate"] == 1.0
    assert quality["tradeoff_model_owned_schema_rate"] == 1.0
    assert quality["tradeoff_controller_authority_present_rate"] == 1.0
    assert quality["private_label_leak_count"] == 0
    assert quality["exact_model_visible_duplicate_count"] == 0
    assert quality["action_counts"] == {
        "abort": 50,
        "propose_tradeoff": 180,
        "retry_solve": 50,
    }

    for split, expected in (("train", 180), ("validation", 50), ("test", 50)):
        examples = [SFTExample(**row) for row in _read_jsonl(tmp_path / f"{split}.jsonl")]
        assert len(examples) == expected
        assert all(example.split == split for example in examples)
        assert all(example.policy_version == report["schema_version"] for example in examples)


def test_builder_exports_non_training_negatives_and_blind_review_packet(
    tmp_path, v5_source_train
):
    report = build(v5_source_train, tmp_path)
    rejected = _read_jsonl(tmp_path / "rejected_examples.jsonl")
    packet = _read_jsonl(tmp_path / "blind_review_packet.jsonl")
    key = _read_jsonl(tmp_path / "blind_review_key.jsonl")

    assert report["training_authorized"] is False
    assert report["training_blocker"] == "INDEPENDENT_BLIND_REVIEW_PENDING"
    assert len(rejected) == 240
    assert sum(row["candidate"]["name"] == "abort" for row in rejected) == 120
    assert sum(
        row["candidate"]["name"] == "propose_tradeoff" for row in rejected
    ) == 120
    assert len(packet) == len(key) == 120
    assert {row["review_id"] for row in packet} == {row["review_id"] for row in key}
    assert all("expected_action" not in row for row in packet)
    assert sum(row.get("expected_disposition") == "reject" for row in key) == 30
    assert all(row["review_id"].startswith("r-") for row in packet)
    assert all("review_kind" not in row for row in packet)
    assert all("review_category" not in row for row in packet)

    # The packet intentionally contains invalid candidates for a blind reviewer
    # to reject.  Its accepted candidates, however, must remain exactly on the
    # serving-time model surface: controller-owned fields can never appear in a
    # supervised completion or in a positive blind-review example.
    key_by_id = {row["review_id"]: row for row in key}
    accepted = [
        row
        for row in packet
        if key_by_id[row["review_id"]].get("expected_disposition") != "reject"
    ]
    assert len(accepted) == 90
    for row in accepted:
        calls = row["candidate"]["tool_calls"]
        assert len(calls) == 1
        arguments = calls[0]["function"]["arguments"]
        assert set(arguments) == {"reason"}
        assert "options" not in arguments
        assert "strategy" not in arguments


def test_builder_rejects_historical_v3_source_corpus(tmp_path):
    with pytest.raises(ValueError, match="unsupported decision schema"):
        build(SOURCE_TRAIN, tmp_path)


def test_hard_case_actions_fail_closed_without_strict_source_profile(
    tmp_path, v5_source_train
):
    with pytest.raises(ValueError, match="strict-counterfactual"):
        build(v5_source_train, tmp_path, hard_case_actions=True)
