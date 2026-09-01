import pytest

from ml.agentic.training.train_grpo import (
    latest_completed_eval_metrics,
    resolve_num_generations_eval,
    validate_eval_corpus_size,
    validate_routed_audit_protocol,
    validate_turn_credit_totals,
)


def _totals(**overrides):
    values = {
        "train_effective_nonzero_credited_turns": 8,
        "train_compared_turn_buckets": 4,
        "train_zero_variance_turn_buckets": 1,
        "train_invalid_action_positive_credit_count": 0,
        "alignment_rejected_trajectories": 0,
        "extra_unmatched_model_turns": 0,
    }
    values.update(overrides)
    return values


def test_r1_v2_training_evidence_gate_accepts_real_nonzero_safe_credit():
    assert validate_turn_credit_totals(_totals()) == []


def test_r1_v2_training_evidence_gate_rejects_fake_or_unsafe_learning():
    assert "NO_EFFECTIVE_NONZERO_TRAIN_TURN_CREDIT" in validate_turn_credit_totals(
        _totals(train_effective_nonzero_credited_turns=0)
    )
    assert "ALL_COMPARABLE_TURN_BUCKETS_ZERO_VARIANCE" in validate_turn_credit_totals(
        _totals(train_zero_variance_turn_buckets=4)
    )
    assert "INVALID_ACTION_RECEIVED_POSITIVE_CREDIT" in validate_turn_credit_totals(
        _totals(train_invalid_action_positive_credit_count=1)
    )
    assert "TURN_TO_TOKEN_ALIGNMENT_NOT_PROVEN" in validate_turn_credit_totals(
        _totals(alignment_rejected_trajectories=1)
    )
    assert "EXTRA_UNMATCHED_MODEL_TURNS" in validate_turn_credit_totals(
        _totals(extra_unmatched_model_turns=1)
    )


def test_reuses_completed_epoch_end_eval_instead_of_running_it_twice():
    history = [
        {"step": 5, "reward": 0.1},
        {"step": 18, "eval_reward": 0.25, "eval_runtime": 12.0, "epoch": 1.0},
    ]

    assert latest_completed_eval_metrics(history) == {
        "eval_reward": 0.25,
        "eval_runtime": 12.0,
    }


def test_resolves_independent_train_and_eval_generation_groups():
    assert resolve_num_generations_eval(8, 4) == 4
    assert resolve_num_generations_eval(8, 0) == 8

    with pytest.raises(ValueError, match="num_generations must be at least 4"):
        resolve_num_generations_eval(2, 4)
    with pytest.raises(ValueError, match="num_generations_eval"):
        resolve_num_generations_eval(8, 2)


def test_validates_corpus_against_eval_group_instead_of_train_group():
    validate_eval_corpus_size(4, 4)

    with pytest.raises(ValueError, match=r"\(3<4\)"):
        validate_eval_corpus_size(3, 4)
    with pytest.raises(ValueError, match=r"\(5%4\)"):
        validate_eval_corpus_size(5, 4)


def test_rejects_routed_curriculum_when_audit_and_training_protocols_differ():
    manifest = {
        "schema_version": "routed-grpo-curriculum.v1",
        "uniform_audit_protocol": {
            "checkpoint_adapter_sha256": "adapter-sha",
            "execution_mode": "react",
            "temperature": 1.2,
            "decoding_mode": "native-unconstrained",
            "quantization": "nf4-double-quant",
            "group_size": 8,
            "max_new_tokens": 192,
            "max_tool_calling_iterations": 1,
            "reward_config_versions": ["hierarchical-b0.v2"],
        },
    }
    common = {
        "manifest": manifest,
        "source_adapter_sha256": "adapter-sha",
        "num_generations": 8,
        "temperature": 1.2,
        "execution_mode": "react",
        "audit_max_new_tokens": 192,
        "max_tool_calling_iterations": 1,
        "reward_config_version": "hierarchical-b0.v2",
    }

    assert validate_routed_audit_protocol(**common) == []
    assert validate_routed_audit_protocol(
        **{**common, "num_generations": 4}
    ) == ["AUDIT_TRAIN_PROTOCOL_MISMATCH:group_size"]
    assert validate_routed_audit_protocol(
        {"schema_version": "routed-grpo-curriculum.v1"},
        **{key: value for key, value in common.items() if key != "manifest"},
    ) == ["UNIFORM_AUDIT_PROTOCOL_MISSING"]
