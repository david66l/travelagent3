from ml.agentic.training.train_grpo import (
    latest_completed_eval_metrics,
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
