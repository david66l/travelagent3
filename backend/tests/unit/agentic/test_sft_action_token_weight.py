import pytest

torch = pytest.importorskip("torch", reason="agentic-training optional dependency")

from ml.agentic.training.train_sft import apply_action_sequence_weights  # noqa: E402


def test_action_sequence_weight_only_changes_matching_completion_tokens():
    labels = torch.tensor(
        [
            [-100, -100, 10, 20, 30, 40],
            [-100, 10, 99, 20, 30, 40],
        ]
    )
    weights = torch.where(labels == -100, 0.0, 1.0)

    apply_action_sequence_weights(
        labels,
        weights,
        ((20, 30),),
        action_token_weight=6.0,
    )

    assert weights.tolist() == [
        [0.0, 0.0, 1.0, 6.0, 6.0, 1.0],
        [0.0, 1.0, 1.0, 6.0, 6.0, 1.0],
    ]


def test_action_sequence_weight_does_not_reduce_existing_weight():
    labels = torch.tensor([[-100, 5, 6]])
    weights = torch.tensor([[0.0, 8.0, 8.0]])

    apply_action_sequence_weights(labels, weights, ((5, 6),), 4.0)

    assert weights.tolist() == [[0.0, 8.0, 8.0]]
