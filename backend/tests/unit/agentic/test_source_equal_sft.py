import math

import pytest

from agentic.source_equal_sft import completion_weighted_loss, source_equal_weights


def records():
    rows = [{"example_id": key} for key in ("a1", "a2", "b1")]
    lineage = [{"record_id": key, "source_state_id": key[0], "split": "train"} for key in ("a1", "a2", "b1")]
    return rows, lineage


def test_two_targets_do_not_double_source_mass():
    rows, lineage = records()
    weights, report = source_equal_weights(rows, lineage, split="train")
    assert weights == [0.75, 0.75, 1.5]
    assert report["verified_equal_source_mass"] is True
    assert report["total_example_weight"] == 3
    assert report["source_total_weight_min"] == report["source_total_weight_max"] == 1.5


@pytest.mark.parametrize("problem", ["missing", "duplicate", "wrong_split", "cross_split", "too_many"])
def test_source_weighting_fails_closed(problem):
    rows, lineage = records()
    if problem == "missing":
        lineage.pop()
    elif problem == "duplicate":
        lineage.append(lineage[0])
    elif problem == "wrong_split":
        lineage[-1]["split"] = "train-shadow"
    elif problem == "cross_split":
        lineage.append({"record_id": "a3", "source_state_id": "a", "split": "train-shadow"})
    else:
        rows.append({"example_id": "a3"})
        lineage.append({"record_id": "a3", "source_state_id": "a", "split": "train"})
    with pytest.raises(ValueError):
        source_equal_weights(rows, lineage, split="train")


def test_one_example_multiplier_is_not_normalized_away():
    torch = pytest.importorskip("torch")
    logits = torch.zeros((1, 4, 2), requires_grad=True)
    labels = torch.tensor([[-100, -100, 0, 1]])
    token_weights = torch.ones_like(labels, dtype=torch.float32)
    half = completion_weighted_loss(logits, labels, token_weights, torch.tensor([0.5]))
    assert half.item() == pytest.approx(math.log(2) * 0.5)
    half.backward()
    assert logits.grad[0, 0].abs().sum().item() == 0


def test_weighted_row_mean_equals_source_mean_including_gradient():
    torch = pytest.importorskip("torch")
    scalar = torch.tensor(0.4, requires_grad=True)
    values = [1.0, 3.0, 2.0]
    labels = torch.tensor([[-100, 0, 0]])
    token_weights = torch.ones_like(labels, dtype=torch.float32)
    unweighted = []
    weighted = []
    for value, weight in zip(values, [0.75, 0.75, 1.5], strict=True):
        logits = torch.stack((scalar * value, scalar * 0)).expand(1, 3, 2)
        unweighted.append(completion_weighted_loss(logits, labels, token_weights))
        weighted.append(completion_weighted_loss(logits, labels, token_weights, torch.tensor([weight])))
    expected = ((unweighted[0] + unweighted[1]) / 2 + unweighted[2]) / 2
    actual = sum(weighted) / 3
    assert actual.item() == pytest.approx(expected.item())
    expected_grad = torch.autograd.grad(expected, scalar, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual, scalar)[0]
    assert actual_grad.item() == pytest.approx(expected_grad.item())
