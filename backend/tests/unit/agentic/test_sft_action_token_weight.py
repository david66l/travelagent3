import hashlib

import pytest

torch = pytest.importorskip("torch", reason="agentic-training optional dependency")

from agentic.chat_template_contract import AGENT_CHAT_TEMPLATE_SHA256  # noqa: E402
from ml.agentic.training.train_sft import (  # noqa: E402
    apply_action_sequence_weights,
    configure_agent_sft_tokenizer,
    declare_microbatch_mean_loss,
    preflight_rationale_sequence_coverage,
    sequence_match_count,
    stratify_sft_rows_by_action,
)


class _Tokenizer:
    chat_template = "native-tool-template"
    pad_token = None
    eos_token = "<eos>"


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


def test_rationale_weight_changes_only_the_declared_prefix_tokens():
    labels = torch.tensor([[-100, 10, 11, 20, 21, 22, 30, 99]])
    weights = torch.where(labels == -100, 0.0, 1.0)

    apply_action_sequence_weights(labels, weights, ((20, 21, 22),), 3.0)

    assert weights.tolist() == [[0.0, 1.0, 1.0, 3.0, 3.0, 3.0, 1.0, 1.0]]


def test_rationale_sequence_preflight_requires_exactly_one_match_per_row():
    class _EncodingTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [int(value) for value in text.split()]

    report = preflight_rationale_sequence_coverage(
        [{"completion": "1 20 21 2"}, {"completion": "3 20 21 4"}],
        _EncodingTokenizer(),
        ((20, 21),),
    )

    assert report == {
        "ready": True,
        "rows_checked": 2,
        "matched_once": 2,
        "missing": 0,
        "multiple": 0,
    }
    assert sequence_match_count([20, 21, 20, 21], ((20, 21),)) == 2


def test_sft_tokenizer_uses_shared_prefix_preserving_template():
    tokenizer = configure_agent_sft_tokenizer(_Tokenizer())

    assert tokenizer.pad_token == "<eos>"
    assert hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() == (
        AGENT_CHAT_TEMPLATE_SHA256
    )


def _row(action: str, row_id: int) -> dict:
    return {
        "row_id": row_id,
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": action}}],
            }
        ],
    }


def test_stratified_sft_rows_give_four_of_each_action_per_twelve():
    actions = ("abort", "propose_tradeoff", "solve_itinerary")
    rows = [_row(action, index) for action in reversed(actions) for index in range(8)]

    ordered = stratify_sft_rows_by_action(rows)

    for start in (0, 12):
        batch_actions = [
            row["messages"][-1]["tool_calls"][0]["function"]["name"]
            for row in ordered[start : start + 12]
        ]
        assert {action: batch_actions.count(action) for action in actions} == {
            action: 4 for action in actions
        }


def test_transformers_training_step_matches_manual_twelve_microbatch_mean(tmp_path):
    from transformers import Trainer, TrainingArguments

    class _ScalarModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

    class _MicroMeanTrainer(Trainer):
        def compute_loss(
            self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch=None,
        ):
            del num_items_in_batch
            loss = (model.weight * inputs["x"]).pow(2).mean()
            return (loss, {}) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=12,
        report_to=[],
        use_cpu=True,
    )
    values = [torch.tensor([float(index)]) for index in range(1, 13)]

    def accumulated_gradient(*, accepts_loss_kwargs: bool) -> float:
        model = _ScalarModel()
        trainer = _MicroMeanTrainer(model=model, args=args)
        trainer.current_gradient_accumulation_steps = 12
        trainer.model_accepts_loss_kwargs = accepts_loss_kwargs
        model.zero_grad(set_to_none=True)
        for value in values:
            trainer.training_step(
                model,
                {"x": value},
                num_items_in_batch=torch.tensor(12),
            )
        return float(model.weight.grad)

    model = _ScalarModel()
    manual_losses = [(model.weight * value).pow(2).mean() for value in values]
    manual_gradient = float(
        torch.autograd.grad(sum(manual_losses) / len(manual_losses), model.weight)[0]
    )
    corrected = accumulated_gradient(accepts_loss_kwargs=False)
    incorrect = accumulated_gradient(accepts_loss_kwargs=True)

    assert corrected == pytest.approx(manual_gradient, rel=1e-5)
    assert incorrect / corrected == pytest.approx(12.0, rel=1e-5)

    marker = type("_TrainerMarker", (), {"model_accepts_loss_kwargs": True})()
    declare_microbatch_mean_loss(marker)
    assert marker.model_accepts_loss_kwargs is False
    assert marker.loss_normalization_contract == ("mean-of-microbatch-token-weighted-means.v1")
