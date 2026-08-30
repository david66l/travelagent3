import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="agentic-training optional dependency")

from ml.agentic.training.turn_credit_trainer import create_turn_credit_trainer_class  # noqa: E402


def test_thin_trainer_projects_turn_credit_without_importing_real_trl(monkeypatch):
    class FakeGRPOTrainer:
        def __init__(self, *args, **kwargs):
            self.model = SimpleNamespace(training=True)
            self.num_generations = 2
            self.num_generations_eval = 2
            self.environments = [
                SimpleNamespace(
                    _policy_turn_credit_records=lambda gamma: [
                        {"credit": -1.0, "validity": "invalid"},
                        {"credit": 0.9, "validity": "valid"},
                    ]
                ),
                SimpleNamespace(
                    _policy_turn_credit_records=lambda gamma: [
                        {"credit": 0.0, "validity": "external_failure"},
                        {"credit": -0.9, "validity": "valid"},
                    ]
                ),
            ]
            self.logged = []

        def _generate_and_score_completions(self, inputs):
            return {
                "advantages": torch.tensor([1.0, -1.0]),
                "completion_mask": torch.ones((2, 5), dtype=torch.int),
                "tool_mask": torch.tensor([[1, 0, 0, 1, 0], [1, 0, 0, 1, 0]]),
            }

        def _log_metric(self, name, value):
            self.logged.append((name, value))

    fake_trl = ModuleType("trl")
    fake_trl.GRPOTrainer = FakeGRPOTrainer
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    trainer_class = create_turn_credit_trainer_class()
    trainer = trainer_class(turn_credit_blend=0.5)

    output = trainer._generate_and_score_completions([])

    assert output["advantages"].shape == (2, 5)
    assert output["advantages"][0, 0] < output["advantages"][0, 3]
    assert output["advantages"][0, 1:3].tolist() == [0.0, 0.0]
    assert trainer.turn_credit_totals["locally_credited_turns"] == 2
    assert trainer.turn_credit_totals["effective_nonzero_credited_turns"] == 2
    assert trainer.turn_credit_totals["train_locally_credited_turns"] == 2
    assert trainer.turn_credit_totals["train_effective_nonzero_credited_turns"] == 2
    assert trainer.turn_credit_totals["eval_locally_credited_turns"] == 0
    assert trainer.turn_credit_totals["invalid_model_turns"] == 1
    assert trainer.turn_credit_totals["external_failure_turns"] == 1
    assert trainer.turn_credit_totals["invalid_action_positive_credit_count"] == 0
    assert trainer.turn_credit_totals["train_compared_turn_buckets"] == 1
    assert trainer.turn_credit_totals["train_zero_variance_turn_buckets"] == 0
    assert trainer.turn_credit_totals["train_invalid_action_positive_credit_count"] == 0
    assert any(name == "turn_credit/credited_turn_ratio" for name, _ in trainer.logged)
    assert any(
        name == "turn_credit/invalid_action_positive_credit_rate" for name, _ in trainer.logged
    )


def _audit_trainer(monkeypatch, tmp_path, *, completion_ids, tool_mask, decode):
    class FakeGRPOTrainer:
        def __init__(self, *args, **kwargs):
            self.model = SimpleNamespace(training=True)
            self.args = SimpleNamespace(output_dir=str(tmp_path))
            self.processing_class = SimpleNamespace(decode=decode)
            self.num_generations = 2
            self.num_generations_eval = 2
            self.environments = [
                SimpleNamespace(
                    _task_id="task-mismatch",
                    _session=SimpleNamespace(
                        recorder=SimpleNamespace(
                            episode=SimpleNamespace(trajectory_id="trajectory-mismatch")
                        )
                    ),
                    _policy_turn_credit_records=lambda gamma: [
                        {"credit": 0.8, "validity": "valid", "step_index": 0}
                    ],
                ),
                SimpleNamespace(
                    _task_id="task-normal",
                    _session=SimpleNamespace(
                        recorder=SimpleNamespace(
                            episode=SimpleNamespace(trajectory_id="trajectory-normal")
                        )
                    ),
                    _policy_turn_credit_records=lambda gamma: [
                        {"credit": 0.7, "validity": "valid", "step_index": 0}
                    ],
                ),
            ]

        def _generate_and_score_completions(self, inputs):
            return {
                "advantages": torch.tensor([1.0, -1.0]),
                "completion_ids": completion_ids,
                "completion_mask": torch.ones_like(tool_mask),
                "tool_mask": tool_mask,
            }

        def _log_metric(self, name, value):
            pass

    fake_trl = ModuleType("trl")
    fake_trl.GRPOTrainer = FakeGRPOTrainer
    monkeypatch.setitem(sys.modules, "trl", fake_trl)
    trainer_class = create_turn_credit_trainer_class()
    return trainer_class(turn_credit_blend=0.5)


def test_alignment_audit_is_not_written_for_normal_final_assistant_span(tmp_path, monkeypatch):
    tool_mask = torch.tensor([[1, 0, 1], [1, 0, 1]])
    trainer = _audit_trainer(
        monkeypatch,
        tmp_path,
        completion_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        tool_mask=tool_mask,
        decode=lambda token_ids, **kwargs: "normal",
    )

    trainer._generate_and_score_completions([])

    assert trainer.turn_credit_totals["alignment_rejected_trajectories"] == 0
    assert not trainer.turn_credit_alignment_audit_path.exists()


def test_alignment_mismatch_is_zeroed_and_writes_bounded_evidence(tmp_path, monkeypatch):
    tool_mask = torch.tensor([[1, 0, 1, 0, 1], [1, 0, 1, 0, 0]])
    trainer = _audit_trainer(
        monkeypatch,
        tmp_path,
        completion_ids=torch.tensor([[11, 12, 13, 14, 15], [21, 22, 23, 24, 25]]),
        tool_mask=tool_mask,
        decode=lambda token_ids, **kwargs: "decoded:" + ",".join(map(str, token_ids)),
    )

    output = trainer._generate_and_score_completions([])

    assert output["advantages"][0].tolist() == [0.0] * 5
    assert trainer.turn_credit_totals["alignment_rejected_trajectories"] == 1
    assert trainer.turn_credit_totals["extra_unmatched_model_turns"] == 1
    records = [
        json.loads(line)
        for line in trainer.turn_credit_alignment_audit_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(records) == 1
    audit = records[0]
    assert audit["task_id"] == "task-mismatch"
    assert audit["trajectory_id"] == "trajectory-mismatch"
    assert audit["trainer_batch_index"] == 0
    assert audit["batch_row_index"] == 0
    assert audit["model_span_count"] == 3
    assert audit["model_span_ranges"] == [[0, 1], [2, 3], [4, 5]]
    assert audit["credit_record_count"] == 1
    assert audit["credit_records"] == [{"credit": 0.8, "validity": "valid", "step_index": 0}]
    assert audit["extra_unmatched_model_spans"] == 1
    assert audit["mismatch_type"] == "extra_model_spans"
    assert audit["decoded_model_spans"][1]["decoded_text"] == "decoded:13"


def test_alignment_audit_decoding_is_token_and_character_bounded(tmp_path, monkeypatch):
    long_span = 70
    tool_mask = torch.tensor(
        [[*([1] * long_span), 0, 1, 0, 1], [1, 0, 1, *([0] * (long_span + 1))]]
    )
    decoded_token_lengths = []

    def decode(token_ids, **kwargs):
        decoded_token_lengths.append(len(token_ids))
        return "x" * 700

    trainer = _audit_trainer(
        monkeypatch,
        tmp_path,
        completion_ids=torch.arange(tool_mask.numel()).reshape(tool_mask.shape),
        tool_mask=tool_mask,
        decode=decode,
    )

    trainer._generate_and_score_completions([])

    audit = json.loads(
        trainer.turn_credit_alignment_audit_path.read_text(encoding="utf-8").splitlines()[0]
    )
    first_span = audit["decoded_model_spans"][0]
    assert decoded_token_lengths[0] == 64
    assert len(first_span["decoded_text"]) == 512
    assert first_span["token_count"] == long_span
    assert first_span["truncated"] is True
