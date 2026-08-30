import json

from ml.agentic.training.train_dpo import (
    install_frozen_sft_reference_adapter,
    preflight_model,
    select_stratified_rows,
    to_dpo_rows,
    validate_preference_dataset,
)


def _row(index: int, family: str, split: str) -> dict:
    return {
        "pair_id": f"pair-{split}-{family}-{index}",
        "family": family,
        "context_hash": f"context-{split}-{family}-{index}",
        "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "go"},
        ],
        "tools": [{"type": "function", "function": {"name": "act", "parameters": {}}}],
        "chosen": {"role": "assistant", "content": "good"},
        "rejected": {"role": "assistant", "content": "bad"},
        "reason_codes": ["VERIFIER_SUCCESS_OVER_FAILURE"],
    }


def test_dpo_preflight_and_conversion_preserve_tools_and_completion_boundary(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "dataset_version": "prefs-v1",
                "requires_verifier_success_over_failure": True,
            }
        ),
        encoding="utf-8",
    )
    for split in ("train", "validation", "test"):
        rows = [_row(0, "search", split), _row(0, "clarification", split)]
        (tmp_path / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )

    report = validate_preference_dataset(tmp_path, minimum_train_examples=2)
    converted = to_dpo_rows([_row(0, "search", "train")])[0]

    assert report["ready"] is True
    assert report["unique_pairs"] == 6
    assert converted["prompt"][-1]["role"] == "user"
    assert converted["chosen"] == [{"role": "assistant", "content": "good"}]
    assert converted["tools"][0]["function"]["name"] == "act"


def test_dpo_smoke_selection_is_family_stratified():
    rows = [_row(i, "search", "train") for i in range(10)] + [
        _row(i, "tradeoff", "train") for i in range(2)
    ]

    selected = select_stratified_rows(rows, 4)

    assert {row["family"] for row in selected} == {"search", "tradeoff"}


def test_dpo_preflight_rejects_unverified_pair(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"status": "passed", "requires_verifier_success_over_failure": True}),
        encoding="utf-8",
    )
    for split in ("train", "validation", "test"):
        row = _row(0, "search", split)
        if split == "train":
            row["reason_codes"] = []
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = validate_preference_dataset(tmp_path, minimum_train_examples=1)

    assert report["ready"] is False
    assert any(error.startswith("UNVERIFIED_PAIR") for error in report["errors"])


def test_dpo_preflight_accepts_mechanically_verified_duplicate_call_pair(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "preference_evidence_policy": (
                    "verifier_success_or_deterministic_single_action_contract"
                ),
            }
        ),
        encoding="utf-8",
    )
    for split in ("train", "validation", "test"):
        row = _row(0, "single_action_abort", split)
        call = {
            "type": "function",
            "function": {"name": "abort", "arguments": {"reason": "unsafe"}},
        }
        row["chosen"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [call],
        }
        row["rejected"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [call, call],
        }
        row["reason_codes"] = ["SINGLE_ACTION_CONTRACT_OVER_DUPLICATE_CALL"]
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = validate_preference_dataset(tmp_path, minimum_train_examples=1)

    assert report["ready"] is True


def test_dpo_preflight_accepts_deterministic_decision_boundary_pairs(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "preference_evidence_policy": (
                    "verifier_success_or_deterministic_decision_boundary_contract"
                ),
            }
        ),
        encoding="utf-8",
    )
    for split in ("train", "validation", "test"):
        row = _row(0, "decision_boundary_abort", split)
        row["messages"][1]["content"] = json.dumps(
            {"capability": {"actionable_alternatives": False}}
        )
        row["chosen"] = {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "abort", "arguments": {}}}],
        }
        row["rejected"] = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "propose_tradeoff", "arguments": {}},
                }
            ],
        }
        row["reason_codes"] = ["DECISION_BOUNDARY_CONTRACT_OVER_OPPOSITE_ACTION"]
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = validate_preference_dataset(tmp_path, minimum_train_examples=1)

    assert report["ready"] is True


def test_dpo_preflight_rejects_reversed_decision_boundary_pair(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "preference_evidence_policy": (
                    "verifier_success_or_deterministic_decision_boundary_contract"
                ),
            }
        ),
        encoding="utf-8",
    )
    for split in ("train", "validation", "test"):
        row = _row(0, "decision_boundary_reversed", split)
        row["messages"][1]["content"] = json.dumps(
            {"capability": {"actionable_alternatives": False}}
        )
        row["chosen"] = {
            "role": "assistant",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "propose_tradeoff", "arguments": {}},
                }
            ],
        }
        row["rejected"] = {
            "role": "assistant",
            "tool_calls": [{"type": "function", "function": {"name": "abort", "arguments": {}}}],
        }
        row["reason_codes"] = ["DECISION_BOUNDARY_CONTRACT_OVER_OPPOSITE_ACTION"]
        (tmp_path / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = validate_preference_dataset(tmp_path, minimum_train_examples=1)

    assert report["ready"] is False
    assert any(error.startswith("UNVERIFIED_PAIR") for error in report["errors"])


def test_dpo_model_preflight_accepts_batch_encoding_style_tokenizer_output():
    class Tokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            prompt = [10, 11, 12]
            return {"input_ids": prompt if add_generation_prompt else [*prompt, 20, 21]}

    report = preflight_model([_row(0, "search", "train")], Tokenizer(), max_length=8)

    assert report == {
        "ready": True,
        "pairs_checked": 1,
        "sequences_checked": 2,
        "max_sequence_tokens": 5,
        "p95_sequence_tokens": 5,
        "over_max_length": 0,
        "prefix_errors": [],
    }


def test_dpo_installs_explicit_frozen_sft_reference_adapter(tmp_path):
    class Model:
        peft_config = {"default": object()}
        active_adapter = "default"

        def load_adapter(self, path, *, adapter_name, is_trainable):
            assert path == str(tmp_path)
            assert adapter_name == "ref"
            assert is_trainable is False
            self.peft_config[adapter_name] = object()

        def set_adapter(self, adapter_name):
            self.active_adapter = adapter_name

    model = Model()

    identity = install_frozen_sft_reference_adapter(model, tmp_path)

    assert identity == "frozen-sft-adapter:ref"
    assert model.active_adapter == "default"
    assert "ref" in model.peft_config
