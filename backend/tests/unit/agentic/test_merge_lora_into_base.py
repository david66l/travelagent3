import json

import pytest

from ml.agentic.training.merge_lora_into_base import (
    copy_base_tokenizer_files,
    load_merge_contract,
    merge,
)


def test_static_merge_contract_requires_trained_lora_and_refuses_overwrite(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "/models/base"}),
        encoding="utf-8",
    )
    (adapter / "training_report.json").write_text(
        json.dumps({"status": "trained", "run_scope": "formal", "dataset_version": "sft-v1"}),
        encoding="utf-8",
    )
    output = tmp_path / "merged"

    contract = load_merge_contract(adapter, output)

    assert contract["base_model"] == "/models/base"
    assert contract["run_scope"] == "formal"
    output.mkdir()
    with pytest.raises(ValueError, match="overwrite"):
        load_merge_contract(adapter, output)


def test_static_merge_copies_base_tokenizer_bytes_without_reserializing(tmp_path):
    source = tmp_path / "base"
    output = tmp_path / "merged"
    source.mkdir()
    output.mkdir()
    original = b'{"extra_special_tokens":{"tool":"<tool>"}}\n'
    (source / "tokenizer_config.json").write_bytes(original)
    (source / "tokenizer.json").write_bytes(b"opaque-tokenizer-bytes")

    copied = copy_base_tokenizer_files(source, output)

    assert copied == ["tokenizer.json", "tokenizer_config.json"]
    assert (output / "tokenizer_config.json").read_bytes() == original
    assert (output / "tokenizer.json").read_bytes() == b"opaque-tokenizer-bytes"


def test_static_merge_rejects_unsafe_adapter_scale_before_loading_model(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "/models/base"}),
        encoding="utf-8",
    )
    (adapter / "training_report.json").write_text(
        json.dumps({"status": "trained", "run_scope": "formal"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="adapter scale"):
        merge(adapter, tmp_path / "merged", adapter_scale=0)
