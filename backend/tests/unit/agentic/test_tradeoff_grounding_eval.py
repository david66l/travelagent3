import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic.chat_template_contract import (
    AGENT_CHAT_TEMPLATE_KWARGS,
    AGENT_CHAT_TEMPLATE_SHA256,
    AGENT_RENDER_PROTOCOL_VERSION,
)
from scripts.evaluate_tradeoff_grounding_sft import (
    _checkpoint_adapter_sha256,
    _full_success,
    _load_selected_corpora,
    _runtime_versions,
    _sha256,
    _shadow_claim_preflight,
    _validate_corpus,
)


def test_checkpoint_adapter_sha256_supports_base_and_adapter_checkpoints(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    assert _checkpoint_adapter_sha256(base) is None

    adapter = base / "adapter_model.safetensors"
    adapter.write_bytes(b"adapter-weights")
    assert _checkpoint_adapter_sha256(base) == _sha256(adapter)


def test_partial_positive_reward_is_not_a_full_success():
    assert not _full_success(
        {"reward": 0.714286, "checks": {"decision_step_valid": False}}
    )
    assert _full_success(
        {
            "reward": 1.0,
            "checks": {"decision_step_valid": True},
            "model_contract_compliant": True,
            "controller_override_attempt": False,
            "target_action": "abort",
        }
    )


def test_internal_dev_selection_never_opens_unselected_shadow(monkeypatch, tmp_path):
    internal = tmp_path / "internal.jsonl"
    poison_shadow = tmp_path / "must-not-be-opened.jsonl"
    opened = []

    def fake_load(path):
        opened.append(path)
        return ["internal-row"]

    monkeypatch.setattr(
        "scripts.evaluate_tradeoff_grounding_sft.load_grpo_corpus",
        fake_load,
    )
    paths, corpora = _load_selected_corpora(
        Namespace(
            splits=["internal_dev"],
            internal_dev_corpus=internal,
            train_shadow_corpus=poison_shadow,
        )
    )

    assert paths == {"internal_dev": internal}
    assert corpora == {"internal_dev": ["internal-row"]}
    assert opened == [internal]


def test_selected_split_requires_only_its_own_path(monkeypatch, tmp_path):
    internal = tmp_path / "internal.jsonl"
    monkeypatch.setattr(
        "scripts.evaluate_tradeoff_grounding_sft.load_grpo_corpus",
        lambda path: [Path(path).name],
    )

    paths, corpora = _load_selected_corpora(
        Namespace(
            splits=["internal_dev"],
            internal_dev_corpus=internal,
            train_shadow_corpus=None,
        )
    )

    assert paths == {"internal_dev": internal}
    assert corpora == {"internal_dev": ["internal.jsonl"]}


def test_internal_corpus_accepts_source_balanced_six_tradeoff_profile(monkeypatch):
    rows = []
    for source_index in range(10):
        for target, count in (("abort", 1), ("propose_tradeoff", 6), ("retry_solve", 1)):
            for copy_index in range(count):
                rows.append(
                    SimpleNamespace(
                        task=SimpleNamespace(
                            task_id=f"task-{source_index}-{target}-{copy_index}"
                        ),
                        contract={
                            "source_task_id": f"source-{source_index}",
                            "target_action": target,
                        },
                    )
                )
    monkeypatch.setattr(
        "scripts.evaluate_tradeoff_grounding_sft.verifier_repair_metadata",
        lambda row: row.contract,
    )

    report = _validate_corpus("internal_dev", rows)

    assert report["rows"] == 80
    assert report["target_counts"] == {
        "abort": 10,
        "propose_tradeoff": 60,
        "retry_solve": 10,
    }
    assert report["targets_per_source"] == {
        "abort": 1,
        "propose_tradeoff": 6,
        "retry_solve": 1,
    }


def test_shadow_lock_rejects_adapter_sha_mismatch(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"candidate")
    shadow = tmp_path / "shadow.jsonl"
    shadow.write_text("shadow\n", encoding="utf-8")
    internal_report = tmp_path / "internal-report.json"
    internal_report.write_text("{}\n", encoding="utf-8")
    selection = {
        "evidence": {
            "epoch1": {
                "report_path": str(internal_report),
            }
        }
    }
    selection_path = tmp_path / "selection_report.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    route_hashes = {"propose_tradeoff": "schema-sha"}
    lock = {
        "schema_version": "tradeoff-grounding-candidate-lock.v1",
        "status": "locked_internal_dev_only",
        "selected_label": "epoch1",
        "checkpoint": str(checkpoint),
        "checkpoint_adapter_sha256": "0" * 64,
        "selection_report_sha256": _sha256(selection_path),
        "internal_eval_report_sha256": _sha256(internal_report),
        "authorized_shadow_corpus_sha256": _sha256(shadow),
        "evaluation_contract": {
            "seed": 20260910,
            "seed_protocol": "sha256-task-sample-v1",
            "samples_per_task": 1,
            "temperature": 0.8,
            "max_new_tokens": 192,
            "load_in_4bit": True,
            "render_protocol": AGENT_RENDER_PROTOCOL_VERSION,
            "chat_template_sha256": AGENT_CHAT_TEMPLATE_SHA256,
            "chat_template_kwargs": dict(AGENT_CHAT_TEMPLATE_KWARGS),
            "route_schema_sha256": route_hashes,
            "runtime_versions": _runtime_versions(),
        },
        "shadow_claim_file": str(tmp_path / "shadow_eval.claim.json"),
    }
    lock_path = tmp_path / "candidate_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    args = Namespace(
        splits=["train_shadow"],
        candidate_lock=lock_path,
        checkpoint=str(checkpoint),
        seed=20260910,
        samples_per_task=1,
        temperature=0.8,
        max_new_tokens=192,
        load_in_4bit=True,
        output_dir=tmp_path / "shadow-output",
    )

    with pytest.raises(ValueError, match="adapter SHA"):
        _shadow_claim_preflight(
            args,
            corpus_paths={"train_shadow": shadow},
            route_schema_hashes=route_hashes,
        )
