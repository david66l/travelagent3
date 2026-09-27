from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.compare_h005_same_stack as comparison


def _report(
    checkpoint: str,
    *,
    corpus_sha256: str,
    rows: list[dict[str, object]],
    task_count: int,
) -> dict[str, object]:
    successes = sum(
        bool((row.get("checks") or {}).get("system_step_valid"))  # type: ignore[union-attr]
        for row in rows
    )
    success_rate = successes / len(rows)
    contract = {
        "schema_version": "tradeoff-grounding-internal-eval.v7",
        "reason_assembly_version": "repair-reason-assembly.v1",
        "seed": 20260910,
        "seed_protocol": "sha256-task-sample-v1",
        "samples_per_task": 4,
        "temperature": 0.8,
        "max_new_tokens": 192,
        "load_in_4bit": True,
        "render_protocol": "render.v1",
        "chat_template_sha256": "template",
        "chat_template_kwargs": {"enable_thinking": False},
        "route_schema_sha256": {"abort": "schema"},
        "evaluation_code_snapshot": comparison.evaluation_code_snapshot(),
        "corpus_sha256": {"internal_dev": corpus_sha256},
        "evaluated_splits": ["internal_dev"],
        "runtime_versions": {"python": "test"},
    }
    assert set(contract) == set(comparison.CONTRACT_KEYS)
    return {
        **contract,
        "checkpoint": checkpoint,
        "checkpoint_adapter_sha256": hashlib.sha256(
            (Path(checkpoint) / "adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "effective_model_manifest": {
            "schema_version": "effective-model-manifest.v1",
            "model_files": comparison.model_file_manifest(checkpoint),
            "tokenizer": comparison.effective_tokenizer_manifest_for_checkpoint(checkpoint),
        },
        "production_promotion_eligible": False,
        "corpus_integrity": {
            "internal_dev": {
                "passed": True,
                "rows": task_count,
                "unique_task_ids": task_count,
                "unique_source_task_ids": task_count,
                "target_counts": {"abort": task_count},
                "targets_per_source": {"abort": 1},
            }
        },
        "splits": {
            "internal_dev": {
                "rollouts": len(rows),
                "raw_model_legacy_full_success_rate": success_rate,
                "raw_model_contract_success_rate": success_rate,
                "assembled_system_success_rate": success_rate,
                "action_accuracy": 1.0,
                "reason_assembly_exact_rate": 1.0,
            }
        },
    }


def _row(task: str, sample: int, *, checkpoint: str, success: bool) -> dict[str, object]:
    return {
        "split": "internal_dev",
        "task_id": task,
        "source_task_id": f"source-{task}",
        "sample_index": sample,
        "rollout_seed": comparison._expected_seed(20260910, task_id=task, sample_index=sample),
        "target_action": "abort",
        "observed_action": "abort",
        "observed_arguments": {"reason": "assembled"},
        "model_raw_arguments": {"reason": "raw"},
        "reason_assembly_version": "repair-reason-assembly.v1",
        "reason_assembly_exact": True,
        "initial_state_fingerprint": f"state-{task}",
        "authority_transport": comparison.AUTHORITY_TRANSPORT_VERSION,
        "inference_metrics": [
            {
                "model": checkpoint,
                "completion_tokens": 10,
                "request_latency_ms": 20,
            }
        ],
        "generation_audit": {
            "prompt_input_ids_sha256": hashlib.sha256(task.encode()).hexdigest(),
            "inference_metrics": {
                "model": checkpoint,
                "completion_tokens": 10,
                "request_latency_ms": 20,
            },
        },
        "reward": 1.0 if success else 0.5,
        "model_contract_compliant": True,
        "controller_override_attempt": False,
        "controller_hydration_exact": None,
        "checks": {
            "action_match": True,
            "decision_step_valid": success,
            "model_semantic_valid": success,
            "system_step_valid": success,
        },
    }


def _write_eval(
    directory: Path,
    report: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    directory.mkdir()
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / "rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _checkpoint(tmp_path: Path, label: str) -> tuple[str, str]:
    checkpoint = tmp_path / f"{label}-checkpoint"
    checkpoint.mkdir()
    adapter = checkpoint / "adapter_model.safetensors"
    adapter.write_bytes(label.encode())
    return str(checkpoint), hashlib.sha256(adapter.read_bytes()).hexdigest()


def _install_fake_corpus(
    monkeypatch: pytest.MonkeyPatch,
    corpus_path: Path,
    tasks: tuple[str, ...],
) -> str:
    corpus_path.write_text("fixture\n", encoding="utf-8")
    fake_rows = [
        SimpleNamespace(
            task=SimpleNamespace(task_id=task),
            snapshot=SimpleNamespace(),
            contract={"source_task_id": f"source-{task}", "target_action": "abort"},
        )
        for task in tasks
    ]
    monkeypatch.setattr(comparison, "load_grpo_corpus", lambda path: fake_rows)
    monkeypatch.setattr(comparison, "verifier_repair_metadata", lambda row: row.contract)
    monkeypatch.setattr(
        comparison,
        "environment_fingerprint",
        lambda task, snapshot: f"state-{task.task_id}",
    )
    monkeypatch.setattr(
        comparison,
        "model_file_manifest",
        lambda checkpoint: {
            "schema_version": "effective-model-files.v1",
            "checkpoint": str(Path(checkpoint).resolve()),
            "adapter": {"identity": Path(checkpoint).name},
            "base_model": {"identity": "shared-base"},
            "checkpoint_tokenizer": {"identity": Path(checkpoint).name},
        },
    )
    monkeypatch.setattr(
        comparison,
        "effective_tokenizer_manifest_for_checkpoint",
        lambda checkpoint: {"identity": "shared-effective-tokenizer"},
    )
    corpus_sha256 = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    monkeypatch.setattr(comparison, "EXPECTED_INTERNAL_DEV_CORPUS_SHA256", corpus_sha256)
    monkeypatch.setattr(comparison, "EXPECTED_TASKS", len(tasks))
    monkeypatch.setattr(comparison, "EXPECTED_SOURCES", len(tasks))
    monkeypatch.setattr(comparison, "EXPECTED_TARGET_COUNTS", {"abort": len(tasks)})
    monkeypatch.setattr(comparison, "EXPECTED_TARGETS_PER_SOURCE", {"abort": 1})
    return corpus_sha256


def test_compare_requires_and_reports_exact_pairing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    corpus_path = tmp_path / "corpus.jsonl"
    tasks = ("task-a", "task-b")
    corpus_sha256 = _install_fake_corpus(monkeypatch, corpus_path, tasks)
    baseline_checkpoint, baseline_sha256 = _checkpoint(tmp_path, "baseline")
    candidate_checkpoint, candidate_sha256 = _checkpoint(tmp_path, "candidate")
    baseline_rows = [
        _row(
            task,
            sample,
            checkpoint=baseline_checkpoint,
            success=not (task == "task-b" and sample == 1),
        )
        for task in tasks
        for sample in range(4)
    ]
    candidate_rows = [
        _row(task, sample, checkpoint=candidate_checkpoint, success=True)
        for task in tasks
        for sample in range(4)
    ]
    _write_eval(
        baseline,
        _report(
            baseline_checkpoint,
            corpus_sha256=corpus_sha256,
            rows=baseline_rows,
            task_count=len(tasks),
        ),
        baseline_rows,
    )
    _write_eval(
        candidate,
        _report(
            candidate_checkpoint,
            corpus_sha256=corpus_sha256,
            rows=candidate_rows,
            task_count=len(tasks),
        ),
        candidate_rows,
    )

    report = comparison.compare(
        corpus_path=corpus_path,
        baseline_dir=baseline,
        candidate_dir=candidate,
        expected_baseline_adapter_sha256=baseline_sha256,
        expected_candidate_adapter_sha256=candidate_sha256,
        output_path=tmp_path / "comparison.json",
        bootstrap_seed=1,
        bootstrap_rounds=100,
    )

    assert report["paired_rollouts"] == 8
    assert report["paired_tasks"] == 2
    assert report["baseline"]["assembled_system_success_rate"] == 0.875
    assert report["candidate"]["assembled_system_success_rate"] == 1.0
    assert report["paired_assembled_system"]["descriptive_rollout_mcnemar"]["candidate_only"] == 1
    assert report["baseline_validation"]["four_sample_indices_exact"] is True
    assert report["paired_by_target"]["abort"]["paired_tasks"] == 2
    assert (
        report["h005_safety_gate"]["baseline"]["checks"]["reason_assembly_coverage_100pct"] is True
    )


def test_compare_rejects_stack_drift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    corpus_path = tmp_path / "corpus.jsonl"
    tasks = ("task-a",)
    corpus_sha256 = _install_fake_corpus(monkeypatch, corpus_path, tasks)
    baseline_checkpoint, baseline_sha256 = _checkpoint(tmp_path, "baseline")
    candidate_checkpoint, candidate_sha256 = _checkpoint(tmp_path, "candidate")
    baseline_rows = [
        _row("task-a", sample, checkpoint=baseline_checkpoint, success=True) for sample in range(4)
    ]
    candidate_rows = [
        _row("task-a", sample, checkpoint=candidate_checkpoint, success=True) for sample in range(4)
    ]
    baseline_report = _report(
        baseline_checkpoint,
        corpus_sha256=corpus_sha256,
        rows=baseline_rows,
        task_count=1,
    )
    candidate_report = _report(
        candidate_checkpoint,
        corpus_sha256=corpus_sha256,
        rows=candidate_rows,
        task_count=1,
    )
    candidate_report["temperature"] = 0.7
    _write_eval(baseline, baseline_report, baseline_rows)
    _write_eval(candidate, candidate_report, candidate_rows)

    with pytest.raises(ValueError, match="temperature"):
        comparison.compare(
            corpus_path=corpus_path,
            baseline_dir=baseline,
            candidate_dir=candidate,
            expected_baseline_adapter_sha256=baseline_sha256,
            expected_candidate_adapter_sha256=candidate_sha256,
            output_path=tmp_path / "comparison.json",
            bootstrap_seed=1,
            bootstrap_rounds=10,
        )


def test_compare_rejects_non_four_sample_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    corpus_path = tmp_path / "corpus.jsonl"
    tasks = ("task-a",)
    corpus_sha256 = _install_fake_corpus(monkeypatch, corpus_path, tasks)
    baseline_checkpoint, baseline_sha256 = _checkpoint(tmp_path, "baseline")
    candidate_checkpoint, candidate_sha256 = _checkpoint(tmp_path, "candidate")
    baseline_rows = [
        _row("task-a", sample, checkpoint=baseline_checkpoint, success=True) for sample in range(4)
    ]
    candidate_rows = [
        _row("task-a", sample, checkpoint=candidate_checkpoint, success=True) for sample in range(4)
    ]
    baseline_report = _report(
        baseline_checkpoint,
        corpus_sha256=corpus_sha256,
        rows=baseline_rows,
        task_count=1,
    )
    candidate_report = _report(
        candidate_checkpoint,
        corpus_sha256=corpus_sha256,
        rows=candidate_rows,
        task_count=1,
    )
    baseline_report["samples_per_task"] = 2
    candidate_report["samples_per_task"] = 2
    _write_eval(baseline, baseline_report, baseline_rows)
    _write_eval(candidate, candidate_report, candidate_rows)

    with pytest.raises(ValueError, match="exactly 4"):
        comparison.compare(
            corpus_path=corpus_path,
            baseline_dir=baseline,
            candidate_dir=candidate,
            expected_baseline_adapter_sha256=baseline_sha256,
            expected_candidate_adapter_sha256=candidate_sha256,
            output_path=tmp_path / "comparison.json",
            bootstrap_seed=1,
            bootstrap_rounds=10,
        )


def test_compare_rejects_empty_cluster_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    corpus_path = tmp_path / "corpus.jsonl"
    tasks = ("task-a",)
    corpus_sha256 = _install_fake_corpus(monkeypatch, corpus_path, tasks)
    baseline_checkpoint, baseline_sha256 = _checkpoint(tmp_path, "baseline")
    candidate_checkpoint, candidate_sha256 = _checkpoint(tmp_path, "candidate")
    baseline_rows = [
        _row("task-a", sample, checkpoint=baseline_checkpoint, success=True) for sample in range(4)
    ]
    candidate_rows = [
        _row("task-a", sample, checkpoint=candidate_checkpoint, success=True) for sample in range(4)
    ]
    for row in baseline_rows + candidate_rows:
        row["source_task_id"] = ""
    baseline_report = _report(
        baseline_checkpoint,
        corpus_sha256=corpus_sha256,
        rows=baseline_rows,
        task_count=1,
    )
    candidate_report = _report(
        candidate_checkpoint,
        corpus_sha256=corpus_sha256,
        rows=candidate_rows,
        task_count=1,
    )
    _write_eval(baseline, baseline_report, baseline_rows)
    _write_eval(candidate, candidate_report, candidate_rows)

    with pytest.raises(ValueError, match="empty source_task_id"):
        comparison.compare(
            corpus_path=corpus_path,
            baseline_dir=baseline,
            candidate_dir=candidate,
            expected_baseline_adapter_sha256=baseline_sha256,
            expected_candidate_adapter_sha256=candidate_sha256,
            output_path=tmp_path / "comparison.json",
            bootstrap_seed=1,
            bootstrap_rounds=10,
        )


def test_compare_rejects_frozen_corpus_contract_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    corpus_path = tmp_path / "corpus.jsonl"
    tasks = ("task-a",)
    corpus_sha256 = _install_fake_corpus(monkeypatch, corpus_path, tasks)
    monkeypatch.setattr(
        comparison,
        "EXPECTED_TARGET_COUNTS",
        {"abort": 1, "propose_tradeoff": 1, "retry_solve": 1},
    )
    baseline_checkpoint, baseline_sha256 = _checkpoint(tmp_path, "baseline")
    candidate_checkpoint, candidate_sha256 = _checkpoint(tmp_path, "candidate")
    baseline_rows = [
        _row("task-a", sample, checkpoint=baseline_checkpoint, success=True) for sample in range(4)
    ]
    candidate_rows = [
        _row("task-a", sample, checkpoint=candidate_checkpoint, success=True) for sample in range(4)
    ]
    _write_eval(
        baseline,
        _report(
            baseline_checkpoint,
            corpus_sha256=corpus_sha256,
            rows=baseline_rows,
            task_count=1,
        ),
        baseline_rows,
    )
    _write_eval(
        candidate,
        _report(
            candidate_checkpoint,
            corpus_sha256=corpus_sha256,
            rows=candidate_rows,
            task_count=1,
        ),
        candidate_rows,
    )

    with pytest.raises(ValueError, match="frozen H-005 contract"):
        comparison.compare(
            corpus_path=corpus_path,
            baseline_dir=baseline,
            candidate_dir=candidate,
            expected_baseline_adapter_sha256=baseline_sha256,
            expected_candidate_adapter_sha256=candidate_sha256,
            output_path=tmp_path / "comparison.json",
            bootstrap_seed=1,
            bootstrap_rounds=10,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value", "error"),
    [
        ("authority_transport", "wrong-transport", "authority transport mismatch"),
        ("initial_state_fingerprint", "wrong-state", "state fingerprint"),
        ("request_latency_ms", float("nan"), "invalid request_latency_ms"),
    ],
)
def test_compare_rejects_row_identity_or_cost_audit_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid_value: object,
    error: str,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    corpus_path = tmp_path / "corpus.jsonl"
    tasks = ("task-a",)
    corpus_sha256 = _install_fake_corpus(monkeypatch, corpus_path, tasks)
    baseline_checkpoint, baseline_sha256 = _checkpoint(tmp_path, "baseline")
    candidate_checkpoint, candidate_sha256 = _checkpoint(tmp_path, "candidate")
    baseline_rows = [
        _row("task-a", sample, checkpoint=baseline_checkpoint, success=True) for sample in range(4)
    ]
    candidate_rows = [
        _row("task-a", sample, checkpoint=candidate_checkpoint, success=True) for sample in range(4)
    ]
    if field == "request_latency_ms":
        baseline_rows[0]["inference_metrics"][0][field] = invalid_value  # type: ignore[index]
    else:
        baseline_rows[0][field] = invalid_value
    _write_eval(
        baseline,
        _report(
            baseline_checkpoint,
            corpus_sha256=corpus_sha256,
            rows=baseline_rows,
            task_count=1,
        ),
        baseline_rows,
    )
    _write_eval(
        candidate,
        _report(
            candidate_checkpoint,
            corpus_sha256=corpus_sha256,
            rows=candidate_rows,
            task_count=1,
        ),
        candidate_rows,
    )

    with pytest.raises(ValueError, match=error):
        comparison.compare(
            corpus_path=corpus_path,
            baseline_dir=baseline,
            candidate_dir=candidate,
            expected_baseline_adapter_sha256=baseline_sha256,
            expected_candidate_adapter_sha256=candidate_sha256,
            output_path=tmp_path / "comparison.json",
            bootstrap_seed=1,
            bootstrap_rounds=10,
        )


def test_compare_rejects_adapter_file_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    corpus_path = tmp_path / "corpus.jsonl"
    tasks = ("task-a",)
    corpus_sha256 = _install_fake_corpus(monkeypatch, corpus_path, tasks)
    baseline_checkpoint, baseline_sha256 = _checkpoint(tmp_path, "baseline")
    candidate_checkpoint, candidate_sha256 = _checkpoint(tmp_path, "candidate")
    baseline_rows = [
        _row("task-a", sample, checkpoint=baseline_checkpoint, success=True) for sample in range(4)
    ]
    candidate_rows = [
        _row("task-a", sample, checkpoint=candidate_checkpoint, success=True) for sample in range(4)
    ]
    _write_eval(
        baseline,
        _report(
            baseline_checkpoint,
            corpus_sha256=corpus_sha256,
            rows=baseline_rows,
            task_count=1,
        ),
        baseline_rows,
    )
    _write_eval(
        candidate,
        _report(
            candidate_checkpoint,
            corpus_sha256=corpus_sha256,
            rows=candidate_rows,
            task_count=1,
        ),
        candidate_rows,
    )
    (Path(candidate_checkpoint) / "adapter_model.safetensors").write_bytes(b"replaced")

    with pytest.raises(ValueError, match="candidate adapter file"):
        comparison.compare(
            corpus_path=corpus_path,
            baseline_dir=baseline,
            candidate_dir=candidate,
            expected_baseline_adapter_sha256=baseline_sha256,
            expected_candidate_adapter_sha256=candidate_sha256,
            output_path=tmp_path / "comparison.json",
            bootstrap_seed=1,
            bootstrap_rounds=10,
        )


@pytest.mark.parametrize(
    ("drift", "error"),
    [
        ("code", "candidate evaluator code snapshot"),
        ("prompt", "prompt_input_ids_sha256_exact"),
    ],
)
def test_compare_rejects_code_or_prompt_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
    error: str,
) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    corpus_path = tmp_path / "corpus.jsonl"
    tasks = ("task-a",)
    corpus_sha256 = _install_fake_corpus(monkeypatch, corpus_path, tasks)
    baseline_checkpoint, baseline_sha256 = _checkpoint(tmp_path, "baseline")
    candidate_checkpoint, candidate_sha256 = _checkpoint(tmp_path, "candidate")
    baseline_rows = [
        _row("task-a", sample, checkpoint=baseline_checkpoint, success=True) for sample in range(4)
    ]
    candidate_rows = [
        _row("task-a", sample, checkpoint=candidate_checkpoint, success=True) for sample in range(4)
    ]
    baseline_report = _report(
        baseline_checkpoint,
        corpus_sha256=corpus_sha256,
        rows=baseline_rows,
        task_count=1,
    )
    candidate_report = _report(
        candidate_checkpoint,
        corpus_sha256=corpus_sha256,
        rows=candidate_rows,
        task_count=1,
    )
    if drift == "code":
        candidate_report["evaluation_code_snapshot"] = {
            **candidate_report["evaluation_code_snapshot"],  # type: ignore[dict-item]
            "combined_sha256": "0" * 64,
        }
    else:
        candidate_rows[0]["generation_audit"]["prompt_input_ids_sha256"] = (  # type: ignore[index]
            "a" * 64
        )
    _write_eval(baseline, baseline_report, baseline_rows)
    _write_eval(candidate, candidate_report, candidate_rows)

    with pytest.raises(ValueError, match=error):
        comparison.compare(
            corpus_path=corpus_path,
            baseline_dir=baseline,
            candidate_dir=candidate,
            expected_baseline_adapter_sha256=baseline_sha256,
            expected_candidate_adapter_sha256=candidate_sha256,
            output_path=tmp_path / "comparison.json",
            bootstrap_seed=1,
            bootstrap_rounds=10,
        )
