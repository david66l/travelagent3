from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from evaluation.posttraining_promotion_protocol import LineageRecord, canonical_hash


SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "audit_h006_data_lineage.py"
SPEC = importlib.util.spec_from_file_location("audit_h006_data_lineage", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(task_id: str, source_id: str, request: str, marker: str) -> dict:
    tools = {"search": [{"marker": marker}]}
    lineage = LineageRecord(
        record_id=task_id,
        split="train-hard-donor",
        source_state_id=source_id,
        source_hash=canonical_hash({"source": source_id}),
        template_lineage="h006:new-case",
        tool_snapshot_hash=canonical_hash(tools),
        generator_model="deterministic",
        generator_prompt_hash=canonical_hash(request),
        parent_trajectory_id=source_id,
    )
    return {
        "task": {"task_id": task_id},
        "snapshot": {
            "state_id": marker,
            "tool_responses": tools,
            "hidden_test_facts": {
                "training_lineage": lineage.model_dump(mode="json"),
                "grpo_decision_state": {
                    "source_task_id": source_id,
                    "prompt_messages": [
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "policy_state": {
                                        "original_request": request,
                                        "relevant_artifacts": [],
                                        "capability": {"alternatives": []},
                                    }
                                },
                                ensure_ascii=False,
                            ),
                        }
                    ],
                },
            },
        },
    }


def test_audit_allows_fresh_donor_but_blocks_legacy_overlap(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidate"
    candidate = _row("candidate-1", "fresh-source", "全新的广州旅行约束", "fresh")
    _write(candidate_dir / "train.jsonl", [candidate])
    _write(candidate_dir / "validation.jsonl", [])
    _write(
        candidate_dir / "lineage.jsonl",
        [candidate["snapshot"]["hidden_test_facts"]["training_lineage"]],
    )
    forbidden = tmp_path / "forbidden.jsonl"
    _write(forbidden, [_row("old-hard", "old-source", "旧的北京冲突题", "old")])
    legacy = tmp_path / "h001.jsonl"
    _write(legacy, [{"trajectory_id": "old-source"}])

    report = MODULE.audit(candidate_dir, legacy, [forbidden])

    assert report["train_only_donor_rollout_authorized"] is True
    assert report["h006_gpu_training_authorized"] is False
    assert report["legacy_h001"]["direct_payload_reuse_allowed"] is False
    assert report["counts"]["candidate_forbidden_source_overlaps"] == 0
    assert "protected_hash_registry_complete" in report["blocking_reasons"]


def test_audit_rejects_candidate_source_overlap(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "candidate"
    candidate = _row("candidate-1", "shared-source", "同一条旅行请求", "fresh")
    _write(candidate_dir / "train.jsonl", [candidate])
    _write(candidate_dir / "validation.jsonl", [])
    _write(
        candidate_dir / "lineage.jsonl",
        [candidate["snapshot"]["hidden_test_facts"]["training_lineage"]],
    )
    forbidden = tmp_path / "forbidden.jsonl"
    _write(forbidden, [_row("old-hard", "shared-source", "另一条请求", "old")])
    legacy = tmp_path / "h001.jsonl"
    _write(legacy, [{"trajectory_id": "shared-source"}])

    report = MODULE.audit(candidate_dir, legacy, [forbidden])

    assert report["train_only_donor_rollout_authorized"] is False
    assert report["gates"]["forbidden_source_id_overlap_zero"] is False
