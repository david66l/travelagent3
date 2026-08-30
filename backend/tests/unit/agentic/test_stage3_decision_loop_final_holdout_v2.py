import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from agentic.environment import environment_fingerprint
from agentic.grpo_training import load_grpo_corpus


SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CURRICULUM = _load_script("build_stage3_decision_loop_curriculum_v3")
HOLDOUT = _load_script("build_stage3_decision_loop_final_holdout_v2")


def _failure_messages(rows):
    return {
        str(response.fallback_reason)
        for row in rows
        for response in row.snapshot.tool_responses.get("search_pois") or []
        if response.fallback_reason
    }


def test_final_holdout_v2_is_orthogonal_and_leakage_safe(tmp_path: Path):
    development = tmp_path / "development"
    CURRICULUM.build(
        development,
        start_index=101000,
        train_count=64,
        validation_count=32,
        test_count=32,
    )
    output = tmp_path / "holdout"
    manifest = HOLDOUT.build(
        output,
        development_dir=development,
        start_index=130000,
        count=128,
    )

    assert manifest["count"] == 128
    assert set(manifest["coverage"]) == {
        "change_arguments/diagnostic_evidence",
        "change_arguments/explicit_instruction",
        "retry_same_arguments/diagnostic_evidence",
        "retry_same_arguments/explicit_instruction",
    }
    for facts in manifest["coverage"].values():
        assert facts["tasks"] == 32
        assert len(facts["cities"]) == 4
        assert set(facts["cities"].values()) == {8}
        assert facts["target_positions"] == {"0": 16, "1": 16}
        assert len(facts["template_ids"]) == 4
        assert set(facts["template_ids"].values()) == {8}
    assert all(not values for values in manifest["leakage"].values())

    development_rows = [
        row
        for split in ("train", "validation", "test")
        for row in load_grpo_corpus(development / f"{split}.jsonl")
    ]
    rows = load_grpo_corpus(output / "test.jsonl")
    assert len(rows) == 128
    assert len({row.task.task_id for row in rows}) == 128
    assert len({environment_fingerprint(row.task, row.snapshot) for row in rows}) == 128
    assert _failure_messages(rows).isdisjoint(_failure_messages(development_rows))
    assert all(row.snapshot.environment_version == HOLDOUT.SCHEMA_VERSION for row in rows)
    assert (
        hashlib.sha256((output / "test.jsonl").read_bytes()).hexdigest() == manifest["test_sha256"]
    )


def test_final_holdout_v2_manifest_is_deterministic_and_rejects_small_counts(
    tmp_path: Path,
):
    development = tmp_path / "development"
    CURRICULUM.build(
        development,
        start_index=102000,
        train_count=64,
        validation_count=32,
        test_count=32,
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = HOLDOUT.build(
        first,
        development_dir=development,
        start_index=140000,
    )
    second_manifest = HOLDOUT.build(
        second,
        development_dir=development,
        start_index=140000,
    )

    assert first_manifest["test_sha256"] == second_manifest["test_sha256"]
    assert (
        json.loads((first / "manifest.json").read_text(encoding="utf-8"))["coverage"]
        == first_manifest["coverage"]
    )
    with pytest.raises(ValueError, match="multiple of 128"):
        HOLDOUT.build(
            tmp_path / "invalid",
            development_dir=development,
            start_index=150000,
            count=64,
        )
