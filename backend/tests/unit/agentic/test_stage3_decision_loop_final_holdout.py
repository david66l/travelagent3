import importlib.util
from pathlib import Path

from agentic.grpo_training import load_grpo_corpus


SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_script("build_stage3_decision_loop_curriculum")
HOLDOUT = _load_script("build_stage3_decision_loop_final_holdout")


def test_final_holdout_is_balanced_and_disjoint(tmp_path: Path):
    development = tmp_path / "development"
    BUILDER.build(
        development,
        start_index=75000,
        train_count=16,
        validation_count=8,
        test_count=8,
    )
    output = tmp_path / "holdout"
    manifest = HOLDOUT.build(
        output,
        development_dir=development,
        start_index=95000,
        count=8,
    )

    assert manifest["count"] == 8
    assert set(manifest["strata"].values()) == {2}
    assert manifest["development_task_overlap"] == []
    assert manifest["development_environment_overlap"] == []
    assert manifest["development_failure_message_overlap"] == []
    rows = load_grpo_corpus(output / "test.jsonl")
    assert all(row.snapshot.environment_version == HOLDOUT.SCHEMA_VERSION for row in rows)
