"""Regression tests for the Stage36 boundary-SFT package."""

from pathlib import Path

from agentic.sft_dataset import DatasetManifest
from scripts.build_stage36_boundary_sft import build


REPO_ROOT = Path(__file__).resolve().parents[4]
SOURCE_DIR = (
    REPO_ROOT
    / "ml"
    / "agentic"
    / "datasets"
    / "qwen3-stage35-isolated-action-preferences-v1"
    / "sft_replay"
)


def test_stage36_dataset_preserves_audited_isolation_and_split_counts(tmp_path):
    report = build(SOURCE_DIR, tmp_path)
    manifest = DatasetManifest.model_validate_json(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )

    assert report["status"] == "passed"
    assert report["split_counts"] == {"train": 192, "validation": 24, "test": 24}
    assert report["action_counts"] == {
        "abort": 60,
        "ask_user": 60,
        "propose_tradeoff": 60,
        "search_pois": 60,
    }
    assert report["unique_model_visible_payloads"] == 240
    assert report["scenario_split_overlap"] == 0
    assert report["frozen_holdout_payload_overlap"] == 0
    assert manifest.dataset_version == report["dataset_version"]
    assert manifest.split_group_overlap is False
