"""Tests for verifier-labelled status-bridge SFT data."""

from pathlib import Path

from agentic.sft_dataset import DatasetManifest, SFTExample
from scripts.build_status_bridge_sft_dataset import build


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_status_bridge_is_verified_balanced_and_split_safe(tmp_path):
    report = build(
        REPO_ROOT
        / "ml"
        / "agentic"
        / "datasets"
        / "stage2-decision-boundary-grpo-v2-status-balanced",
        REPO_ROOT / "ml" / "agentic" / "datasets" / "stage2-sft-v2-curriculum-v1",
        tmp_path,
        train_pair_quotas={"infeasible": 1, "missing_tool": 1, "unsafe": 1},
        eval_pair_quotas={"infeasible": 1, "missing_tool": 1, "unsafe": 1},
        train_replay=2,
        eval_replay=2,
    )

    assert report["split_counts"] == {"train": 8, "validation": 8, "test": 8}
    assert report["verified_boundary_examples"] == 18
    assert report["scenario_split_overlap"] == 0
    assert report["action_counts"] == {
        "abort": 9,
        "propose_tradeoff": 9,
        "ask_user": 3,
        "search_pois": 3,
    }
    manifest = DatasetManifest.model_validate_json(
        (tmp_path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.split_group_overlap is False
    for split in ("train", "validation", "test"):
        examples = [
            SFTExample.model_validate_json(line)
            for line in (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(examples) == 8
