from pathlib import Path

from scripts.build_stage27_ai_pilot_report import build


REPO_ROOT = Path(__file__).resolve().parents[4]


def test_stage27_pilot_report_preserves_claim_boundary_and_pairing():
    report = build(
        REPO_ROOT / "ml/agentic/reports",
        REPO_ROOT / "ml/agentic/datasets/external-benchmark-v1/ai-assisted-pilot-v1",
    )

    assert report["status"] == "passed_for_schema_calibration"
    assert report["eligible_for_external_claim"] is False
    assert report["arms"]["base_4b"]["successful_runs"] == 27
    assert report["arms"]["sft_4b"]["successful_runs"] == 26
    assert report["arms"]["dpo_4b"]["successful_runs"] == 25
    assert report["arms"]["teacher_8b"]["successful_runs"] == 30
    assert report["deterministic_router_replay"]["successful_cases"] == 30
    assert report["deterministic_router_replay"]["teacher_cases"] == 10
    assert report["diagnostic_input_ablation"]["metadata_only_dpo_success"] == "15/30"
