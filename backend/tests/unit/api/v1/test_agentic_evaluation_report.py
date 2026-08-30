import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.v1.agentic_evaluation import get_agentic_evaluation_report


@pytest.mark.asyncio
async def test_report_filters_current_live_shadow_by_default():
    repo = MagicMock()
    repo.completed_pairs = AsyncMock(return_value=([], []))
    with patch("api.v1.agentic_evaluation.AgenticEvaluationRepository", return_value=repo):
        response = await get_agentic_evaluation_report(
            limit=1000,
            minimum_paired_scenarios=300,
            evaluation_source="live_shadow",
            deployment_id="stage31-test",
            batch_id=None,
            db=MagicMock(),
            admin=MagicMock(),
        )

    call = repo.completed_pairs.await_args.kwargs
    assert call["evaluation_source"] == "live_shadow"
    assert call["deployment_id"] == "stage31-test"
    assert call["release_gate_eligible"] is True
    assert json.loads(response.body)["data"]["paired_scenarios"] == 0


@pytest.mark.asyncio
async def test_authorized_replay_can_pass_quality_but_not_canary_gate():
    repo = MagicMock()
    repo.completed_pairs = AsyncMock(
        return_value=([{"mode": "deterministic"}], [{"mode": "agent"}])
    )
    comparison = MagicMock(release_eligible=True)
    comparison.model_dump.return_value = {
        "paired_scenarios": 300,
        "release_eligible": True,
        "checks": [],
    }
    evaluator = MagicMock()
    evaluator.compare.return_value = comparison
    with (
        patch(
            "api.v1.agentic_evaluation.AgenticEvaluationRepository",
            return_value=repo,
        ),
        patch("api.v1.agentic_evaluation.AgenticEvaluator", return_value=evaluator),
    ):
        response = await get_agentic_evaluation_report(
            limit=1000,
            minimum_paired_scenarios=300,
            evaluation_source="authorized_replay",
            deployment_id="stage31-test",
            batch_id="batch-1",
            db=MagicMock(),
            admin=MagicMock(),
        )

    data = json.loads(response.body)["data"]
    assert data["quality_gates_passed"] is True
    assert data["canary_evidence"] is False
    assert data["release_eligible"] is False
    assert data["checks"][-1]["code"] == "LIVE_SHADOW_EVIDENCE"
