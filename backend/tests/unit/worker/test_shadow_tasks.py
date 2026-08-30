"""Tests for idempotent Agent shadow Celery execution."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic.runtime import initialize_agent_ledger
from worker import shadow_tasks


def _session() -> MagicMock:
    session = MagicMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.commit = AsyncMock()
    return session


@pytest.mark.asyncio
async def test_duplicate_shadow_delivery_is_not_executed_twice():
    repo = MagicMock()
    repo.claim_agent = AsyncMock(return_value=False)
    session = _session()
    with (
        patch("worker.shadow_tasks.async_session_maker", return_value=session),
        patch("worker.shadow_tasks.AgenticEvaluationRepository", return_value=repo),
        patch("worker.shadow_tasks.run_agent_branch", new_callable=AsyncMock) as run,
        patch("worker.shadow_tasks._ensure_redis", new=AsyncMock()),
    ):
        result = await shadow_tasks._execute_agent_shadow_async(
            "scenario-1", "hash-1", {"policy_mode": "shadow"}
        )

    assert result is False
    run.assert_not_awaited()


@pytest.mark.asyncio
async def test_claimed_shadow_records_agent_metrics_and_episode():
    state = initialize_agent_ledger(
        {"user_input": "Plan Shanghai", "slots": {"destination": "Shanghai"}},
        mode="shadow",
    )
    episode = {"trajectory_id": "unused"}
    run_metric = MagicMock()
    run_metric.model_dump.return_value = {"scenario_id": "scenario-1", "mode": "agent"}
    repo = MagicMock()
    repo.claim_agent = AsyncMock(return_value=True)
    repo.complete = AsyncMock(return_value=True)
    session = _session()
    with (
        patch("worker.shadow_tasks.async_session_maker", return_value=session),
        patch("worker.shadow_tasks.AgenticEvaluationRepository", return_value=repo),
        patch("worker.shadow_tasks._ensure_redis", new=AsyncMock()),
        patch(
            "worker.shadow_tasks.run_agent_branch",
            new=AsyncMock(return_value={"agent_episode": episode}),
        ),
        patch(
            "worker.shadow_tasks.AgenticEvaluator.from_agent_episode",
            return_value=run_metric,
        ),
    ):
        result = await shadow_tasks._execute_agent_shadow_async("scenario-1", "hash-1", state)

    assert result is True
    repo.complete.assert_awaited_once()
    assert repo.complete.await_args.kwargs["episode"] == episode
