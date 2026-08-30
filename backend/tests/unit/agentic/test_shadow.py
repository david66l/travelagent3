"""Shadow execution must be paired, isolated, sampled and fail-open."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentic.runtime import initialize_agent_ledger
from agentic.shadow import (
    ShadowProvenance,
    default_shadow_provenance,
    project_shadow_input,
    should_sample_shadow,
    start_shadow_run,
    training_partition_key,
)


def _state() -> dict:
    initialized = initialize_agent_ledger(
        {
            "user_input": "Plan Shanghai for one day, call 13800138000",
            "user_id": "private-user",
            "session_id": "private-session",
            "job_id": "private-job",
            "messages": [{"role": "user", "content": "private history"}],
            "slots": {"destination": "Shanghai", "travel_days": 1},
            "profile": {"destination": "Shanghai", "travel_days": 1},
        },
        mode="shadow",
    )
    return {
        "user_input": "Plan Shanghai for one day, call 13800138000",
        "user_id": "private-user",
        "session_id": "private-session",
        "job_id": "private-job",
        "messages": [{"role": "user", "content": "private history"}],
        "slots": {"destination": "Shanghai", "travel_days": 1},
        "profile": {"destination": "Shanghai", "travel_days": 1},
        **initialized,
    }


def test_shadow_projection_excludes_side_effect_channels_and_redacts_pii():
    projected = project_shadow_input(_state())

    assert projected["policy_mode"] == "shadow"
    assert "session_id" not in projected
    assert "job_id" not in projected
    assert "messages" not in projected
    assert "user_id" not in projected
    assert "[REDACTED_PHONE]" in projected["user_input"]


def test_shadow_sampling_is_stable(monkeypatch):
    monkeypatch.setattr("agentic.shadow.settings.agentic_shadow_sample_rate", 0.5)
    assert should_sample_shadow("scenario-a") == should_sample_shadow("scenario-a")


def test_training_partition_is_stable_and_not_a_raw_identifier():
    first = training_partition_key({"user_id": "user-123"})
    second = training_partition_key({"user_id": "user-123"})

    assert first == second
    assert "user-123" not in first


def test_default_shadow_provenance_is_live_and_release_eligible(monkeypatch):
    monkeypatch.setattr("agentic.shadow.settings.agentic_deployment_id", "stage31-test")

    provenance = default_shadow_provenance()

    assert provenance.evaluation_source == "live_shadow"
    assert provenance.deployment_id == "stage31-test"
    assert provenance.release_gate_eligible is True


@pytest.mark.asyncio
async def test_force_sample_rejects_live_shadow():
    with pytest.raises(ValueError, match="reserved"):
        await start_shadow_run(_state(), force_sample=True)


@pytest.mark.asyncio
async def test_shadow_persistence_failure_does_not_break_authoritative_path():
    broken_session = MagicMock()
    broken_session.__aenter__ = AsyncMock(side_effect=RuntimeError("db offline"))
    broken_session.__aexit__ = AsyncMock(return_value=False)

    with patch("agentic.shadow.async_session_maker", return_value=broken_session):
        result = await start_shadow_run(_state())

    assert result["shadow_status"] == "persistence_failed:RuntimeError"
    assert result["shadow_scenario_id"]


@pytest.mark.asyncio
async def test_shadow_enqueue_uses_database_record_instead_of_celery_result_backend(
    monkeypatch,
):
    monkeypatch.setattr("agentic.shadow.settings.agentic_shadow_sample_rate", 1.0)
    session = AsyncMock()
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=False)
    repo = MagicMock()
    repo.create_pending_agent = AsyncMock(return_value=True)
    task = MagicMock()

    with (
        patch("agentic.shadow.async_session_maker", return_value=session_context),
        patch("agentic.shadow.AgenticEvaluationRepository", return_value=repo),
        patch("worker.shadow_tasks.execute_agent_shadow", task),
    ):
        result = await start_shadow_run(_state())

    assert result["shadow_status"] == "running"
    assert task.apply_async.call_args.kwargs["queue"] == "shadow"
    assert task.apply_async.call_args.kwargs["ignore_result"] is True
    create_call = repo.create_pending_agent.await_args.kwargs
    assert create_call["evaluation_source"] == "live_shadow"
    assert create_call["release_gate_eligible"] is True


@pytest.mark.asyncio
async def test_labeled_replay_can_force_sampling_without_changing_global_rate(
    monkeypatch,
):
    monkeypatch.setattr("agentic.shadow.settings.agentic_shadow_sample_rate", 0.0)
    session = AsyncMock()
    session_context = MagicMock()
    session_context.__aenter__ = AsyncMock(return_value=session)
    session_context.__aexit__ = AsyncMock(return_value=False)
    repo = MagicMock()
    repo.create_pending_agent = AsyncMock(return_value=True)
    task = MagicMock()
    provenance = ShadowProvenance(
        evaluation_source="authorized_replay",
        deployment_id="stage31-test",
        batch_id="batch-1",
        source_case_id="case-1",
        release_gate_eligible=True,
    )

    with (
        patch("agentic.shadow.async_session_maker", return_value=session_context),
        patch("agentic.shadow.AgenticEvaluationRepository", return_value=repo),
        patch("worker.shadow_tasks.execute_agent_shadow", task),
    ):
        result = await start_shadow_run(_state(), provenance=provenance, force_sample=True)

    assert result["shadow_status"] == "running"
    call = repo.create_pending_agent.await_args.kwargs
    assert call["evaluation_source"] == "authorized_replay"
    assert call["batch_id"] == "batch-1"
    assert call["source_case_id"] == "case-1"
