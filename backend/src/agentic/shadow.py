"""Isolated shadow-run coordination and deterministic/Agent pair recording."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from agentic.trajectory import redact_pii
from core.database import async_session_maker
from core.settings import settings
from evaluation.agentic_eval import AgenticEvaluator
from repositories.agentic_evaluation import AgenticEvaluationRepository

logger = logging.getLogger(__name__)


class ShadowProvenance(BaseModel):
    """Auditable origin metadata shared by both sides of one evaluation pair."""

    evaluation_source: Literal["live_shadow", "authorized_replay", "synthetic_smoke"] = (
        "live_shadow"
    )
    deployment_id: str = Field(min_length=1, max_length=64)
    batch_id: str | None = Field(default=None, max_length=64)
    source_case_id: str | None = Field(default=None, max_length=128)
    release_gate_eligible: bool = False


def default_shadow_provenance() -> ShadowProvenance:
    return ShadowProvenance(
        evaluation_source="live_shadow",
        deployment_id=settings.agentic_deployment_id,
        release_gate_eligible=True,
    )


# Deliberately excludes session/job channels, messages, attachments, booking,
# confirmation and output fields. A shadow copy can observe planning inputs but
# cannot publish tokens, mutate a checkpoint, write memory, or book anything.
_SHADOW_INPUT_KEYS = {
    "user_input",
    "slots",
    "profile",
    "preference_vector",
    "inferred_slots",
    "feasibility_report",
    "missing_slots",
    "poi_candidates",
    "knowledge_results",
    "weather",
    "weather_fetched",
    "agent_ledger",
    "current_task_id",
}


def training_partition_key(state: dict[str, Any]) -> str:
    """Return a non-reversible user partition for dataset split isolation."""
    raw = str(state.get("user_id") or state.get("session_id") or "anonymous")
    material = f"{settings.privacy_encryption_key}:{raw}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def project_shadow_input(state: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe, PII-redacted, side-effect-free state projection."""
    projected = {key: state[key] for key in _SHADOW_INPUT_KEYS if key in state}
    projected["policy_mode"] = "shadow"
    safe = redact_pii(projected)
    # Round-trip through JSON to reject live clients, sets and model instances.
    return json.loads(json.dumps(safe, ensure_ascii=False, default=str))


def shadow_input_hash(snapshot: dict[str, Any]) -> str:
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def shadow_scenario_id(state: dict[str, Any]) -> str | None:
    ledger = state.get("agent_ledger") or {}
    trajectory_id = ledger.get("trajectory_id") if isinstance(ledger, dict) else None
    return str(trajectory_id) if trajectory_id else None


def should_sample_shadow(scenario_id: str) -> bool:
    """Use stable hash sampling so retries always make the same decision."""
    rate = settings.agentic_shadow_sample_rate
    if rate >= 1:
        return True
    if rate <= 0:
        return False
    bucket = int(hashlib.sha256(scenario_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < rate


async def start_shadow_run(
    state: dict[str, Any],
    *,
    provenance: ShadowProvenance | None = None,
    force_sample: bool = False,
) -> dict[str, Any]:
    """Persist then enqueue an isolated Agent run without affecting user routing."""
    provenance = provenance or default_shadow_provenance()
    if force_sample and provenance.evaluation_source == "live_shadow":
        raise ValueError("force_sample is reserved for labeled replay or smoke runs")
    scenario_id = shadow_scenario_id(state)
    if not scenario_id or (not force_sample and not should_sample_shadow(scenario_id)):
        return {"shadow_status": "not_sampled"}
    if state.get("shadow_scenario_id") == scenario_id:
        return {"shadow_scenario_id": scenario_id, "shadow_status": "already_started"}

    snapshot = project_shadow_input(state)
    snapshot["_training_partition"] = training_partition_key(state)
    snapshot["_evaluation_provenance"] = provenance.model_dump(mode="json")
    input_hash = shadow_input_hash(snapshot)
    try:
        async with async_session_maker() as db:
            repo = AgenticEvaluationRepository(db)
            inserted = await repo.create_pending_agent(
                scenario_id=scenario_id,
                input_hash=input_hash,
                input_snapshot=snapshot,
                **provenance.model_dump(mode="python"),
            )
            await db.commit()
    except Exception as exc:
        # Shadow telemetry is intentionally fail-open: it may never degrade the
        # authoritative deterministic response.
        logger.exception("Could not persist Agent shadow scenario %s", scenario_id)
        return {
            "shadow_scenario_id": scenario_id,
            "shadow_input_hash": input_hash,
            "shadow_status": f"persistence_failed:{type(exc).__name__}",
        }
    if not inserted:
        return {
            "shadow_scenario_id": scenario_id,
            "shadow_input_hash": input_hash,
            "shadow_status": "already_started",
        }

    try:
        from worker.shadow_tasks import execute_agent_shadow

        execute_agent_shadow.apply_async(
            args=[scenario_id, input_hash, snapshot],
            queue=settings.celery_shadow_queue,
            # The database row is the authoritative Shadow result. Avoid
            # opening a Redis result-consumer subscription in the API process.
            ignore_result=True,
        )
    except Exception as exc:
        logger.exception("Could not enqueue Agent shadow scenario %s", scenario_id)
        async with async_session_maker() as db:
            await AgenticEvaluationRepository(db).fail_agent(
                scenario_id=scenario_id,
                error=f"enqueue failed: {type(exc).__name__}: {exc}",
            )
            await db.commit()
        return {
            "shadow_scenario_id": scenario_id,
            "shadow_input_hash": input_hash,
            "shadow_status": "enqueue_failed",
        }
    return {
        "shadow_scenario_id": scenario_id,
        "shadow_input_hash": input_hash,
        "shadow_status": "running",
    }


async def record_deterministic_shadow_result(state: dict[str, Any], *, latency_ms: int) -> bool:
    """Record the authoritative draft once, paired to its isolated Agent run."""
    if state.get("policy_mode") != "shadow":
        return False
    scenario_id = str(state.get("shadow_scenario_id") or "")
    if not scenario_id:
        return False
    snapshot = project_shadow_input(state)
    input_hash = str(state.get("shadow_input_hash") or shadow_input_hash(snapshot))
    tool_calls = (
        int(bool(state.get("poi_candidates")))
        + int(bool(state.get("weather_fetched")))
        + int(bool(state.get("itinerary") or state.get("solve_status")))
        + len(state.get("tool_results") or [])
    )
    run = AgenticEvaluator().from_deterministic_state(
        scenario_id,
        state,
        latency_ms=latency_ms,
        tool_calls=tool_calls,
    )
    async with async_session_maker() as db:
        repo = AgenticEvaluationRepository(db)
        agent_record = await repo.get(scenario_id=scenario_id, mode="agent")
        if agent_record is not None:
            input_hash = agent_record.input_hash
            snapshot = agent_record.input_snapshot or snapshot
            provenance = ShadowProvenance(
                evaluation_source=agent_record.evaluation_source,
                deployment_id=agent_record.deployment_id,
                batch_id=agent_record.batch_id,
                source_case_id=agent_record.source_case_id,
                release_gate_eligible=agent_record.release_gate_eligible,
            )
        else:
            provenance = default_shadow_provenance()
        written = await repo.complete(
            scenario_id=scenario_id,
            mode="deterministic",
            input_hash=input_hash,
            input_snapshot=snapshot,
            metrics=run.model_dump(mode="json"),
            **provenance.model_dump(mode="python"),
        )
        await db.commit()
    return written
