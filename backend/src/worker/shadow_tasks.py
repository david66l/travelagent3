"""Celery execution boundary for side-effect-isolated Agent shadow runs."""

from __future__ import annotations

import logging
from typing import Any, cast

from agentic.integration import run_agent_branch
from core.celery_app import celery_app
from core.database import async_session_maker
from evaluation.agentic_eval import AgenticEvaluator
from repositories.agentic_evaluation import AgenticEvaluationRepository
from worker.memory_tasks import _ensure_redis, _run_async

logger = logging.getLogger(__name__)
TASK_NAME = "worker.shadow_tasks.execute_agent_shadow"


async def _execute_agent_shadow_async(
    scenario_id: str,
    input_hash: str,
    state: dict[str, Any],
) -> bool:
    try:
        await _ensure_redis()
        async with async_session_maker() as db:
            claimed = await AgenticEvaluationRepository(db).claim_agent(
                scenario_id=scenario_id,
                input_hash=input_hash,
            )
            await db.commit()
        if not claimed:
            logger.info("Agent shadow scenario %s was already claimed", scenario_id)
            return False
        patch = await run_agent_branch(state)
        episode = patch.get("agent_episode")
        if not isinstance(episode, dict):
            raise RuntimeError(
                f"Agent shadow did not produce a replayable episode: "
                f"{patch.get('termination_reason') or patch.get('agent_status')}"
            )
        run = AgenticEvaluator().from_agent_episode(scenario_id, episode)
        async with async_session_maker() as db:
            written = await AgenticEvaluationRepository(db).complete(
                scenario_id=scenario_id,
                mode="agent",
                input_hash=input_hash,
                metrics=run.model_dump(mode="json"),
                episode=episode,
            )
            await db.commit()
        return written
    except Exception as exc:
        logger.exception("Agent shadow scenario %s failed", scenario_id)
        async with async_session_maker() as db:
            await AgenticEvaluationRepository(db).fail_agent(
                scenario_id=scenario_id,
                error=f"{type(exc).__name__}: {exc}",
            )
            await db.commit()
        return False


@celery_app.task(name=TASK_NAME, acks_late=True)  # type: ignore[untyped-decorator]
def execute_agent_shadow(
    scenario_id: str,
    input_hash: str,
    state: dict[str, Any],
) -> bool:
    return cast(bool, _run_async(_execute_agent_shadow_async(scenario_id, input_hash, state)))
