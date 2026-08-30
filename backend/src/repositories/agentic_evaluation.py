"""Persistence operations for paired Agentic shadow evaluations."""

from __future__ import annotations

from typing import Any

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.clock import utc_now_naive
from models.agentic_evaluation import AgenticEvaluationRecord


class AgenticEvaluationRepository:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_pending_agent(
        self,
        *,
        scenario_id: str,
        input_hash: str,
        input_snapshot: dict[str, Any],
        evaluation_source: str = "live_shadow",
        deployment_id: str = "local",
        batch_id: str | None = None,
        source_case_id: str | None = None,
        release_gate_eligible: bool = False,
    ) -> bool:
        """Create an Agent-side pending row once; return whether it was inserted."""
        statement = (
            insert(AgenticEvaluationRecord)
            .values(
                scenario_id=scenario_id,
                mode="agent",
                status="pending",
                evaluation_source=evaluation_source,
                deployment_id=deployment_id,
                batch_id=batch_id,
                source_case_id=source_case_id,
                release_gate_eligible=release_gate_eligible,
                input_hash=input_hash,
                input_snapshot=input_snapshot,
            )
            .on_conflict_do_nothing(constraint="uq_agentic_eval_scenario_mode")
        )
        result = await self.db.execute(statement)
        return bool(result.rowcount)

    async def complete(
        self,
        *,
        scenario_id: str,
        mode: str,
        input_hash: str,
        metrics: dict[str, Any],
        input_snapshot: dict[str, Any] | None = None,
        episode: dict[str, Any] | None = None,
        evaluation_source: str = "live_shadow",
        deployment_id: str = "local",
        batch_id: str | None = None,
        source_case_id: str | None = None,
        release_gate_eligible: bool = False,
    ) -> bool:
        """Write a terminal result once so later confirm turns cannot rewrite a pair."""
        statement = (
            insert(AgenticEvaluationRecord)
            .values(
                scenario_id=scenario_id,
                mode=mode,
                status="completed",
                evaluation_source=evaluation_source,
                deployment_id=deployment_id,
                batch_id=batch_id,
                source_case_id=source_case_id,
                release_gate_eligible=release_gate_eligible,
                input_hash=input_hash,
                input_snapshot=input_snapshot,
                metrics=metrics,
                episode=episode,
                completed_at=utc_now_naive(),
            )
            .on_conflict_do_update(
                constraint="uq_agentic_eval_scenario_mode",
                set_={
                    "status": "completed",
                    "metrics": metrics,
                    "episode": episode,
                    "error": None,
                    "completed_at": utc_now_naive(),
                },
                where=AgenticEvaluationRecord.status.in_(("pending", "running")),
            )
        )
        result = await self.db.execute(statement)
        return bool(result.rowcount)

    async def claim_agent(self, *, scenario_id: str, input_hash: str) -> bool:
        """Atomically claim a pending shadow run and reject duplicate delivery."""
        result = await self.db.execute(
            update(AgenticEvaluationRecord)
            .where(
                AgenticEvaluationRecord.scenario_id == scenario_id,
                AgenticEvaluationRecord.mode == "agent",
                AgenticEvaluationRecord.status == "pending",
                AgenticEvaluationRecord.input_hash == input_hash,
            )
            .values(status="running")
        )
        return bool(result.rowcount)

    async def fail_agent(self, *, scenario_id: str, error: str) -> bool:
        record = await self.get(scenario_id=scenario_id, mode="agent")
        if record is None or record.status not in {"pending", "running"}:
            return False
        record.status = "failed"
        record.error = error[:4000]
        record.completed_at = utc_now_naive()
        await self.db.flush()
        return True

    async def get(self, *, scenario_id: str, mode: str) -> AgenticEvaluationRecord | None:
        result = await self.db.execute(
            select(AgenticEvaluationRecord).where(
                AgenticEvaluationRecord.scenario_id == scenario_id,
                AgenticEvaluationRecord.mode == mode,
            )
        )
        return result.scalar_one_or_none()

    async def completed_runs(self, *, mode: str, limit: int = 1000) -> list[dict[str, Any]]:
        result = await self.db.execute(
            select(AgenticEvaluationRecord)
            .where(
                AgenticEvaluationRecord.mode == mode,
                AgenticEvaluationRecord.status == "completed",
            )
            .order_by(AgenticEvaluationRecord.created_at.desc())
            .limit(limit)
        )
        return [record.metrics for record in result.scalars() if record.metrics]

    async def completed_pairs(
        self,
        *,
        limit: int = 1000,
        evaluation_source: str | None = None,
        deployment_id: str | None = None,
        batch_id: str | None = None,
        release_gate_eligible: bool | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return only complete, hash- and provenance-matched scenario pairs."""
        deterministic = AgenticEvaluationRecord
        conditions = [
            deterministic.mode == "deterministic",
            deterministic.status == "completed",
            deterministic.metrics.is_not(None),
        ]
        if evaluation_source is not None:
            conditions.append(deterministic.evaluation_source == evaluation_source)
        if deployment_id is not None:
            conditions.append(deterministic.deployment_id == deployment_id)
        if batch_id is not None:
            conditions.append(deterministic.batch_id == batch_id)
        if release_gate_eligible is not None:
            conditions.append(deterministic.release_gate_eligible == release_gate_eligible)
        result = await self.db.execute(
            select(deterministic)
            .where(*conditions)
            .order_by(deterministic.created_at.desc())
            .limit(limit)
        )
        deterministic_rows = list(result.scalars())
        scenario_ids = [row.scenario_id for row in deterministic_rows]
        if not scenario_ids:
            return [], []
        agent_result = await self.db.execute(
            select(AgenticEvaluationRecord).where(
                AgenticEvaluationRecord.mode == "agent",
                AgenticEvaluationRecord.status == "completed",
                AgenticEvaluationRecord.metrics.is_not(None),
                AgenticEvaluationRecord.scenario_id.in_(scenario_ids),
            )
        )
        agents_by_id = {row.scenario_id: row for row in agent_result.scalars()}
        matched = []
        for deterministic_row in deterministic_rows:
            agent_row = agents_by_id.get(deterministic_row.scenario_id)
            if agent_row is None:
                continue
            if not (
                agent_row.input_hash == deterministic_row.input_hash
                and agent_row.evaluation_source == deterministic_row.evaluation_source
                and agent_row.deployment_id == deterministic_row.deployment_id
                and agent_row.batch_id == deterministic_row.batch_id
                and agent_row.source_case_id == deterministic_row.source_case_id
                and agent_row.release_gate_eligible == deterministic_row.release_gate_eligible
            ):
                continue
            matched.append((deterministic_row, agent_row))
        deterministic_runs = [row.metrics for row, _ in matched]
        agent_runs = [row.metrics for _, row in matched]
        return deterministic_runs, agent_runs
