"""Admin endpoint for paired Agentic shadow release-gate reports."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import get_db, require_admin
from core.responses import success_response
from core.settings import settings
from evaluation.agentic_eval import AgenticEvaluator, ReleaseGateConfig
from models import User
from repositories.agentic_evaluation import AgenticEvaluationRepository

router = APIRouter(prefix="/admin/agentic-evaluation", tags=["admin"])


@router.get("/report")
async def get_agentic_evaluation_report(
    limit: int = Query(default=1000, ge=1, le=10000),
    minimum_paired_scenarios: int = Query(default=300, ge=1),
    evaluation_source: Literal["live_shadow", "authorized_replay", "synthetic_smoke"] = Query(
        default="live_shadow"
    ),
    deployment_id: str = Query(default=settings.agentic_deployment_id, min_length=1),
    batch_id: str | None = Query(default=None, min_length=1),
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Compare completed deterministic/Agent pairs using deterministic gates."""
    deterministic, agent = await AgenticEvaluationRepository(db).completed_pairs(
        limit=limit,
        evaluation_source=evaluation_source,
        deployment_id=deployment_id,
        batch_id=batch_id,
        release_gate_eligible=True,
    )
    if not deterministic:
        return success_response(
            data={
                "paired_scenarios": 0,
                "release_eligible": False,
                "reason": "NO_COMPLETED_PAIRS",
                "evaluation_source": evaluation_source,
                "deployment_id": deployment_id,
                "batch_id": batch_id,
            }
        )
    report = AgenticEvaluator().compare(
        deterministic,
        agent,
        gate=ReleaseGateConfig(minimum_paired_scenarios=minimum_paired_scenarios),
    )
    payload = report.model_dump(mode="json")
    payload.update(
        {
            "evaluation_source": evaluation_source,
            "deployment_id": deployment_id,
            "batch_id": batch_id,
            "quality_gates_passed": report.release_eligible,
            "canary_evidence": evaluation_source == "live_shadow",
        }
    )
    if evaluation_source != "live_shadow":
        payload["release_eligible"] = False
        payload["checks"].append(
            {
                "code": "LIVE_SHADOW_EVIDENCE",
                "passed": False,
                "actual": 0,
                "expected": "evaluation_source=live_shadow",
            }
        )
    return success_response(data=payload)
