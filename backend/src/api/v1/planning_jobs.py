"""Planning job v1 API endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.chat_runtime import enqueue_planning_job, schedule_cancel_enforcement
from api.deps import get_current_user, get_db, get_planning_job_service
from api.v1.schemas import (
    CreatePlanningJobRequest,
    PlanningJobResponse,
    UpdatePlanningJobRequest,
)
from core.exceptions import ForbiddenException, NotFoundException
from core.redis_client import redis_client
from core.responses import success_response
from models import User
from services import PlanningJobService

router = APIRouter(prefix="/planning-jobs", tags=["planning-jobs"])


@router.post("")
async def create_planning_job(
    body: CreatePlanningJobRequest,
    user: User = Depends(get_current_user),
    service: PlanningJobService = Depends(get_planning_job_service),
):
    """Create a new planning job."""
    job = await service.create(
        user_uuid=user.id,
        conversation_id=body.conversation_id,
        queue_name=body.queue_name,
        input_requirements=body.input_requirements,
    )
    enqueue_planning_job(job.id)
    return success_response(
        data=PlanningJobResponse.model_validate(job).model_dump(),
        status_code=201,
    )


@router.get("/{job_id}")
async def get_planning_job(
    job_id: str,
    user: User = Depends(get_current_user),
    service: PlanningJobService = Depends(get_planning_job_service),
):
    """Get a planning job by ID."""
    job = await service.get(job_id)
    if job is None or (job.user_uuid != user.id and job.user_id != str(user.id)):
        raise NotFoundException("PlanningJob", job_id)
    return success_response(data=PlanningJobResponse.model_validate(job).model_dump())


@router.patch("/{job_id}/status")
async def update_planning_job_status(
    job_id: str,
    body: UpdatePlanningJobRequest,
    user: User = Depends(get_current_user),
    service: PlanningJobService = Depends(get_planning_job_service),
):
    """Update planning job status (admin/worker use in P6)."""
    job = await service.get(job_id)
    if job is None or (job.user_uuid != user.id and job.user_id != str(user.id)):
        raise NotFoundException("PlanningJob", job_id)
    await service.update_status(job_id, body.status)
    return success_response(message="Status updated")


@router.post("/{job_id}/cancel")
async def cancel_planning_job(
    job_id: str,
    user: User = Depends(get_current_user),
    service: PlanningJobService = Depends(get_planning_job_service),
    db: AsyncSession = Depends(get_db),
):
    """Request job cancellation (PRD §4.7)."""
    ok = await service.request_cancel(job_id, user.id)
    if not ok:
        job = await service.get(job_id)
        if job is None:
            raise NotFoundException("PlanningJob", job_id)
        raise ForbiddenException("Job cannot be cancelled", code="CANCEL_NOT_ALLOWED")
    await db.commit()
    await redis_client._client.publish(f"job:cancel:{job_id}", "cancel")
    schedule_cancel_enforcement(job_id)
    return success_response(message="Cancel requested", status_code=202)
