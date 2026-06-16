"""Admin endpoints for planning dead-letter queue (PRD §4.7.4)."""

from fastapi import APIRouter, Depends

from api.deps import require_admin
from core.dead_letter import list_dead_letters, remove_dead_letter_by_task_id
from core.exceptions import NotFoundException
from core.responses import success_response
from models import User
from worker.dlq_tasks import retry_dead_letter

router = APIRouter(prefix="/dead-letters", tags=["dead-letters"])


@router.get("")
async def list_planning_dead_letters(
    limit: int = 50,
    _: User = Depends(require_admin),
):
    """List pending dead-letter entries (admin)."""
    items = await list_dead_letters(limit=limit)
    return success_response(data={"items": items, "count": len(items)})


@router.post("/{task_id}/retry")
async def retry_planning_dead_letter(
    task_id: str,
    job_id: str,
    _: User = Depends(require_admin),
):
    """Remove from DLQ and re-dispatch planning job."""
    retry_dead_letter.delay(task_id, job_id)
    return success_response(message="Retry dispatched")


@router.post("/{task_id}/dismiss")
async def dismiss_planning_dead_letter(
    task_id: str,
    _: User = Depends(require_admin),
):
    """Remove dead-letter entry without retry."""
    removed = await remove_dead_letter_by_task_id(task_id)
    if not removed:
        raise NotFoundException("DeadLetter", task_id)
    return success_response(message="Dismissed")
