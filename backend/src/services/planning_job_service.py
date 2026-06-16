"""Planning job application service."""

from typing import Optional
from uuid import UUID

from models import PlanningJob
from repositories.v1 import ConversationRepository, PlanningJobRepository
from services.base import BaseService


class PlanningJobService(BaseService):
    """Service for planning job orchestration."""

    def __init__(
        self,
        repo: PlanningJobRepository,
        conversation_repo: ConversationRepository,
    ):
        self.repo = repo
        self.conversation_repo = conversation_repo

    async def create(
        self,
        *,
        user_uuid: Optional[UUID] = None,
        conversation_id: Optional[UUID] = None,
        queue_name: str = "default",
        input_requirements: Optional[dict] = None,
    ) -> PlanningJob:
        """Create a new planning job."""
        if conversation_id is not None:
            conversation = await self.conversation_repo.get_by_id(conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation {conversation_id} not found")
        return await self.repo.create_api(
            user_uuid=user_uuid,
            conversation_id=conversation_id,
            queue_name=queue_name,
            input_requirements=input_requirements,
        )

    async def get(self, job_id: str) -> Optional[PlanningJob]:
        """Fetch a planning job by ID."""
        return await self.repo.get_by_id(job_id)

    async def update_status(self, job_id: str, status: str) -> bool:
        """Update the status of a planning job."""
        rowcount = await self.repo.update_status(job_id, status)
        return rowcount > 0

    async def update_result(
        self,
        job_id: str,
        *,
        result: Optional[dict] = None,
        token_usage: Optional[dict] = None,
        latency_ms: Optional[int] = None,
    ) -> bool:
        """Update the result of a completed planning job."""
        return await self.repo.update_result(
            job_id,
            result=result,
            token_usage=token_usage,
            latency_ms=latency_ms,
        )
