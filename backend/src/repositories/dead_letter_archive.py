"""Repository for dead_letter_archive table."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.dead_letter_archive import DeadLetterArchive


class DeadLetterArchiveRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_from_dlq(self, data: dict[str, Any]) -> DeadLetterArchive:
        failed_at_raw = data.get("failed_at")
        failed_at: Optional[datetime] = None
        if failed_at_raw:
            try:
                failed_at = datetime.fromisoformat(str(failed_at_raw).replace("Z", "+00:00"))
            except ValueError:
                failed_at = None

        row = DeadLetterArchive(
            task_id=str(data.get("task_id", "")),
            task_name=str(data.get("task_name", "")),
            job_id=data.get("job_id"),
            payload=data,
            exception_type=data.get("exception"),
            traceback=data.get("traceback"),
            failed_at=failed_at,
        )
        self.db.add(row)
        await self.db.flush()
        await self.db.refresh(row)
        return row

    async def list_recent(self, limit: int = 50) -> list[DeadLetterArchive]:
        result = await self.db.execute(
            select(DeadLetterArchive).order_by(DeadLetterArchive.archived_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
