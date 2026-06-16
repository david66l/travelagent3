"""Archived dead-letter records (PRD §4.7.4)."""

import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text, JSON

from core.database import Base


class DeadLetterArchive(Base):
    """Long-term storage for stale planning dead-letter messages."""

    __tablename__ = "dead_letter_archive"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    task_id = Column(String(64), nullable=False, index=True)
    task_name = Column(String(128), nullable=False)
    job_id = Column(String(36), nullable=True, index=True)
    payload = Column(JSON, nullable=False, default=dict)
    exception_type = Column(String(256), nullable=True)
    traceback = Column(Text, nullable=True)
    failed_at = Column(DateTime, nullable=True)
    archived_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
