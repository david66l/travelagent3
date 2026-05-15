"""Planning job models for async task execution."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    Text,
    JSON,
    BigInteger,
    ForeignKey,
)

from core.database import Base


class PlanningJob(Base):
    """Async planning job with lease-based worker coordination."""

    __tablename__ = "planning_jobs"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    session_id = Column(String(64), nullable=False, index=True)
    user_id = Column(String(64), nullable=False)
    user_input = Column(Text, nullable=False)

    # Status machine
    status = Column(
        String(32),
        nullable=False,
        default="pending",
        index=True,
    )
    # pending / running / intent_ready / data_collected / strategy_ready /
    # draft_ready / critic_done / itinerary_final / writing /
    # completed / failed / cancelling / cancelled

    # Stage timings: {stage_name: elapsed_seconds}
    stage_timings = Column(JSON, default=dict)

    # Intermediate results (JSON snapshots)
    intent_result = Column(JSON, nullable=True)
    strategy = Column(JSON, nullable=True)
    itinerary_draft = Column(JSON, nullable=True)
    itinerary_final = Column(JSON, nullable=True)
    proposal_text = Column(Text, nullable=True)

    # Error info
    error_message = Column(Text, nullable=True)
    error_stage = Column(String(32), nullable=True)

    # Worker lease (prevents permanent stuck jobs on worker crash)
    locked_by = Column(String(64), nullable=True)
    lock_expires_at = Column(DateTime, nullable=True)
    heartbeat_at = Column(DateTime, nullable=True)
    attempt_count = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    last_error = Column(Text, nullable=True)

    # User interaction
    user_feedback = Column(JSON, default=dict)
    version = Column(Integer, default=1)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
    completed_at = Column(DateTime, nullable=True)


class PlanningJobEvent(Base):
    """Audit log for each pipeline stage transition."""

    __tablename__ = "planning_job_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id = Column(
        String(36),
        ForeignKey("planning_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stage = Column(String(32), nullable=False)
    event_type = Column(
        String(32),
        nullable=False,
    )  # started / completed / failed / cancelled
    payload = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
