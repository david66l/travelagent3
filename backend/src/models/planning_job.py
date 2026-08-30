"""Planning job models for async task execution."""

import uuid
from sqlalchemy import (
    Column,
    String,
    Integer,
    DateTime,
    Text,
    JSON,
    BigInteger,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from core.database import Base
from core.clock import utc_now_naive


class PlanningJob(Base):
    """Async planning job with lease-based worker coordination.

    This model is being evolved toward the PRD schema. Existing columns are
    preserved for backward compatibility while new PRD columns are added.

    Note: id is kept as String(36) in P1 to maintain compatibility with
    existing worker/pipeline code that serializes job_id via json.dumps.
    Will migrate to UUID in P6 (Celery refactoring).
    """

    __tablename__ = "planning_jobs"
    __table_args__ = (
        UniqueConstraint(
            "user_uuid",
            "idempotency_key",
            name="uq_planning_jobs_user_idempotency_key",
        ),
    )

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    # Legacy identifiers (kept for backward compatibility)
    session_id = Column(String(64), nullable=True, index=True)
    user_id = Column(String(64), nullable=True, index=True)
    user_input = Column(Text, nullable=True)
    idempotency_key = Column(String(128), nullable=True)

    # PRD-aligned identifiers
    user_uuid = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Legacy string conversation_id for backward compatibility
    conversation_id_str = Column(String(64), nullable=True, index=True)

    # Metadata label (worker polls all pending jobs; not a Celery queue name)
    queue_name = Column(
        String(50),
        nullable=False,
        default="default",
        index=True,
    )

    # Status machine
    status = Column(
        String(32),
        nullable=False,
        default="pending",
        index=True,
    )
    # pending / running / retrying / intent_ready / data_collected / strategy_ready /
    # draft_ready / critic_done / itinerary_final / writing /
    # completed / failed / cancelling / cancelled

    # PRD-aligned input/output
    input_requirements = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    token_usage = Column(JSON, nullable=True)
    latency_ms = Column(Integer, nullable=True)

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

    created_at = Column(DateTime, default=utc_now_naive)
    updated_at = Column(
        DateTime,
        default=utc_now_naive,
        onupdate=utc_now_naive,
    )
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="planning_jobs", foreign_keys=[user_uuid])
    conversation = relationship("Conversation", back_populates="planning_jobs")
    itineraries = relationship("Itinerary", back_populates="job")
    events = relationship(
        "PlanningJobEvent",
        back_populates="job",
        cascade="all, delete-orphan",
    )


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
    # PRD-aligned status field alongside legacy event_type
    status = Column(
        String(32),
        nullable=True,
    )  # running / completed / failed / cancelled
    event_type = Column(
        String(32),
        nullable=True,
    )  # started / completed / failed / cancelled (legacy)
    payload = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now_naive)

    job = relationship("PlanningJob", back_populates="events")
