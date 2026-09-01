"""Durable paired-evaluation records for deterministic and Agent policies."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from core.clock import utc_now_naive

from core.database import Base


class AgenticEvaluationRecord(Base):
    """One side of a deterministic/Agent comparison for the same scenario."""

    __tablename__ = "agentic_evaluation_records"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    scenario_id: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    evaluation_source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="live_shadow"
    )
    deployment_id: Mapped[str] = mapped_column(String(64), nullable=False, default="local")
    batch_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source_case_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    release_gate_eligible: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    input_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    input_snapshot: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    episode: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=utc_now_naive
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("scenario_id", "mode", name="uq_agentic_eval_scenario_mode"),
        Index("ix_agentic_eval_status_created", "status", "created_at"),
        Index(
            "ix_agentic_eval_provenance",
            "evaluation_source",
            "deployment_id",
            "batch_id",
            "status",
        ),
    )
