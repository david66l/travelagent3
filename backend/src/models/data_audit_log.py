"""Data audit log ORM model."""

import uuid
from sqlalchemy import Column, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import UUID

from core.database import Base
from core.clock import utc_now


class DataAuditLog(Base):
    """数据质量审计日志，记录疑似脏数据。"""

    __tablename__ = "data_audit_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    table_name = Column(String(50), nullable=False)
    record_id = Column(UUID(as_uuid=True), nullable=True)
    field = Column(String(50), nullable=False)
    reported_value = Column(Text, nullable=True)
    status = Column(String(20), nullable=False, default="pending")
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    __table_args__ = (Index("ix_data_audit_log_status", "status"),)
