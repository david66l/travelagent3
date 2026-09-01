"""Planning request log ORM model."""

import uuid
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID

from core.database import Base
from core.clock import utc_now


class PlanningLog(Base):
    """每次规划请求的输入输出、状态、冲突原因与修改次数。"""

    __tablename__ = "planning_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    request_input = Column(JSONB, default=dict, nullable=False)
    output_itinerary = Column(JSONB, default=dict, nullable=False)
    status = Column(String(20), nullable=False, default="pending")  # pending / success / failed
    conflict_reason = Column(Text, nullable=True)
    modification_count = Column(Integer, nullable=False, default=0)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
