"""User feedback / itinerary modification log ORM model."""

import uuid
from sqlalchemy import Column, DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB, UUID

from core.database import Base
from core.clock import utc_now


class UserModificationLog(Base):
    """用户对行程的反馈动作记录，用于 Human-in-the-loop 闭环。"""

    __tablename__ = "user_modification_log"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    planning_log_id = Column(
        UUID(as_uuid=True),
        ForeignKey("planning_log.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    action_type = Column(String(50), nullable=False)  # replace / remove / add / reorder / adjust_time
    payload = Column(JSONB, default=dict, nullable=False)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
