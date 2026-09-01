"""User trip history ORM model."""

import uuid
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID

from core.database import Base
from core.clock import utc_now


class UserTripHistory(Base):
    """用户历史行程记录，含行程向量化表示，用于长期画像召回。"""

    __tablename__ = "user_trip_history"
    __table_args__ = (
        Index(
            "ix_user_trip_history_trip_vector_hnsw",
            "trip_vector",
            postgresql_using="hnsw",
            postgresql_ops={"trip_vector": "vector_cosine_ops"},
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    trip_json = Column(JSONB, default=dict, nullable=False)
    trip_vector = Column(Vector(1024), nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
