"""User long-term profile vector ORM model."""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID

from core.database import Base
from core.clock import utc_now


class UserProfileVector(Base):
    """用户长期画像向量表，用于画像召回与相似推荐。"""

    __tablename__ = "user_profile_vectors"

    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE", name="fk_user_profile_vectors_user_id"),
        primary_key=True,
    )

    preference_embedding = Column(Vector(1024), nullable=True)
    profile_json = Column(JSON, default=dict, nullable=False)

    visited_cities = Column(ARRAY(Text), nullable=False, default=list)
    favorite_spots = Column(ARRAY(Text), nullable=False, default=list)
    avoid_spots = Column(ARRAY(Text), nullable=False, default=list)
    liked_foods = Column(ARRAY(Text), nullable=False, default=list)
    avoided_foods = Column(ARRAY(Text), nullable=False, default=list)

    avg_daily_budget = Column(Numeric(10, 2), nullable=True)
    trip_count = Column(Integer, nullable=False, default=0)

    preferred_transport = Column(String(50), nullable=False, default="")
    preferred_accommodation = Column(String(50), nullable=False, default="")

    updated_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False,
    )

    __table_args__ = (Index("ix_profile_vectors_updated", "updated_at"),)
