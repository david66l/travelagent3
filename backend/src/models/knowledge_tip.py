"""Knowledge tip / travel guide ORM model."""

import uuid
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from core.database import Base
from core.clock import utc_now


class KnowledgeTip(Base):
    """城市攻略/知识贴士，支持向量语义检索。"""

    __tablename__ = "knowledge_tips"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    city = Column(String(50), nullable=True, index=True)
    content = Column(Text, nullable=False)
    content_type = Column(String(20), nullable=False, default="guide")
    walk_intensity = Column(Integer, nullable=True)
    suitable_for = Column(ARRAY(Text), nullable=False, default=list)

    embedding = Column(Vector(1024), nullable=True)

    source = Column(String(20), nullable=False, default="manual")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
