"""Restaurant POI ORM model."""

import uuid
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    Time,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from core.database import Base
from core.clock import utc_now


class Restaurant(Base):
    """餐厅 POI。"""

    __tablename__ = "restaurants"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    city = Column(String(50), nullable=False, index=True)
    cuisine = Column(String(50), nullable=True)
    avg_price = Column(Numeric(10, 2), nullable=True)

    open_time = Column(Time, nullable=True)
    close_time = Column(Time, nullable=True)

    lat = Column(Float, nullable=False, default=0.0)
    lng = Column(Float, nullable=False, default=0.0)
    address = Column(String(500), nullable=False, default="")

    tags = Column(ARRAY(Text), nullable=False, default=list)
    signature_dishes = Column(ARRAY(Text), nullable=False, default=list)
    rating = Column(Float, nullable=True)

    queue_time_min = Column(Integer, nullable=True)
    cancel_policy = Column(String(50), nullable=True)

    source = Column(String(20), nullable=False, default="amap")
    source_updated_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="active")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    __table_args__ = (Index("ix_restaurants_name_city", "name", "city", unique=True),)
