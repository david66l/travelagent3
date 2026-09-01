"""Hotel POI ORM model."""

import uuid
from sqlalchemy import Boolean, Column, DateTime, Float, String
from sqlalchemy.dialects.postgresql import UUID

from core.database import Base
from core.clock import utc_now


class Hotel(Base):
    """酒店 POI。"""

    __tablename__ = "hotels"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    city = Column(String(50), nullable=False, index=True)
    district = Column(String(100), nullable=False, default="")
    price_range = Column(String(20), nullable=False, default="mid")

    has_elevator = Column(Boolean, nullable=False, default=True)
    has_breakfast = Column(Boolean, nullable=False, default=False)
    has_parking = Column(Boolean, nullable=False, default=False)
    child_friendly = Column(Boolean, nullable=False, default=False)

    lat = Column(Float, nullable=False, default=0.0)
    lng = Column(Float, nullable=False, default=0.0)
    rating = Column(Float, nullable=True)

    cancel_policy = Column(String(100), nullable=True)
    distance_to_center_km = Column(Float, nullable=True)

    source = Column(String(20), nullable=False, default="amap")
    source_updated_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="active")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
