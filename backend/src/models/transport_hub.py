"""Transport hub ORM model."""

import uuid
from sqlalchemy import Column, DateTime, Float, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, UUID

from core.database import Base
from core.clock import utc_now


class TransportHub(Base):
    """交通枢纽：机场、火车站、汽车站、地铁站等。"""

    __tablename__ = "transport_hubs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    city = Column(String(50), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    hub_type = Column(String(20), nullable=False)  # airport / railway / bus / subway

    lat = Column(Float, nullable=False, default=0.0)
    lng = Column(Float, nullable=False, default=0.0)
    lines = Column(ARRAY(Text), nullable=True, default=list)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=True)
