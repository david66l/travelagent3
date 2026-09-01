"""Multi-modal spot-to-spot distance matrix ORM model."""

import uuid
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.dialects.postgresql import UUID

from core.database import Base
from core.clock import utc_now


class SpotDistanceMulti(Base):
    """多交通方式通勤矩阵：from_spot → to_spot 在不同交通方式下的耗时与花费。"""

    __tablename__ = "spot_distance_multi"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    from_spot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("attractions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    to_spot_id = Column(
        UUID(as_uuid=True),
        ForeignKey("attractions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    mode = Column(String(20), nullable=False)  # walk / subway / taxi / bus / driving
    duration_min = Column(Integer, nullable=False)  # 通勤时长（分钟）
    cost = Column(Numeric(10, 2), nullable=True)  # 通勤花费（元）
    time_window = Column(String(50), nullable=True)  # 时间段，如 "08:00-10:00"
    is_default = Column(Boolean, nullable=False, default=False)  # 是否默认交通方式

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    __table_args__ = (
        Index(
            "ix_spot_distance_from_to_mode",
            "from_spot_id",
            "to_spot_id",
            "mode",
            unique=True,
        ),
    )
