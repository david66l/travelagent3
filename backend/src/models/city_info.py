"""City metadata ORM model."""

import uuid
from sqlalchemy import Column, DateTime, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID

from core.database import Base
from core.clock import utc_now


class CityInfo(Base):
    """城市元信息：气候、最佳季节、日均成本、推荐天数等。"""

    __tablename__ = "city_info"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    city = Column(String(50), nullable=False, unique=True)

    climate = Column(String(100), nullable=False, default="")
    best_season = Column(String(50), nullable=False, default="")

    # 新增：蓝图 v3.0 要求字段
    daily_avg_cost = Column(Numeric(10, 2), nullable=True)  # 日均花费（元/人）
    recommended_days = Column(Integer, nullable=True)  # 推荐游玩天数
    notes = Column(Text, nullable=True)  # 备注/旅行提示

    district_count = Column(Integer, nullable=True, default=0)
    main_districts = Column(ARRAY(Text), nullable=True, default=list)
    transport_hubs = Column(JSON, default=dict, nullable=True)
    peak_months = Column(ARRAY(Text), nullable=True, default=list)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=True)
