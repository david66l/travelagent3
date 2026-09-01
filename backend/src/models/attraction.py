"""Attraction POI ORM model aligned with PRD v3.0 + v4.0."""

import uuid
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
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
from sqlalchemy.dialects.postgresql import ARRAY, JSON, TSVECTOR, UUID

from core.database import Base
from core.clock import utc_now


class Attraction(Base):
    """景点 POI，支持结构化过滤 + 向量语义检索 + BM25 全文检索。"""

    __tablename__ = "attractions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(200), nullable=False)
    city = Column(String(50), nullable=False, index=True)
    category = Column(String(50), nullable=False, default="attraction", index=True)

    # 票务与营业时间
    ticket_price = Column(Numeric(10, 2), nullable=True)
    open_time = Column(Time, nullable=True)
    close_time = Column(Time, nullable=True)

    # 评分（0-5，用于排序与筛选）
    rating = Column(Float, nullable=True)

    # 地理信息
    lat = Column(Float, nullable=False, default=0.0)
    lng = Column(Float, nullable=False, default=0.0)
    address = Column(String(500), nullable=False, default="")

    # 游玩时长（分钟）
    duration_minutes = Column(Integer, nullable=False, default=120)
    walk_intensity = Column(Integer, nullable=False, default=3)  # 体力强度 1-5
    min_play_time = Column(Integer, nullable=True)  # 最短游玩时长
    max_play_time = Column(Integer, nullable=True)  # 最长游玩时长

    # 预约与排队
    need_reservation = Column(Boolean, nullable=False, default=False)  # 是否需要预约
    reservation_advance_days = Column(Integer, nullable=True)  # 需提前 N 天预约
    queue_time_avg = Column(Integer, nullable=True)  # 平均排队时长（分钟）

    # 可达性与环境
    wheelchair_accessible = Column(Boolean, nullable=False, default=False)
    accessibility = Column(
        JSON, default=dict, server_default="{}", nullable=False
    )  # 无障碍设施详情
    indoor_outdoor = Column(String(20), nullable=True)  # indoor / outdoor / mixed
    night_open = Column(
        Boolean, nullable=False, default=False, server_default="false"
    )  # 是否夜间开放

    # 标签与适用人群
    tags = Column(ARRAY(Text), nullable=False, default=list)
    spot_tags = Column(
        ARRAY(Text), nullable=False, default=list, server_default="{}"
    )  # 景点细分标签
    suitable_for = Column(ARRAY(Text), nullable=False, default=list)

    # 季节与闭园
    best_season = Column(String(50), nullable=True)
    season_restriction = Column(
        ARRAY(Text), nullable=False, default=list, server_default="{}"
    )  # 季节限制
    temp_closure_dates = Column(
        ARRAY(Text), nullable=False, default=list, server_default="{}"
    )  # 临时闭园日期
    peak_hours = Column(String(50), nullable=True)

    # 描述与向量 / 全文检索
    description = Column(Text, nullable=True)
    description_vector = Column(Vector(1024), nullable=True)  # BGE-large-zh-v1.5 向量
    search_vector = Column(TSVECTOR, nullable=True)  # BM25 全文检索向量

    # 来源与状态
    source = Column(String(20), nullable=False, default="amap")
    source_updated_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), nullable=False, default="active")
    review_status = Column(String(20), nullable=False, default="verified")
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)

    __table_args__ = (
        Index("ix_attractions_name_city", "name", "city", unique=True),
        Index(
            "idx_attractions_vector",
            "description_vector",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"description_vector": "vector_cosine_ops"},
        ),
        Index(
            "idx_attractions_search",
            "search_vector",
            postgresql_using="gin",
        ),
    )
