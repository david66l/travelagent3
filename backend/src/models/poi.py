"""Unified structured POI model for RAG retrieval and itinerary planning."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class POI(BaseModel):
    """Structured POI — the core data unit of the knowledge base layer.

    Maps to PostgreSQL tables:
    - spot_type="attraction" → attractions
    - spot_type="restaurant" → restaurants
    - spot_type="hotel"      → hotels
    """

    # ── 基础信息 ──
    spot_id: str = Field(..., description="全局唯一POI ID（UUID字符串）")
    spot_name: str = Field(..., description="POI名称，如'故宫博物院'")
    spot_type: Literal["attraction", "restaurant", "hotel"] = Field(
        ..., description="类型: attraction/restaurant/hotel"
    )
    city: str = Field(..., description="所属城市")
    district: Optional[str] = Field(None, description="区县")

    # ── 位置 ──
    lat: float = Field(..., description="纬度 WGS84")
    lng: float = Field(..., description="经度 WGS84")
    address: Optional[str] = Field(None, description="详细地址")

    # ── 时间 ──
    open_time: str = Field("08:00", description="开门时间 HH:MM")
    close_time: str = Field("18:00", description="关门时间 HH:MM")
    duration_minutes: int = Field(120, description="建议游玩分钟数")
    best_visit_time: Optional[str] = Field(None, description="最佳游览时段")

    # ── 费用 ──
    ticket_price: float = Field(0.0, description="门票/人均价格（元）")
    price_level: int = Field(2, ge=1, le=5, description="价格等级 1-5")

    # ── 体力 ──
    walk_intensity: int = Field(3, ge=1, le=5, description="步行强度 1-5")
    queue_time_avg: int = Field(30, description="平均排队分钟数")
    indoor_outdoor: Literal["indoor", "outdoor", "mixed"] = Field("outdoor", description="室内外")

    # ── 预约 ──
    need_reservation: bool = Field(False, description="是否需要预约")
    reservation_advance_days: int = Field(0, description="需提前预约天数")
    reservation_channel: Optional[str] = Field(None, description="预约渠道")

    # ── 标签与描述 ──
    tags: list[str] = Field(default_factory=list, description="标签")
    description: Optional[str] = Field(None, description="POI简介（用于Embedding）")

    # ── 人群适配 ──
    suitable_for: list[str] = Field(
        default_factory=lambda: ["solo", "couple", "family_kid", "family_elder", "friends"],
        description="适合的人群类型",
    )
    accessibility: list[str] = Field(default_factory=list, description="无障碍设施")

    # ── 评分 ──
    rating: float = Field(4.0, ge=0, le=5, description="用户评分 0-5")
    review_count: int = Field(0, description="评价数量")

    # ── 动态字段（实时API更新，不存DB）──
    current_weather: Optional[str] = Field(None, description="实时天气（API填充）")
    current_queue_time: Optional[int] = Field(None, description="实时排队分钟（API填充）")
    is_open_today: Optional[bool] = Field(None, description="今日是否开放（API填充）")

    # ── 系统字段 ──
    rrf_score: Optional[float] = Field(None, description="融合排序分数")
    reservation_reminder: Optional[bool] = Field(None, description="需预约必去标记")
    source: Literal["structured", "vector", "bm25"] = Field("structured", description="来自哪一路检索")

    @classmethod
    def from_attraction_row(cls, row: dict) -> "POI":
        """Build POI from an attractions table row."""
        row = dict(row)
        open_time = row.get("open_time")
        close_time = row.get("close_time")
        return cls(
            spot_id=str(row.get("id", "")),
            spot_name=row.get("name", ""),
            spot_type="attraction",
            city=row.get("city", ""),
            lat=float(row.get("lat", 0.0)),
            lng=float(row.get("lng", 0.0)),
            address=row.get("address") or None,
            open_time=_time_to_str(open_time, "08:00"),
            close_time=_time_to_str(close_time, "18:00"),
            duration_minutes=int(row.get("duration_minutes", 120)),
            ticket_price=float(row.get("ticket_price", 0.0)) if row.get("ticket_price") is not None else 0.0,
            walk_intensity=int(row.get("walk_intensity", 3)),
            queue_time_avg=int(row.get("queue_time_avg", 30)) if row.get("queue_time_avg") is not None else 30,
            indoor_outdoor=row.get("indoor_outdoor") or "outdoor",
            need_reservation=bool(row.get("need_reservation", False)),
            reservation_advance_days=int(row.get("reservation_advance_days", 0)) if row.get("reservation_advance_days") is not None else 0,
            tags=list(row.get("tags", [])) or [],
            description=row.get("description") or None,
            suitable_for=list(row.get("suitable_for", [])) or ["solo", "couple", "family_kid", "family_elder", "friends"],
            accessibility=_accessibility_from_row(row),
            source="structured",
        )

    @classmethod
    def from_restaurant_row(cls, row: dict) -> "POI":
        """Build POI from a restaurants table row."""
        row = dict(row)
        open_time = row.get("open_time")
        close_time = row.get("close_time")
        return cls(
            spot_id=str(row.get("id", "")),
            spot_name=row.get("name", ""),
            spot_type="restaurant",
            city=row.get("city", ""),
            lat=float(row.get("lat", 0.0)),
            lng=float(row.get("lng", 0.0)),
            address=row.get("address") or None,
            open_time=_time_to_str(open_time, "08:00"),
            close_time=_time_to_str(close_time, "22:00"),
            duration_minutes=90,
            ticket_price=float(row.get("avg_price", 0.0)) if row.get("avg_price") is not None else 0.0,
            price_level=2,
            walk_intensity=2,
            tags=list(row.get("tags", [])) or [],
            description=None,
            suitable_for=["solo", "couple", "family_kid", "family_elder", "friends"],
            source="structured",
        )


def _time_to_str(value, default: str) -> str:
    """Normalize datetime.time or string to HH:MM."""
    if value is None:
        return default
    if hasattr(value, "strftime"):
        return value.strftime("%H:%M")
    return str(value)[:5]


def _accessibility_from_row(row: dict) -> list[str]:
    """Extract accessibility list from attraction row."""
    acc = row.get("accessibility")
    if isinstance(acc, list):
        return list(acc)
    if isinstance(acc, dict):
        return [k for k, v in acc.items() if v]
    if isinstance(acc, str):
        return [acc]
    if row.get("wheelchair_accessible"):
        return ["wheelchair"]
    return []
