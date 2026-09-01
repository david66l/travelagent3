"""Structured travel demand slots aligned with PRD v3.0."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from contracts.constraint_flexibility import ConstraintFlexibilityContract


class TravelSlots(BaseModel):
    """User travel demand as structured slots.

    Fields are intentionally nullable so the agent can represent partial
    understanding and identify missing slots for clarification.
    """

    # --- 基础信息 ---
    origin: Optional[str] = Field(default=None, description="出发城市")
    destination: Optional[str] = Field(default=None, description="目的地城市")
    travel_days: Optional[int] = Field(default=None, ge=1, le=30, description="旅行天数")
    travel_dates: Optional[str] = Field(default=None, description="旅行日期描述，如'下周'、'5月1日'")

    # --- 人群 ---
    travelers_count: Optional[int] = Field(default=None, ge=1, le=50, description="出行人数")
    travel_companion: Optional[Literal["alone", "couple", "family", "friends", "parents", "colleagues"]] = (
        Field(default=None, description="同行类型")
    )
    has_elderly: Optional[bool] = Field(default=None, description="是否有老人同行")
    has_children: Optional[bool] = Field(default=None, description="是否有儿童同行")
    has_pregnant: Optional[bool] = Field(default=None, description="是否有孕妇同行")
    has_wheelchair: Optional[bool] = Field(default=None, description="是否有轮椅出行")

    # --- 预算 ---
    total_budget: Optional[float] = Field(default=None, ge=0, description="总预算（元）")
    budget_per_person: Optional[float] = Field(default=None, ge=0, description="人均预算（元）")

    # --- 偏好 ---
    interests: list[str] = Field(default_factory=list, description="兴趣标签")
    food_prefs: list[str] = Field(default_factory=list, description="饮食偏好")
    food_taboos: list[str] = Field(default_factory=list, description="饮食禁忌/忌口")
    must_visit: list[str] = Field(default_factory=list, description="必去景点/地点")
    must_not_visit: list[str] = Field(default_factory=list, description="排除景点/地点")

    # --- 约束 ---
    pace: Optional[Literal["relaxed", "moderate", "intensive"]] = Field(
        default=None, description="行程节奏"
    )
    play_mode: Optional[Literal["sightseeing", "foodie", "shopping", "adventure", "culture", "leisure"]] = (
        Field(default=None, description="游玩模式")
    )
    max_walk_minutes: Optional[int] = Field(default=None, ge=0, description="单日最大步行分钟数")
    max_transit_minutes: Optional[int] = Field(default=None, ge=0, description="单次通勤最大分钟数")
    avoid_crowds: Optional[bool] = Field(default=None, description="是否避开拥挤")
    prefer_morning: Optional[bool] = Field(default=None, description="是否偏好上午出行")
    include_restaurant: Optional[bool] = Field(default=None, description="是否包含餐厅推荐")
    transport_preference: Optional[Literal["public", "taxi", "walk", "rental_car", "mixed", "any"]] = (
        Field(default=None, description="交通偏好")
    )
    fatigue_preference: Optional[Literal["low", "medium", "high"]] = Field(
        default=None, description="疲劳接受度"
    )

    # --- Agent 编排语义（由意图模型识别，不进入长期用户画像） ---
    intent_kind: Optional[Literal["itinerary", "event_trip"]] = Field(
        default=None,
        description="普通行程或包含具有确定时间地点的活动行程",
    )
    event_query: Optional[str] = Field(
        default=None,
        max_length=200,
        description="需要通过统一搜索核实的活动主体，如歌手演唱会或体育比赛",
    )
    transport_modes_requested: list[Literal["flight", "train", "bus", "ferry"]] = Field(
        default_factory=list,
        description="用户明确要求查询班次的城际交通方式",
    )
    information_needs: list[
        Literal[
            "event",
            "transport",
            "weather",
            "opening_hours",
            "closure",
            "restaurant",
            "seasonal_activity",
            "general",
        ]
    ] = Field(
        default_factory=list,
        description="规划前必须通过工具补齐的外部事实类型",
    )
    current_info_queries: list[str] = Field(
        default_factory=list,
        description="需要通过统一搜索工具核实的自然语言查询",
    )
    constraint_flexibility: Optional[ConstraintFlexibilityContract] = Field(
        default=None,
        description=(
            "用户明确声明的约束可变性事实；仅记录锁定项、可放宽项和获准范围，"
            "不得输出重试、权衡或终止动作标签"
        ),
    )

    # --- 画像推断标记 ---
    inferred_slots: list[str] = Field(default_factory=list, description="哪些槽位来自画像推断")

    @field_validator("interests", "food_prefs", "food_taboos", "must_visit", "must_not_visit", mode="before")
    @classmethod
    def _ensure_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    def missing_required(self) -> list[str]:
        """Return required slots that are still empty."""
        required = ["destination", "travel_days"]
        if self.transport_modes_requested:
            required.append("origin")
        return [field for field in required if getattr(self, field) is None]

    def to_flat_dict(self) -> dict:
        """Flatten slots to a dictionary, dropping empty lists."""
        return {
            k: v
            for k, v in self.model_dump().items()
            if v is not None and v != []
        }


class SlotParseOutput(BaseModel):
    """Direct output of the demand parser agent.

    Designed to be produced by a lightweight LLM (task_type="intent" → 7B/small).
    """

    intent: Literal[
        "generate_itinerary",
        "modify_itinerary",
        "update_preferences",
        "query_info",
        "confirm_itinerary",
        "view_history",
        "chitchat",
    ] = "generate_itinerary"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    sentiment: Literal["positive", "neutral", "negative", "urgent"] = "neutral"
    slots: TravelSlots = Field(default_factory=TravelSlots)
    missing_slots: list[str] = Field(default_factory=list)
    clarifying_question: Optional[str] = None
    disambiguation: Optional[dict[str, Any]] = None
    parse_source: Literal["llm", "deterministic_fallback"] = "llm"
    token_usage: int = Field(default=0, ge=0, exclude=True)


RevisionField = Literal[
    "origin",
    "destination",
    "travel_days",
    "start_date",
    "end_date",
    "budget_range",
    "must_visit",
    "must_not_visit",
    "mobility_constraints",
    "max_transit_minutes",
    "intent_kind",
    "event_query",
    "transport_modes_requested",
    "information_needs",
    "current_info_queries",
    "interests",
    "food_preferences",
    "transport_preference",
    "hotel_preference",
    "pace",
    "travelers_type",
    "travelers_count",
    "has_children",
    "has_elderly",
    "avoid_pois",
]


class RevisionOperation(BaseModel):
    """One model-proposed edit; the controller still validates value types."""

    field: RevisionField
    operation: Literal["set", "add", "remove", "clear"] = "set"
    value: Any = None


class RevisionParseOutput(BaseModel):
    """Structured interpretation of free-form feedback on an itinerary draft."""

    intent: Literal[
        "revise_itinerary",
        "clarify_revision",
        "accept_itinerary",
        "start_new_trip",
    ] = "revise_itinerary"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    operations: list[RevisionOperation] = Field(default_factory=list)
    affected_domains: list[
        Literal["research", "candidates", "transport", "schedule", "budget", "presentation"]
    ] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
