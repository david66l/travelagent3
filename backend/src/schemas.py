from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


# ===== User Profile =====


class UserProfile(BaseModel):
    destination: Optional[str] = None
    travel_days: Optional[int] = None
    travel_dates: Optional[str] = None
    travelers_count: int = 1
    travelers_type: Optional[str] = None  # 独自/情侣/亲子/朋友/父母
    budget_range: Optional[float] = None
    food_preferences: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    pace: str = "moderate"  # relaxed / moderate / intensive
    accommodation_preference: Optional[str] = None
    special_requests: list[str] = Field(default_factory=list)
    preference_history: list[dict] = Field(default_factory=list)


# ===== Location =====


class Location(BaseModel):
    lat: float
    lng: float
    address: Optional[str] = None


# ===== Activity / DayPlan / Itinerary =====


class Activity(BaseModel):
    poi_name: str
    poi_id: Optional[str] = None
    category: str = "attraction"  # attraction / restaurant / hotel / transport
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_min: int = 120
    ticket_price: Optional[float] = None
    meal_cost: Optional[float] = None
    transport_cost: Optional[float] = None
    location: Optional[Location] = None
    recommendation_reason: str = ""
    transit_from_prev: Optional[dict] = None
    time_constraint: str = "flexible"  # flexible / morning_only / afternoon_only / evening_only
    open_time: Optional[str] = None
    close_time: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class DayPlan(BaseModel):
    day_number: int
    date: Optional[str] = None
    theme: Optional[str] = None
    activities: list[Activity] = Field(default_factory=list)
    total_cost: float = 0
    total_walking_steps: int = 0
    total_transit_time_min: int = 0


class ItineraryRecord(BaseModel):
    record_id: str
    session_id: str
    destination: str
    travel_days: int
    daily_plans: list[DayPlan] = Field(default_factory=list)
    preference_snapshot: dict = Field(default_factory=dict)
    budget_snapshot: dict = Field(default_factory=dict)
    status: str = "draft"  # draft / confirmed / completed


# ===== Panels =====


class BudgetPanel(BaseModel):
    total_budget: Optional[float] = None
    spent: float = 0
    remaining: Optional[float] = None
    breakdown: dict = Field(
        default_factory=dict
    )  # accommodation/meals/transport/tickets/shopping/buffer
    status: str = "within_budget"  # within_budget / over_budget


class PreferencePanel(BaseModel):
    destination: Optional[str] = None
    travel_days: Optional[int] = None
    travel_dates: Optional[str] = None
    travelers_count: Optional[int] = None
    travelers_type: Optional[str] = None
    budget_range: Optional[float] = None
    food_preferences: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    pace: Optional[str] = None
    special_requests: list[str] = Field(default_factory=list)


# ===== Intent Result =====


class IntentResult(BaseModel):
    intent: Literal[
        "generate_itinerary",
        "modify_itinerary",
        "update_preferences",
        "query_info",
        "confirm_itinerary",
        "view_history",
        "chitchat",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    user_entities: dict = Field(default_factory=dict)
    missing_required: list[str] = Field(default_factory=list)
    missing_recommended: list[str] = Field(default_factory=list)
    preference_changes: list[dict] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    reasoning: str = ""

    @field_validator(
        "preference_changes",
        "missing_required",
        "missing_recommended",
        "clarification_questions",
        mode="before",
    )
    @classmethod
    def _ensure_list(cls, v):
        return v if v is not None else []


# ===== Validation =====


class ValidationResult(BaseModel):
    passed: bool = False
    scores: dict = Field(default_factory=dict)
    total_score: float = 0.0
    critical_failures: list[str] = Field(default_factory=list)
    improvement_suggestions: list[str] = Field(default_factory=list)


# ===== Evaluation =====


class EvalResult(BaseModel):
    scores: dict = Field(default_factory=dict)
    total_score: float = 0.0
    passed: bool = False
    critical_failures: list[str] = Field(default_factory=list)


# ===== POI / Weather / Price =====


class ScoredPOI(BaseModel):
    name: str
    category: str
    score: float
    location: Optional[Location] = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    ticket_price: Optional[float] = None
    rating: Optional[float] = None
    review_count: Optional[int] = None
    open_time: Optional[str] = None
    close_time: Optional[str] = None
    best_time: Optional[str] = None
    highlights: Optional[str] = None
    time_constraint: str = "flexible"  # flexible / morning_only / afternoon_only / evening_only
    area: Optional[str] = None  # 所在区域/商圈，如"外滩","朝阳区"
    recommended_hours: Optional[str] = None  # 建议游玩时长，如"2-3小时","半天"
    indoor_outdoor: Optional[str] = None  # indoor / outdoor / mixed


class WeatherDay(BaseModel):
    date: str
    condition: str
    temp_high: int
    temp_low: int
    precipitation_chance: int
    wind_speed: Optional[int] = None
    recommendation: Optional[str] = None


class PriceInfo(BaseModel):
    poi_name: str
    price_type: str  # ticket / meal / hotel
    price_range: Optional[str] = None
    currency: str = "CNY"
    source: str = ""


class RouteInfo(BaseModel):
    origin: Location
    destination: Location
    distance_m: int
    duration_min: int
    mode: str  # walk / transit / taxi
    polyline: Optional[str] = None


# ===== Travel Context (enriched from multi-dimensional search) =====


class TravelContext(BaseModel):
    """Rich travel context gathered from multi-dimensional web search."""

    route_suggestions: str = ""
    accommodation_areas: str = ""
    transport_tips: str = ""
    upcoming_events: list[dict] = Field(default_factory=list)
    food_specialties: list[dict] = Field(default_factory=list)
    pitfall_tips: list[str] = Field(default_factory=list)
    seasonal_highlights: str = ""
    local_customs: str = ""

    def to_prompt_text(self) -> str:
        """Format as human-readable text for LLM prompts."""
        lines = []
        if self.seasonal_highlights:
            lines.append(f"季节限定：{self.seasonal_highlights}")
        if self.upcoming_events:
            lines.append("近期活动：")
            for e in self.upcoming_events[:6]:
                lines.append(
                    f"  - {e.get('name', '')} ({e.get('date_range', '')}) @ {e.get('location', '')}"
                )
        if self.route_suggestions:
            lines.append(f"路线参考：{self.route_suggestions}")
        if self.transport_tips:
            lines.append(f"交通提示：{self.transport_tips}")
        if self.accommodation_areas:
            lines.append(f"住宿建议：{self.accommodation_areas}")
        if self.food_specialties:
            lines.append("美食推荐：")
            for f in self.food_specialties[:6]:
                lines.append(
                    f"  - {f.get('name', '')} ({f.get('cuisine_type', '')}) @ {f.get('area', '')}"
                )
        if self.pitfall_tips:
            lines.append("避坑提醒：")
            for tip in self.pitfall_tips[:6]:
                lines.append(f"  - {tip}")
        if self.local_customs:
            lines.append(f"本地习俗：{self.local_customs}")
        return "\n".join(lines) if lines else "暂无当地实用信息"

    @classmethod
    def from_dict(cls, data: dict | None) -> "TravelContext":
        """Safely create from a dict (e.g. from ItineraryState)."""
        if not data:
            return cls()
        return cls(**data)


# ===== Memory =====


class UserMemory(BaseModel):
    recent_itineraries: list[dict] = Field(default_factory=list)
    preference_patterns: dict = Field(default_factory=dict)
    recent_conversations: list[dict] = Field(default_factory=list)


# ===== Seed Data =====


class AttractionSeed(BaseModel):
    name: str
    category: str
    highlights: str
    best_time: str


class RestaurantSeed(BaseModel):
    name: str
    category: str
    specialties: str


class CityInfoSeed(BaseModel):
    consumption_level: str
    climate: str
    transportation: str


class CitySeedData(BaseModel):
    city: str
    attractions: list[AttractionSeed] = Field(default_factory=list)
    restaurants: list[RestaurantSeed] = Field(default_factory=list)
    city_info: Optional[CityInfoSeed] = None
    collected_at: Optional[str] = None


# ===== Chat / WebSocket =====


class ChatRequest(BaseModel):
    content: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    assistant_message: str
    itinerary: Optional[list[DayPlan]] = None
    budget_panel: Optional[BudgetPanel] = None
    preference_panel: Optional[PreferencePanel] = None
    intent: Optional[str] = None
    needs_clarification: bool = False
    waiting_for_confirmation: bool = False
