"""Pydantic request/response models for the VRP solver service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class POIInput(BaseModel):
    """A single point of interest."""

    id: str
    name: str
    category: str = "spot"
    tags: list[str] = Field(default_factory=list)
    lat: float = 0.0
    lng: float = 0.0
    score: float = 0.5
    ticket_price: float = 0.0
    duration_minutes: int = 60
    min_play_time: int = 15
    max_play_time: int = 240
    open_time: str = "08:00"
    close_time: str = "18:00"
    walk_intensity: int = 1
    reservation: str | None = None
    best_visit_period: str | None = None
    must_visit: bool = False
    is_peak: bool = False


class ReservationInput(BaseModel):
    """User-provided reservation / must-attend event."""

    poi_id: str
    date: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    note: str | None = None


class EpsilonConfig(BaseModel):
    """Epsilon-constraint bounds for multi-objective CP-SAT model."""

    min_preference: int = 0
    max_peak_score: int = 100000
    max_walk_diff: int = 100000
    max_budget_mad: int = 100000


class ConstraintsInput(BaseModel):
    """User constraints and preferences."""

    travel_days: int = 1
    day_start_min: int = 8 * 60
    day_end_min: int = 18 * 60
    max_transit_minutes: int = 120
    max_walk_km: int = 8
    total_budget: float = 0.0  # 0 means unlimited
    food_day: float = 100.0
    rest_day: int = 60
    interests: list[str] = Field(default_factory=list)
    must_visit: list[str] = Field(default_factory=list)
    user_reservations: list[ReservationInput] = Field(default_factory=list)
    epsilon_config: EpsilonConfig = Field(default_factory=EpsilonConfig)

    # v4.0 business logic fields
    play_mode: Literal["quick", "standard", "deep"] = "standard"
    include_restaurant: bool = False
    meals_per_day: int = 0
    lunch_window: tuple[int, int] = (11 * 60 + 30, 13 * 60 + 30)
    dinner_window: tuple[int, int] = (17 * 60 + 30, 20 * 60)
    travelers_type: Literal["solo", "couple", "family_kid", "family_elder", "friends", "adult", "young"] = "adult"
    fatigue_recovery_rate: float | None = None

    max_walk_minutes: int | None = None  # deprecated, converted to max_walk_km

    @model_validator(mode="after")
    def _derive_walk_km(self):
        if self.max_walk_minutes is not None:
            self.max_walk_km = max(1, int(self.max_walk_minutes / 60 * 4.5))
        return self


class ActivityOutput(BaseModel):
    """Single scheduled activity inside a day."""

    poi_id: str
    poi_name: str
    category: str
    start_time: str
    end_time: str
    duration_min: int
    ticket_price: float = 0.0
    transport_cost: float = 0.0
    lat: float = 0.0
    lng: float = 0.0
    tags: list[str] = Field(default_factory=list)


class ReminderOutput(BaseModel):
    """Reservation / warning reminder returned to the user."""

    type: str
    message: str
    poi_id: str | None = None
    poi_name: str | None = None
    date: str | None = None


class DayPlanOutput(BaseModel):
    """One day itinerary."""

    day_number: int
    activities: list[ActivityOutput]
    total_cost: float = 0.0
    transport_cost: float = 0.0
    walk_intensity: int = 0


class SolverRequest(BaseModel):
    """Full VRP solve request."""

    pois: list[POIInput]
    constraints: ConstraintsInput
    strategy: str = "auto"  # "auto" | "cpsat" | "greedy"
    dist_matrix: list[list[int]] | None = None
    tc_matrix: list[list[float]] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SolverResponse(BaseModel):
    """Full VRP solve response."""

    status: str
    days: list[DayPlanOutput] = Field(default_factory=list)
    reminders: list[ReminderOutput] = Field(default_factory=list)
    solve_time_ms: int = 0
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
