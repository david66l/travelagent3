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
    # Weekdays the venue is closed (0=Mon … 6=Sun, Python date.weekday()). Museums
    # typically close Monday. Applied only when the trip's real dates are known
    # (ConstraintsInput.day_weekdays); empty = open every day.
    closed_weekdays: list[int] = Field(default_factory=list)


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
    # Real weekday of each travel day (0=Mon … 6=Sun), length == travel_days.
    # Derived from the trip start date by the caller. Empty = dates unknown, so
    # weekday-closure constraints are skipped (we never guess which day is Monday).
    day_weekdays: list[int] = Field(default_factory=list)
    day_start_min: int = 8 * 60
    # End the active day late enough that the dinner window (default 17:30-20:00)
    # is actually usable; an 18:00 cutoff makes evening dining impossible and
    # forces meals into mid-afternoon gaps.
    day_end_min: int = 21 * 60
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

    # Slack added to every selected inter-POI commute (queueing/photos/rest/transit
    # delay) so the schedule is not packed minute-to-minute. is_peak POIs get an
    # extra queue pad on top (see solver). 0 disables.
    transition_buffer_min: int = 15
    peak_queue_pad_min: int = 20
    # Two attractions whose road-network time exceeds this cannot share a day, so a
    # far-suburb POI (e.g. 松江辰山) is isolated instead of bundled with a downtown
    # cluster (e.g. 陆家嘴), which otherwise causes 3h+ same-day folding. 60min ⇒ any
    # >1h one-way hop forces its own day (a round trip would burn 2h+ of the day).
    remote_pair_min: int = 60
    # Geographic suburb isolation. Driving time underestimates the suburb pain
    # (tourists ride transit ~1.5x-2x slower), so a 松江/惠南/川沙 attraction can sit
    # just under remote_pair_min in driving minutes and still get bundled with a
    # downtown POI. We therefore also isolate by straight-line distance from the
    # itinerary's centroid: a POI beyond suburb_radius_km shares its day only with
    # attractions within suburb_nearby_min driving (i.e. genuinely adjacent), so a
    # far-suburb landmark gets its own day. 0 disables.
    suburb_radius_km: float = 20.0
    suburb_nearby_min: int = 25
    # Tags whose POIs are functionally interchangeable (观景台 etc.); selecting more
    # than one across the whole trip is penalised so duplicate high-cost landmarks
    # (东方明珠/环球/上海中心) collapse to one.
    redundant_tags: list[str] = Field(default_factory=lambda: ["观景"])

    max_walk_minutes: int | None = None  # deprecated, converted to max_walk_km

    @model_validator(mode="after")
    def _derive_walk_km(self):
        if self.max_walk_minutes is not None:
            self.max_walk_km = max(1, int(self.max_walk_minutes / 60 * 4.5))
        # When dinner is part of the plan, the active day must extend at least to
        # the end of the dinner window, otherwise the meal can never be placed.
        if self.include_restaurant and self.meals_per_day >= 2:
            self.day_end_min = max(self.day_end_min, self.dinner_window[1])
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
    # Coord-keyed real driving times ("lat,lng|lat,lng" -> minutes) from AMap;
    # applied while building the matrix, with haversine fallback per missing edge.
    amap_minutes: dict[str, int] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SolverResponse(BaseModel):
    """Full VRP solve response."""

    status: str
    days: list[DayPlanOutput] = Field(default_factory=list)
    reminders: list[ReminderOutput] = Field(default_factory=list)
    solve_time_ms: int = 0
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
