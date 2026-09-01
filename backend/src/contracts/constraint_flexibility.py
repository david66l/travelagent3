"""User constraint-flexibility contract without database dependencies."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


ConstraintDimension = Literal[
    "activity_schedule",
    "activity_set",
    "daily_time_window",
    "fixed_event_time",
    "max_transit_minutes",
    "total_budget",
]


class ConstraintFlexibilityContract(BaseModel):
    """Neutral user constraint mutability facts emitted by intent parsing."""

    model_config = {"extra": "forbid"}

    schema_version: Literal["constraint-flexibility.v1"] = "constraint-flexibility.v1"
    locked_constraints: list[ConstraintDimension] = Field(default_factory=list)
    solver_adjustable_constraints: list[ConstraintDimension] = Field(
        default_factory=list,
        description=(
            "可由求解器在既有用户边界内重排的约束，不代表允许放宽边界或保证重试"
        ),
    )
    relaxable_constraints: list[ConstraintDimension] = Field(default_factory=list)
    relaxation_options: dict[ConstraintDimension, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_contract(self) -> ConstraintFlexibilityContract:
        locked = set(self.locked_constraints)
        adjustable = set(self.solver_adjustable_constraints)
        relaxable = set(self.relaxable_constraints)
        if locked & adjustable or locked & relaxable or adjustable & relaxable:
            raise ValueError("locked, solver-adjustable and relaxable constraints must be disjoint")
        if not set(self.relaxation_options) <= relaxable:
            raise ValueError("relaxation options must belong to relaxable constraints")
        for dimension, options in self.relaxation_options.items():
            normalized = [str(item).strip() for item in options if str(item).strip()]
            if not normalized:
                raise ValueError(f"relaxation options are empty for {dimension}")
            self.relaxation_options[dimension] = list(dict.fromkeys(normalized))
        self.locked_constraints = list(dict.fromkeys(self.locked_constraints))
        self.solver_adjustable_constraints = list(
            dict.fromkeys(self.solver_adjustable_constraints)
        )
        self.relaxable_constraints = list(dict.fromkeys(self.relaxable_constraints))
        return self
