"""Validated case contract and deterministic IDs for authorized Shadow replay."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class AuthorizedReplayCase(BaseModel):
    schema_version: Literal["authorized-shadow-replay.v1"] = "authorized-shadow-replay.v1"
    case_id: str = Field(min_length=1, max_length=128, pattern=r"^[a-z0-9][a-z0-9-]+$")
    family: Literal["solvable_plan"] = "solvable_plan"
    destination: str = Field(min_length=1, max_length=32)
    start_date: str
    end_date: str
    travel_days: int = Field(ge=1, le=7)
    budget: float = Field(gt=0, le=100000)
    interests: list[str] = Field(min_length=1, max_length=5)
    release_gate_eligible: bool = True

    @model_validator(mode="after")
    def validate_dates(self) -> AuthorizedReplayCase:
        from datetime import date

        start = date.fromisoformat(self.start_date)
        end = date.fromisoformat(self.end_date)
        if end < start or (end - start).days + 1 != self.travel_days:
            raise ValueError("date range must match travel_days")
        return self


def load_authorized_replay_cases(path: Path) -> list[AuthorizedReplayCase]:
    cases = [
        AuthorizedReplayCase(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("authorized replay case file is empty")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("authorized replay case_id values must be unique")
    return cases


def replay_scenario_id(*, deployment_id: str, batch_id: str, case_id: str) -> str:
    material = f"{deployment_id}:{batch_id}:{case_id}".encode("utf-8")
    return f"replay-{hashlib.sha256(material).hexdigest()[:48]}"


def replay_case_state(case: AuthorizedReplayCase) -> dict:
    slots = {
        "destination": case.destination,
        "travel_days": case.travel_days,
        "start_date": case.start_date,
        "end_date": case.end_date,
        "budget_range": case.budget,
        "interests": case.interests,
    }
    interests = "、".join(case.interests)
    return {
        "user_input": (
            f"请规划{case.travel_days}天{case.destination}行程，"
            f"预算{case.budget:.0f}元，偏好{interests}。"
        ),
        "slots": slots,
        "profile": slots,
        "missing_slots": [],
        "feasibility_report": {"feasible": True, "status": "solvable"},
    }
