"""Shared production contract for bounded post-verifier repair decisions.

This module deliberately exposes constraint facts, not desired policy actions.
The controller can prove whether a failed constraint is locked, negotiable, or
adjustable within existing bounds; the policy still has to select and ground
the correct action.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ValidationError

from evaluation.validator import VALIDATOR_HARD_VIOLATION_CODES
from contracts.constraint_flexibility import ConstraintFlexibilityContract


ConstraintHandling = Literal["locked", "relaxable", "solver_adjustable"]

VERIFIER_REPAIR_ACTIONS = frozenset({"retry_solve", "propose_tradeoff", "abort"})
VERIFIER_REPAIR_CONSTRAINT_BY_CODE = {
    "ACTIVITY_TIME_OVERLAP": "activity_schedule",
    "DAY_TIME_BOUNDARY_EXCEEDED": "daily_time_window",
    "FIXED_EVENT_END_EXCEEDED": "fixed_event_time",
    "FIXED_EVENT_TIME_MISMATCH": "fixed_event_time",
    "MAX_TRANSIT_EXCEEDED": "max_transit_minutes",
    "TOTAL_BUDGET_EXCEEDED": "total_budget",
}
VERIFIER_REPAIR_VIOLATION_CODES = frozenset(VERIFIER_REPAIR_CONSTRAINT_BY_CODE)

if not VERIFIER_REPAIR_VIOLATION_CODES <= VALIDATOR_HARD_VIOLATION_CODES:
    raise RuntimeError("verifier-repair route contains non-production validation codes")


def parse_constraint_flexibility(raw: Any) -> ConstraintFlexibilityContract | None:
    """Validate a prompt-visible contract and fail closed on any schema drift."""
    if not isinstance(raw, dict):
        return None
    try:
        return ConstraintFlexibilityContract.model_validate(raw)
    except (TypeError, ValidationError, ValueError):
        return None


def constraint_handling_for_violation(
    code: str,
    contract: ConstraintFlexibilityContract,
) -> ConstraintHandling | None:
    """Return the user-declared handling class for one production violation."""
    dimension = VERIFIER_REPAIR_CONSTRAINT_BY_CODE.get(str(code))
    if dimension is None:
        return None
    if dimension in contract.locked_constraints:
        return "locked"
    if dimension in contract.relaxable_constraints:
        return "relaxable"
    if dimension in contract.solver_adjustable_constraints:
        return "solver_adjustable"
    return None


def relaxation_options_for_violation(
    code: str,
    contract: ConstraintFlexibilityContract,
) -> list[str]:
    """Return only user-authorized options for the failed constraint."""
    dimension = VERIFIER_REPAIR_CONSTRAINT_BY_CODE.get(str(code))
    if dimension is None:
        return []
    return list(contract.relaxation_options.get(dimension) or [])
