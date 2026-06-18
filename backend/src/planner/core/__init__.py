"""Planning Core — deterministic draft generation."""

from planner.core.models import (
    Strategy,
    DayTheme,
    AreaGroup,
    RuleViolation,
    ValidationReport,
    RepairPlan,
    RepairResult,
    PlanningCoreResult,
    EnrichedItinerary,
)
from planner.core.heuristic_strategy import build_strategy
from planner.core.daily_scheduler import build_schedule, detect_remote_pois
from planner.core.rule_validator import validate
from planner.core.repair import generate_repairs, apply_repair, run_repair_loop
from planner.core.fact_guard import (
    activity_fields_match,
    protected_field_differences,
    protected_fields_match,
)
from planner.core.writer import enrich

__all__ = [
    "Strategy",
    "DayTheme",
    "AreaGroup",
    "RuleViolation",
    "ValidationReport",
    "RepairPlan",
    "RepairResult",
    "PlanningCoreResult",
    "EnrichedItinerary",
    "build_strategy",
    "build_schedule",
    "detect_remote_pois",
    "validate",
    "generate_repairs",
    "apply_repair",
    "run_repair_loop",
    "activity_fields_match",
    "protected_field_differences",
    "protected_fields_match",
    "enrich",
]
