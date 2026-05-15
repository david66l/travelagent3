"""Planning Core — deterministic draft generation with optional LLM enhancement."""

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
from planner.core.daily_scheduler import build_schedule
from planner.core.rule_validator import validate
from planner.core.llm_strategy import enhance_strategy
from planner.core.repair import generate_repairs, apply_repair, run_repair_loop
from planner.core.fact_checksum import compute_checksum, verify_checksum
from planner.core.writer import enrich, enrich_safe

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
    "validate",
    "enhance_strategy",
    "generate_repairs",
    "apply_repair",
    "run_repair_loop",
    "compute_checksum",
    "verify_checksum",
    "enrich",
    "enrich_safe",
]
