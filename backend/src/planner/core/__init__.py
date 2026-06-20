"""Planning Core — LLM enrichment helpers used by the graph plan node.

Route optimization now lives in `vrp_solver_service` (CP-SAT VRP-TW).
This package only retains the LLM-side helpers actually wired into the graph:
persona feasibility rules, the enrichment writer, and its fact-guard.
"""

from planner.core.enhancements import PersonaRules, feasibility_check
from planner.core.fact_guard import (
    activity_fields_match,
    protected_field_differences,
    protected_fields_match,
)
from planner.core.writer import enrich

__all__ = [
    "PersonaRules",
    "feasibility_check",
    "activity_fields_match",
    "protected_field_differences",
    "protected_fields_match",
    "enrich",
]
