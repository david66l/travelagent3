"""Deterministic evaluation shared by serving, datasets and RL rewards.

Agent episode comparison intentionally lives in ``evaluation.agentic_eval``
and is not imported here: the online agent termination gate imports this
package while the offline evaluator imports agent trajectory models.
"""

from evaluation.validator import ItineraryValidator, ValidationReport

__all__ = ["ItineraryValidator", "ValidationReport"]
