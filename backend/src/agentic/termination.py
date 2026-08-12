"""Global completion guard for online execution and training episodes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from agentic.guard import GuardMode
from evaluation.validator import ValidationReport


class CompletionBlock(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class CompletionDecision(BaseModel):
    mode: GuardMode
    allowed: bool
    would_block: bool = False
    blocks: list[CompletionBlock] = Field(default_factory=list)
    validator_version: str | None = None


class CompletionGuard:
    """Require a programmatic hard-pass before accepting ``finish``."""

    def __init__(self, mode: GuardMode = "shadow") -> None:
        self.mode = mode

    def evaluate(self, report: ValidationReport | dict[str, Any] | None) -> CompletionDecision:
        if self.mode == "off":
            return CompletionDecision(mode="off", allowed=True)

        blocks: list[CompletionBlock] = []
        parsed: ValidationReport | None = None
        if report is None:
            blocks.append(
                CompletionBlock(
                    code="VALIDATOR_NOT_RUN",
                    message="validate_itinerary must run before finish",
                )
            )
        else:
            parsed = report if isinstance(report, ValidationReport) else ValidationReport(**report)
            if not parsed.hard_pass:
                blocks.append(
                    CompletionBlock(
                        code="HARD_CONSTRAINT_FAILED",
                        message="itinerary has unresolved hard-constraint violations",
                        details={
                            "violation_codes": [
                                violation.code for violation in parsed.hard_violations
                            ]
                        },
                    )
                )

        would_block = bool(blocks)
        return CompletionDecision(
            mode=self.mode,
            allowed=not would_block or self.mode == "shadow",
            would_block=would_block,
            blocks=blocks,
            validator_version=parsed.validator_version if parsed else None,
        )
