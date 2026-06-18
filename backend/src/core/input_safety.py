"""Input safety checks (PRD §4.9.1)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from core.exceptions import AppException

# Rule-based prompt-injection patterns (case-insensitive).
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
        r"\bDAN\b",
        r"do\s+anything\s+now",
        r"jailbreak",
        r"you\s+are\s+now\s+in\s+developer\s+mode",
        r"system\s+prompt",
        r"role-?play\s+as",
        r"reveal\s+(the\s+)?(system\s+)?prompt",
        r"sudo\s+",
        r"<\s*script",
    )
)

_GIBBERISH_RE = re.compile(r"^[\W\d\s]{200,}$", re.UNICODE)


class PromptInjectionException(AppException):
    """Blocked user input."""

    def __init__(self, message: str = "检测到异常输入，请用自然的旅行需求描述重新提问。"):
        super().__init__(400, "PROMPT_INJECTION_DETECTED", message)


@dataclass(frozen=True)
class InputSafetyResult:
    allowed: bool
    risk_score: float
    flagged: bool
    reason: Optional[str] = None


def score_prompt_injection(text: str) -> InputSafetyResult:
    """Heuristic injection score without external ML model."""
    if not text or not text.strip():
        return InputSafetyResult(allowed=True, risk_score=0.0, flagged=False)

    normalized = text.strip()
    if len(normalized) > 8000:
        return InputSafetyResult(
            allowed=False,
            risk_score=1.0,
            flagged=True,
            reason="input_too_long",
        )

    for pattern in _INJECTION_PATTERNS:
        if pattern.search(normalized):
            return InputSafetyResult(
                allowed=False,
                risk_score=0.95,
                flagged=True,
                reason=f"pattern:{pattern.pattern}",
            )

    if _GIBBERISH_RE.match(normalized):
        return InputSafetyResult(
            allowed=False,
            risk_score=0.85,
            flagged=True,
            reason="gibberish",
        )

    # Suspicious but allowed — log and lower temperature downstream.
    suspicious_markers = ("developer mode", "ignore instructions", "pretend you are")
    lowered = normalized.lower()
    if any(marker in lowered for marker in suspicious_markers):
        return InputSafetyResult(allowed=True, risk_score=0.6, flagged=True, reason="suspicious")

    return InputSafetyResult(allowed=True, risk_score=0.1, flagged=False)


def validate_user_input(text: str) -> InputSafetyResult:
    """Validate input; raise PromptInjectionException when blocked."""
    result = score_prompt_injection(text)
    if not result.allowed:
        raise PromptInjectionException()
    return result
