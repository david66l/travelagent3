"""Deterministic tool-call guard with off, shadow and enforce modes."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from tools.tool_definitions import TOOL_NAME_TO_MODEL


GuardMode = Literal["off", "shadow", "enforce"]


class GuardViolation(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class GuardDecision(BaseModel):
    mode: GuardMode
    allowed: bool
    would_block: bool = False
    violations: list[GuardViolation] = Field(default_factory=list)


class GuardContext(BaseModel):
    """Trusted controller context; never populated from model self-reports."""

    allowed_tools: set[str] | None = None
    max_calls: int | None = None
    grounded_values: dict[str, set[str]] = Field(default_factory=dict)
    previous_signatures: set[str] = Field(default_factory=set)


def _parse_call(call: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw = function.get("arguments", "{}")
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return name, None, type(exc).__name__
    return name, args, None


def tool_call_signature(name: str, args: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)}"


class ToolGuard:
    """Evaluate a batch without performing any external action."""

    _GROUNDED_FIELDS = {"city", "date", "check_in", "check_out"}

    def __init__(self, mode: GuardMode = "shadow", max_calls: int = 12) -> None:
        self.mode = mode
        self.max_calls = max_calls

    def evaluate_batch(
        self,
        calls: list[dict[str, Any]],
        context: GuardContext | dict[str, Any] | None = None,
    ) -> list[GuardDecision]:
        trusted = context if isinstance(context, GuardContext) else GuardContext(**(context or {}))
        if self.mode == "off":
            return [GuardDecision(mode="off", allowed=True) for _ in calls]

        decisions: list[GuardDecision] = []
        seen = set(trusted.previous_signatures)
        call_limit = trusted.max_calls if trusted.max_calls is not None else self.max_calls

        for index, call in enumerate(calls):
            violations: list[GuardViolation] = []
            name, args, parse_error = _parse_call(call)

            if index >= call_limit:
                violations.append(
                    GuardViolation(
                        code="CALL_BUDGET_EXCEEDED",
                        message=f"tool call budget exceeded ({call_limit})",
                        details={"index": index, "limit": call_limit},
                    )
                )
            if name not in TOOL_NAME_TO_MODEL:
                violations.append(
                    GuardViolation(code="UNKNOWN_TOOL", message=f"unknown tool: {name}")
                )
            if trusted.allowed_tools is not None and name not in trusted.allowed_tools:
                violations.append(
                    GuardViolation(
                        code="TOOL_NOT_ALLOWED",
                        message=f"tool {name} is not allowed in the current state",
                    )
                )

            if parse_error:
                violations.append(
                    GuardViolation(
                        code="INVALID_ARGUMENTS",
                        message="tool arguments are not valid JSON",
                        details={"error_type": parse_error},
                    )
                )
            elif args is not None and name in TOOL_NAME_TO_MODEL:
                try:
                    TOOL_NAME_TO_MODEL[name].model_validate(args)
                except ValidationError as exc:
                    violations.append(
                        GuardViolation(
                            code="INVALID_ARGUMENTS",
                            message="tool arguments do not match the declared schema",
                            details={"errors": exc.errors(include_url=False)},
                        )
                    )

                signature = tool_call_signature(name, args)
                if signature in seen:
                    violations.append(
                        GuardViolation(
                            code="DUPLICATE_TOOL_CALL",
                            message="identical tool call has already been made",
                        )
                    )
                seen.add(signature)

                for field, allowed_values in trusted.grounded_values.items():
                    if field not in self._GROUNDED_FIELDS or field not in args:
                        continue
                    value = args[field]
                    if value is not None and str(value) not in allowed_values:
                        violations.append(
                            GuardViolation(
                                code="UNGROUNDED_ARGUMENT",
                                message=f"{field} is not grounded in trusted state",
                                details={"field": field, "value": str(value)},
                            )
                        )

            would_block = bool(violations)
            decisions.append(
                GuardDecision(
                    mode=self.mode,
                    allowed=not would_block or self.mode == "shadow",
                    would_block=would_block,
                    violations=violations,
                )
            )
        return decisions
