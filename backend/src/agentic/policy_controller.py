"""Policy protocol errors and permission-only context projection."""

from __future__ import annotations

import re
from typing import Any

from agentic.loop import PolicyContext
from agentic.policy_actions import (
    model_visible_policy_actions,
)

_POLICY_ACTION_PATTERN = re.compile(r'"(?:name|action)"\s*:\s*"([^"\\]{1,80})"')
_TOOL_CALL_ENVELOPE_PATTERN = re.compile(r"<tool_call>", re.IGNORECASE)


def summarize_policy_output(raw_output: str | None) -> dict[str, Any] | None:
    """Keep protocol diagnostics without persisting the model's raw response."""
    if not raw_output:
        return None
    actions = _POLICY_ACTION_PATTERN.findall(raw_output)
    envelope_count = len(_TOOL_CALL_ENVELOPE_PATTERN.findall(raw_output))
    call_count = max(envelope_count, len(actions))
    return {
        "tool_call_count": call_count,
        "actions": list(dict.fromkeys(actions))[:8],
    }


class PolicyOutputError(ValueError):
    """Raised when a policy proposes an action outside controller authority."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "POLICY_OUTPUT_ERROR",
        detail_code: str | None = None,
        raw_output: str | None = None,
        output_summary: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.detail_code = detail_code
        # Keep offline diagnostics bounded; callers must not rely on this as
        # part of the online policy contract.
        self.raw_output = raw_output[:2000] if raw_output is not None else None
        self.output_summary = output_summary or summarize_policy_output(self.raw_output)


def constrain_policy_context(context: PolicyContext) -> PolicyContext:
    """Apply permission visibility only; never select strategy or trim missing fields."""
    actions = model_visible_policy_actions(context.allowed_actions, capability=context.capability)
    task = {**context.current_subtask, "allowed_actions": actions}
    return context.model_copy(update={"allowed_actions": actions, "current_subtask": task})
