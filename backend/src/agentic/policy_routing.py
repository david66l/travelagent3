"""Policy response and metrics types; no strategy routing or teacher fallback."""

from typing import Any
from pydantic import BaseModel, Field
from core.inference_metrics import InferenceMetrics


class PolicyDecision(BaseModel):
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)


def _last_inference_metrics(client: Any) -> InferenceMetrics | None:
    """Return only a completed metrics snapshot from a compatible client."""
    metrics = getattr(client, "last_request_metrics", None)
    return metrics if isinstance(metrics, InferenceMetrics) else None


from agentic.loop import PolicyAction


def _normalize_policy_action(action: Any) -> PolicyAction:
    if isinstance(action, PolicyAction):
        return action
    return PolicyAction(
        action=str(action.action),
        arguments=dict(getattr(action, "arguments", {}) or {}),
    )
