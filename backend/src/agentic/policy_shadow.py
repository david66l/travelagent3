"""Shadow double-run comparison policy.

Production canary infrastructure: runs a challenger policy alongside the
champion and records paired traces for promotion decisions.
"""

from __future__ import annotations
import asyncio
from typing import Any
from agentic.loop import PolicyAction, PolicyContext, PolicyShadowTrace
from agentic.policy_routing import (
    _normalize_policy_action,
)

class ShadowComparingAgentPolicy:
    """Run a challenger beside the champion without changing executed actions."""

    def __init__(self, champion: Any, challenger: Any, *, challenger_model: str) -> None:
        if not challenger_model.strip():
            raise ValueError("challenger_model is required")
        self.champion = champion
        self.challenger = challenger
        self.challenger_model = challenger_model

    async def propose(self, context: PolicyContext) -> PolicyAction:
        champion_result, challenger_result = await asyncio.gather(
            self.champion.propose(context),
            self.challenger.propose(context),
            return_exceptions=True,
        )
        if isinstance(champion_result, BaseException):
            raise champion_result
        champion_action = _normalize_policy_action(champion_result)
        if isinstance(challenger_result, BaseException):
            error_code = str(getattr(challenger_result, "code", type(challenger_result).__name__))
            return champion_action.model_copy(
                update={
                    "shadow_trace": PolicyShadowTrace(
                        candidate_model=self.challenger_model,
                        status="failed",
                        error_code=error_code,
                    )
                }
            )
        challenger_action = _normalize_policy_action(challenger_result)
        return champion_action.model_copy(
            update={
                "shadow_trace": PolicyShadowTrace(
                    candidate_model=self.challenger_model,
                    status="completed",
                    action=challenger_action.action,
                    arguments=challenger_action.arguments,
                    token_usage=challenger_action.token_usage,
                    inference_metrics=challenger_action.inference_metrics,
                    route_trace=challenger_action.route_trace,
                )
            }
        )
