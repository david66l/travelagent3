"""Policy adapters for API teacher models and future local checkpoints."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from agentic.loop import PolicyAction, PolicyContext
from core.llm_client import LLMClient


class PolicyOutputError(ValueError):
    """Raised when a policy proposes an action outside controller authority."""


class PolicyDecision(BaseModel):
    action: str
    arguments: dict[str, Any] = Field(default_factory=dict)


AGENT_POLICY_SYSTEM_PROMPT = """You are the action policy inside a bounded travel-planning agent.
Select exactly one action from allowed_actions for the current subtask.
Never claim a task succeeded and never claim constraints passed; programmatic
verifiers decide that. Use only facts present in the supplied context. Return a
compact JSON object with keys action and arguments. Do not add explanations."""


class ApiAgentPolicy:
    """Use the existing OpenAI-compatible client as an Agent Loop policy."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    async def propose(self, context: PolicyContext) -> PolicyAction:
        if not context.allowed_actions:
            raise PolicyOutputError("controller supplied no allowed actions")
        decision = await self.client.structured_call(
            [
                {"role": "system", "content": AGENT_POLICY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        context.model_dump(mode="json"),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                },
            ],
            PolicyDecision,
            temperature=0.1,
            task_type="agent_policy",
        )
        if decision.action not in context.allowed_actions:
            raise PolicyOutputError(
                f"policy proposed {decision.action}, allowed: {context.allowed_actions}"
            )
        return PolicyAction(action=decision.action, arguments=decision.arguments)
