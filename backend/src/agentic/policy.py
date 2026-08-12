"""Policy adapters for API teacher models and future local checkpoints."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from agentic.loop import PolicyAction, PolicyContext
from agentic.policy_actions import policy_action_schemas, validate_policy_arguments
from core.llm_client import LLMClient
from core.settings import settings


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

AGENT_TOOL_POLICY_SYSTEM_PROMPT = """You are the action policy inside a bounded
travel-planning agent. Call exactly one of the supplied functions for the current
subtask. Never claim success or that constraints passed; programmatic verifiers
decide that. Use only grounded values in the supplied context. Trusted cities,
facts, matrices, constraints and itineraries are injected by the controller."""


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
                        policy_prompt_payload(context),
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
        try:
            arguments = validate_policy_arguments(decision.action, decision.arguments)
        except ValueError as exc:
            raise PolicyOutputError(str(exc)) from exc
        return PolicyAction(
            action=decision.action,
            arguments=arguments,
            token_usage=int(getattr(self.client, "last_token_usage", 0) or 0),
        )


class NativeToolAgentPolicy:
    """Policy adapter shared by tool-capable API models and local SFT checkpoints."""

    def __init__(self, client: LLMClient | None = None, *, model: str | None = None) -> None:
        self.client = client or LLMClient()
        self.model = model or settings.agentic_policy_model or None

    async def propose(self, context: PolicyContext) -> PolicyAction:
        if not context.allowed_actions:
            raise PolicyOutputError("controller supplied no allowed actions")
        try:
            raw = await self.client.tool_call(
                [
                    {"role": "system", "content": AGENT_TOOL_POLICY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            policy_prompt_payload(context),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                policy_action_schemas(context.allowed_actions),
                task_type="agent_policy",
                model_override=self.model,
            )
            decision = PolicyDecision(**raw)
        except (TypeError, ValueError) as exc:
            raise PolicyOutputError(str(exc)) from exc
        if decision.action not in context.allowed_actions:
            raise PolicyOutputError(
                f"policy proposed {decision.action}, allowed: {context.allowed_actions}"
            )
        try:
            arguments = validate_policy_arguments(decision.action, decision.arguments)
        except ValueError as exc:
            raise PolicyOutputError(str(exc)) from exc
        return PolicyAction(
            action=decision.action,
            arguments=arguments,
            token_usage=int(getattr(self.client, "last_token_usage", 0) or 0),
        )


def policy_prompt_payload(context: PolicyContext) -> dict[str, Any]:
    """Project stable policy-visible state while keeping audit IDs private."""
    payload = context.model_dump(mode="json")
    payload["trajectory_id"] = "[CURRENT_TRAJECTORY]"
    current = payload.get("current_subtask") or {}
    current.pop("updated_at", None)
    current.pop("verifier_evidence_refs", None)
    current.pop("artifact_refs", None)
    payload["current_subtask"] = current
    payload["relevant_fact_refs"] = [
        f"fact:{index}" for index, _ in enumerate(payload.get("relevant_fact_refs") or [])
    ]
    payload["relevant_artifact_refs"] = [
        f"artifact:{index}" for index, _ in enumerate(payload.get("relevant_artifact_refs") or [])
    ]
    for index, fact in enumerate(payload.get("relevant_facts") or []):
        fact["fact_id"] = f"fact:{index}"
    for index, artifact in enumerate(payload.get("relevant_artifacts") or []):
        artifact["artifact_id"] = f"artifact:{index}"
    payload["failure_summary"] = [
        {
            key: value
            for key, value in failure.items()
            if key
            not in {
                "failure_id",
                "action_id",
                "evidence_refs",
                "created_at",
            }
        }
        for failure in payload.get("failure_summary") or []
    ]
    return payload
