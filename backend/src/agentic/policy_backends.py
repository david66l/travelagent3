"""Concrete policy clients: structured API and native tool-calling.

Production-active: ``NativeToolAgentPolicy`` is the react-mode default;
``ApiAgentPolicy`` serves the structured-JSON API arm.
"""

from __future__ import annotations
import json
from agentic.loop import PolicyAction, PolicyContext
from agentic.policy_actions import (
    controller_override_attempt,
    policy_action_schemas_for_state,
    validate_policy_arguments_for_state,
)
from core.llm_client import LLMClient
from core.settings import settings
from agentic.policy_controller import PolicyOutputError, constrain_policy_context
from agentic.policy_prompts import (
    AGENT_POLICY_SYSTEM_PROMPT,
    AGENT_TOOL_POLICY_SYSTEM_PROMPT,
    policy_prompt_payload,
)
from agentic.policy_routing import PolicyDecision, _last_inference_metrics

class ApiAgentPolicy:
    """Use the existing OpenAI-compatible client as an Agent Loop policy."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    async def propose(self, context: PolicyContext) -> PolicyAction:
        context = constrain_policy_context(context)
        if not context.allowed_actions:
            raise PolicyOutputError(
                "controller supplied no allowed actions",
                code="CONTROLLER_ALLOWLIST_EMPTY",
            )
        try:
            decision = await self.client.structured_call(
                [
                    {"role": "system", "content": AGENT_POLICY_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "context": policy_prompt_payload(context),
                                "action_contracts": policy_action_schemas_for_state(
                                    context.allowed_actions,
                                    capability=context.capability,
                                ),
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                PolicyDecision,
                temperature=0.1,
                task_type="agent_policy",
            )
        except (TypeError, ValueError) as exc:
            raise PolicyOutputError(str(exc), code="POLICY_OUTPUT_MALFORMED") from exc
        if decision.action not in context.allowed_actions:
            raise PolicyOutputError(
                f"policy proposed {decision.action}, allowed: {context.allowed_actions}",
                code="ACTION_NOT_ALLOWED",
            )
        try:
            arguments = validate_policy_arguments_for_state(
                decision.action,
                decision.arguments,
                capability=context.capability,
            )
        except ValueError as exc:
            raise PolicyOutputError(
                str(exc),
                code="ARGUMENT_VALIDATION_FAILED",
                detail_code=getattr(exc, "rejection_code", None),
            ) from exc
        raw_arguments = dict(decision.arguments)
        override_attempt = controller_override_attempt(decision.action, raw_arguments)
        return PolicyAction(
            action=decision.action,
            arguments=arguments,
            model_arguments=raw_arguments,
            controller_override_attempt=override_attempt,
            model_contract_compliant=not override_attempt,
            token_usage=int(getattr(self.client, "last_token_usage", 0) or 0),
            inference_metrics=_last_inference_metrics(self.client),
        )


class NativeToolAgentPolicy:
    """Policy adapter shared by tool-capable API models and local SFT checkpoints."""

    def __init__(
        self,
        client: LLMClient | None = None,
        *,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 256,
        seed: int | None = None,
    ) -> None:
        if temperature < 0:
            raise ValueError("temperature must not be negative")
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        self.client = client or LLMClient()
        self.model = model or settings.agentic_policy_model or None
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed

    def set_rollout_seed(self, seed: int) -> None:
        """Set the checkpoint-independent seed for the next paired rollout."""
        self.seed = seed

    async def propose(self, context: PolicyContext) -> PolicyAction:
        context = constrain_policy_context(context)
        if not context.allowed_actions:
            raise PolicyOutputError(
                "controller supplied no allowed actions",
                code="CONTROLLER_ALLOWLIST_EMPTY",
            )
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
                policy_action_schemas_for_state(
                    context.allowed_actions,
                    capability=context.capability,
                ),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                task_type="agent_policy",
                model_override=self.model,
                seed=self.seed,
            )
            decision = PolicyDecision(**raw)
        except (TypeError, ValueError) as exc:
            raise PolicyOutputError(str(exc), code="POLICY_OUTPUT_MALFORMED") from exc
        if decision.action not in context.allowed_actions:
            raise PolicyOutputError(
                f"policy proposed {decision.action}, allowed: {context.allowed_actions}",
                code="ACTION_NOT_ALLOWED",
            )
        try:
            arguments = validate_policy_arguments_for_state(
                decision.action,
                decision.arguments,
                capability=context.capability,
            )
        except ValueError as exc:
            raise PolicyOutputError(
                str(exc),
                code="ARGUMENT_VALIDATION_FAILED",
                detail_code=getattr(exc, "rejection_code", None),
            ) from exc
        raw_arguments = dict(decision.arguments)
        override_attempt = controller_override_attempt(decision.action, raw_arguments)
        return PolicyAction(
            action=decision.action,
            arguments=arguments,
            model_arguments=raw_arguments,
            controller_override_attempt=override_attempt,
            model_contract_compliant=not override_attempt,
            token_usage=int(getattr(self.client, "last_token_usage", 0) or 0),
            inference_metrics=_last_inference_metrics(self.client),
        )
