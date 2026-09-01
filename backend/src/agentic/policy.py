"""Agent policy layer (façade).

The policy layer is split by responsibility; import symbols from
``agentic.policy`` as before — this module re-exports the public surface:

- ``policy_controller`` — controller-owned transitions and context constraint
- ``policy_repair``     — bounded self-repair wrapper
- ``policy_routing``    — routing decisions and routed/specialist policies
- ``policy_shadow``     — champion/challenger shadow comparison
- ``policy_prompts``    — pinned system prompts and prompt projection
- ``policy_backends``   — API and native tool-calling policy clients

All classes are wired from the composition root in ``agentic.integration``.
"""

from __future__ import annotations

import logging

from agentic.policy_backends import ApiAgentPolicy, NativeToolAgentPolicy
from agentic.policy_controller import (
    CONTROLLER_TASK_ACTIONS,
    ControllerFirstPolicy,
    PolicyOutputError,
    constrain_policy_context,
    controller_policy_action,
)
from agentic.policy_prompts import (
    AGENT_POLICY_SYSTEM_PROMPT,
    AGENT_TOOL_POLICY_SYSTEM_PROMPT,
    minimize_controller_hydrated_payload,
    policy_prompt_payload,
)
from agentic.policy_repair import SelfRepairingAgentPolicy
from agentic.policy_routing import (
    DecisionSpecialistRoutedAgentPolicy,
    PolicyDecision,
    PolicyRouteDecision,
    RoutedAgentPolicy,
    VerifierRepairSpecialistRoutedAgentPolicy,
    is_poi_detail_specialist_state,
    is_verifier_repair_specialist_state,
    route_policy_context,
)
from agentic.policy_shadow import ShadowComparingAgentPolicy

logger = logging.getLogger(__name__)

__all__ = [
    "AGENT_POLICY_SYSTEM_PROMPT",
    "AGENT_TOOL_POLICY_SYSTEM_PROMPT",
    "ApiAgentPolicy",
    "CONTROLLER_TASK_ACTIONS",
    "ControllerFirstPolicy",
    "DecisionSpecialistRoutedAgentPolicy",
    "NativeToolAgentPolicy",
    "PolicyDecision",
    "PolicyOutputError",
    "PolicyRouteDecision",
    "RoutedAgentPolicy",
    "SelfRepairingAgentPolicy",
    "ShadowComparingAgentPolicy",
    "VerifierRepairSpecialistRoutedAgentPolicy",
    "constrain_policy_context",
    "controller_policy_action",
    "is_poi_detail_specialist_state",
    "is_verifier_repair_specialist_state",
    "minimize_controller_hydrated_payload",
    "policy_prompt_payload",
    "route_policy_context",
]
