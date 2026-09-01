"""Versioned train/frozen/serve rendering contract for the Qwen3 Agent policy."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


AGENT_RENDER_PROTOCOL_VERSION = "qwen3-agent-prefix-preserving.v1"
AGENT_CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}
AGENT_CHAT_TEMPLATE_SHA256 = (
    "dfe4e379b6439a9f01e881660c5f1ea57cd8026a8831553492723b38d15c9e63"
)
AGENT_CHAT_TEMPLATE_UPSTREAM_SHA256 = (
    "1d944ff8f268b611abb296cdd24d0f51981eef1c8647ac321c3a0258f61eb6c9"
)
AGENT_CHAT_TEMPLATE_PATH = (
    Path(__file__).resolve().parent
    / "templates"
    / "qwen3_agent_prefix_preserving_v1.jinja"
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_agent_chat_template() -> str:
    """Load the immutable renderer asset and fail closed on source drift."""
    payload = AGENT_CHAT_TEMPLATE_PATH.read_bytes()
    observed = _sha256_bytes(payload)
    if observed != AGENT_CHAT_TEMPLATE_SHA256:
        raise RuntimeError(
            "Agent chat template hash mismatch: "
            f"expected {AGENT_CHAT_TEMPLATE_SHA256}, observed {observed}"
        )
    return payload.decode("utf-8")


def install_agent_chat_template(tokenizer: Any) -> str:
    """Install the shared prefix-preserving template on a tokenizer instance."""
    if not getattr(tokenizer, "chat_template", None):
        raise RuntimeError("Agent policy tokenizer must provide a native chat template")
    template = load_agent_chat_template()
    tokenizer.chat_template = template
    return template


def validate_agent_chat_template(template: str | None) -> None:
    """Require a runtime renderer to match the versioned repository asset."""
    if not isinstance(template, str):
        raise RuntimeError("Agent runtime did not expose an active chat template")
    observed = _sha256_bytes(template.encode("utf-8"))
    if observed != AGENT_CHAT_TEMPLATE_SHA256:
        raise RuntimeError(
            "Agent runtime chat template differs from the shared contract: "
            f"expected {AGENT_CHAT_TEMPLATE_SHA256}, observed {observed}"
        )


def render_agent_tool_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    add_generation_prompt: bool,
    tokenize: bool,
    return_dict: bool = False,
    return_tensors: str | None = None,
) -> Any:
    """Render an Agent tool prompt through the sole local-serving contract.

    The tokenizer must already have the versioned template installed.  Keeping
    this tiny wrapper shared between ``LocalCheckpointAgentPolicy`` and the
    actual TRL audit makes train/serve token parity a code-path property rather
    than two hand-copied ``apply_chat_template`` calls.
    """
    validate_agent_chat_template(getattr(tokenizer, "chat_template", None))
    kwargs: dict[str, Any] = {
        "tools": tools,
        "add_generation_prompt": add_generation_prompt,
        "tokenize": tokenize,
        "return_dict": return_dict,
        **AGENT_CHAT_TEMPLATE_KWARGS,
    }
    if return_tensors is not None:
        kwargs["return_tensors"] = return_tensors
    return tokenizer.apply_chat_template(messages, **kwargs)


__all__ = [
    "AGENT_CHAT_TEMPLATE_KWARGS",
    "AGENT_CHAT_TEMPLATE_PATH",
    "AGENT_CHAT_TEMPLATE_SHA256",
    "AGENT_CHAT_TEMPLATE_UPSTREAM_SHA256",
    "AGENT_RENDER_PROTOCOL_VERSION",
    "install_agent_chat_template",
    "load_agent_chat_template",
    "render_agent_tool_prompt",
    "validate_agent_chat_template",
]
